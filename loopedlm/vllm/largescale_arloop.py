"""Native vLLM V1 implementation of LargeScaleARLoop.

The training model intentionally remains independent of vLLM. This module is
loaded only through the ``vllm.general_plugins`` entry point and consumes an
normalized configuration and HF or native checkpoint weights.

Physical transformer weights are shared across recurrences. Attention runtime
operators are not shared: each effective ``recurrence x layer`` has a unique
vLLM prefix and therefore its own paged KV-cache slot.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors

from loopedlm.vllm.routing import attnres_route


class LargeScaleARLoopRMSNorm(nn.Module):
    """ARLoop RMSNorm semantics, including FP32 statistics."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_dtype = hidden_states.dtype
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            normalized = hidden_states.float() * torch.rsqrt(
                hidden_states.float().square().mean(dim=-1, keepdim=True) + self.eps
            )
        return normalized.to(dtype=output_dtype) * self.weight


class LargeScaleARLoopMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.mlp_type = getattr(config, "arloop_mlp_type", "swiglu")
        if self.mlp_type not in {"swiglu", "base_relu2"}:
            raise ValueError(f"Unsupported ARLoop vLLM MLP type: {self.mlp_type}")
        fc_multiplier = 2 if self.mlp_type == "swiglu" else 1
        self.fc = nn.Linear(
            config.hidden_size,
            fc_multiplier * config.intermediate_size,
            bias=config.attention_bias,
        )
        self.proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "base_relu2":
            hidden_states = F.relu(self.fc(hidden_states)).square().mul(0.5)
            return self.proj(hidden_states)
        gate, value = self.fc(hidden_states).chunk(2, dim=-1)
        return self.proj(F.silu(gate) * value)


class LargeScaleARLoopProjection(nn.Module):
    """Shared QKV/output projections for one physical transformer layer."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.clip_qkv = config.clip_qkv
        self.qk_norm = config.qk_norm
        self.qk_bias_enabled = config.qk_bias
        physical_layer_count = int(
            getattr(config, "arloop_n_layer", config.num_hidden_layers)
        )
        self.use_value_embeddings = bool(config.use_value_embeddings) and (
            layer_idx % 2 == (physical_layer_count - 1) % 2
        )
        self.ve_gate_channels = 32

        bias = config.attention_bias
        self.c_q = nn.Linear(self.hidden_size, self.q_size, bias=bias)
        self.c_k = nn.Linear(self.hidden_size, self.kv_size, bias=bias)
        self.c_v = nn.Linear(self.hidden_size, self.kv_size, bias=bias)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

        if self.qk_bias_enabled:
            self.qk_bias = nn.Parameter(
                torch.zeros(2, 1, self.num_heads, self.head_dim)
            )
        self.ve_gate = None
        if self.use_value_embeddings:
            self.ve_gate = nn.Linear(
                self.ve_gate_channels,
                self.num_kv_heads,
                bias=False,
            )

    @staticmethod
    def _rms_norm_per_head(value: torch.Tensor) -> torch.Tensor:
        # This deliberately follows modules.mixer.norm, including PyTorch's
        # dtype-dependent default epsilon.
        return F.rms_norm(value, (value.shape[-1],))

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
        value_embedding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        query = self.c_q(hidden_states).view(
            num_tokens, self.num_heads, self.head_dim
        )
        key = self.c_k(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        value = self.c_v(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )

        if value_embedding is not None:
            if self.ve_gate is None:
                raise RuntimeError("Received a value embedding without a VE gate")
            value_embedding = value_embedding.view(
                num_tokens, self.num_kv_heads, self.head_dim
            )
            gate = 2 * torch.sigmoid(
                self.ve_gate(hidden_states[..., : self.ve_gate_channels])
            )
            value = value + gate.unsqueeze(-1) * value_embedding

        if self.clip_qkv is not None:
            query = query.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            key = key.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            value = value.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        if self.qk_bias_enabled:
            query_bias, key_bias = self.qk_bias.split(1, dim=0)
            query = (query + query_bias).to(query.dtype)
            key = (key + key_bias).to(key.dtype)

        return query, key, value

    def normalize_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.qk_norm:
            return query, key
        return self._rms_norm_per_head(query), self._rms_norm_per_head(key)


class LargeScaleARLoopLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = LargeScaleARLoopRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.attn = LargeScaleARLoopProjection(config, layer_idx)
        self.mlp_norm = LargeScaleARLoopRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = LargeScaleARLoopMLP(config)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "num_tokens"},
        "positions": {0: "num_tokens"},
        "intermediate_tensors": {0: "num_tokens"},
        "inputs_embeds": {0: "num_tokens"},
    }
)
class LargeScaleARLoopForCausalLM(nn.Module):
    """Inference-only LargeScaleARLoop model for vLLM V1."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config

        if parallel_config.tensor_parallel_size != 1:
            raise ValueError("LargeScaleARLoop vLLM support currently requires tensor_parallel_size=1")
        if parallel_config.pipeline_parallel_size != 1:
            raise ValueError("LargeScaleARLoop vLLM support currently requires pipeline_parallel_size=1")
        if vllm_config.quant_config is not None:
            raise ValueError("LargeScaleARLoop vLLM support does not yet support quantization")
        if getattr(config, "use_abacus", False):
            raise ValueError("LargeScaleARLoop vLLM support does not support abacus embeddings")
        if getattr(config, "randomize_positions_from", None) is not None:
            raise ValueError("randomize_positions_from is a training-only option and must be disabled")

        self.config = config
        self.physical_layer_count = int(config.arloop_n_layer)
        self.loop_count = int(config.loop_count)
        self.effective_layer_count = self.physical_layer_count * self.loop_count
        expected_sublayers = 2 * self.effective_layer_count
        if int(config.effective_sublayers) != expected_sublayers:
            raise ValueError(
                f"effective_sublayers={config.effective_sublayers} does not match "
                f"2 * loop_count * arloop_n_layer={expected_sublayers}"
            )
        if int(config.num_hidden_layers) != self.effective_layer_count:
            raise ValueError(
                "num_hidden_layers must describe effective cache layers: "
                f"expected {self.effective_layer_count}, got {config.num_hidden_layers}"
            )
        if config.qk_bias and config.num_attention_heads != config.num_key_value_heads:
            raise ValueError("qk_bias with GQA is not supported by the original LargeScaleARLoop layout")

        self.embedding_scale = float(config.embedding_scale)
        padded_vocab_size = int(config.padded_vocab_size)
        original_vocab_size = int(config.vocab_size)
        hidden_size = int(config.hidden_size)
        kv_size = int(config.num_key_value_heads * config.head_dim)

        self.transformer = nn.ModuleDict(
            {
                "wte": VocabParallelEmbedding(
                    padded_vocab_size,
                    hidden_size,
                    org_num_embeddings=original_vocab_size,
                    padding_size=int(config.padding_multiple),
                    prefix=self._with_prefix(prefix, "transformer.wte"),
                ),
                "h": nn.ModuleList(
                    LargeScaleARLoopLayer(config, layer_idx)
                    for layer_idx in range(self.physical_layer_count)
                ),
                "ln_f": LargeScaleARLoopRMSNorm(hidden_size, config.rms_norm_eps),
            }
        )

        self.value_embeds = nn.ModuleDict()
        self.value_embedding_keys: dict[int, str] = {}
        if config.use_value_embeddings:
            value_embedding_layers = [
                layer_idx
                for layer_idx in range(self.physical_layer_count)
                if self._has_value_embedding(layer_idx, self.physical_layer_count)
            ]
            shared = bool(getattr(config, "share_value_embeddings", False))
            shared_key = str(value_embedding_layers[0]) if shared else None
            for layer_idx in value_embedding_layers:
                key = shared_key if shared_key is not None else str(layer_idx)
                self.value_embedding_keys[layer_idx] = key
                if key not in self.value_embeds:
                    self.value_embeds[key] = VocabParallelEmbedding(
                        padded_vocab_size,
                        kv_size,
                        org_num_embeddings=original_vocab_size,
                        padding_size=int(config.padding_multiple),
                        prefix=self._with_prefix(prefix, f"value_embeds.{key}"),
                    )

        self.attnres_queries = nn.Parameter(torch.empty(expected_sublayers, hidden_size))
        self.attnres_norms = nn.ModuleList(
            LargeScaleARLoopRMSNorm(hidden_size, config.rms_norm_eps)
            for _ in range(expected_sublayers)
        )
        self.final_attnres_query = nn.Parameter(torch.empty(hidden_size))
        self.final_attnres_norm = LargeScaleARLoopRMSNorm(
            hidden_size, config.rms_norm_eps
        )

        self.runtime_attn = nn.ModuleList(
            Attention(
                config.num_attention_heads,
                config.head_dim,
                config.head_dim**-0.5,
                num_kv_heads=config.num_key_value_heads,
                cache_config=vllm_config.cache_config,
                quant_config=None,
                prefix=self._with_prefix(prefix, f"runtime_attn.{effective_idx}"),
            )
            for effective_idx in range(self.effective_layer_count)
        )

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = rope_parameters.get("rope_theta", 10_000.0)
        rope = self._precompute_rope(
            int(config.max_position_embeddings),
            int(config.head_dim),
            float(rope_theta),
            int(config.rope_condense_ratio),
        )
        self.register_buffer("freqs_cis", rope, persistent=False)

        self.lm_head = ParallelLMHead(
            padded_vocab_size,
            hidden_size,
            org_num_embeddings=original_vocab_size,
            padding_size=int(config.padding_multiple),
            prefix=self._with_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.transformer.wte)
        self.logits_processor = LogitsProcessor(
            padded_vocab_size,
            org_vocab_size=original_vocab_size,
            scale=float(config.logit_scale),
            soft_cap=config.logit_softcap,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], hidden_size
        )

    @staticmethod
    def _with_prefix(prefix: str, name: str) -> str:
        return name if not prefix else f"{prefix}.{name}"

    @staticmethod
    def _has_value_embedding(layer_idx: int, layer_count: int) -> bool:
        return layer_idx % 2 == (layer_count - 1) % 2

    @staticmethod
    def _precompute_rope(
        max_positions: int,
        head_dim: int,
        theta: float,
        condense_ratio: int,
    ) -> torch.Tensor:
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_positions, dtype=torch.float32) / condense_ratio
        frequencies = torch.outer(positions, inv_freq)
        return torch.stack((frequencies.cos(), frequencies.sin()), dim=-1)

    def _apply_rope(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = self.freqs_cis.index_select(0, positions).unsqueeze(1)
        output_dtype = query.dtype
        query_heads = query.shape[1]
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_key = torch.cat((query, key), dim=1)
            paired = query_key.unflatten(dim=-1, sizes=(-1, 2)).float()
            rotated = torch.stack(
                (
                    paired[..., 0] * frequencies[..., 0]
                    - paired[..., 1] * frequencies[..., 1],
                    paired[..., 1] * frequencies[..., 0]
                    + paired[..., 0] * frequencies[..., 1],
                ),
                dim=-1,
            ).flatten(-2)
        query, key = rotated.to(dtype=output_dtype).split(
            (query_heads, key.shape[1]), dim=1
        )
        return query, key

    def _route(
        self,
        query: torch.Tensor,
        norm: LargeScaleARLoopRMSNorm,
        values: tuple[torch.Tensor, ...],
        output_norm: LargeScaleARLoopRMSNorm | None = None,
    ) -> torch.Tensor:
        return attnres_route(
            query=query,
            residuals=values,
            rms_weight=norm.weight,
            rms_eps=norm.eps,
            output_rms_weight=output_norm.weight if output_norm is not None else None,
        )

    @staticmethod
    def _values(
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        if partial_block is None:
            return tuple(blocks)
        return (*blocks, partial_block)

    def _sliding_values(
        self, values: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        if not self.config.attnres_sliding_window:
            return values
        layers_per_block = self.config.attnres_block_size // 2
        blocks_per_loop = self.physical_layer_count // layers_per_block
        recent_count = self.config.sw_blocks
        if recent_count is None:
            recent_count = blocks_per_loop + 1
        if len(values) <= recent_count + 1:
            return values
        return (values[0], *values[-recent_count:])

    def _route_sublayer(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        sublayer_idx: int,
        output_norm: LargeScaleARLoopRMSNorm,
    ) -> torch.Tensor:
        values = self._sliding_values(self._values(blocks, partial_block))
        return self._route(
            self.attnres_queries[sublayer_idx],
            self.attnres_norms[sublayer_idx],
            values,
            output_norm,
        )

    def _route_final(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
    ) -> torch.Tensor:
        values = self._sliding_values(self._values(blocks, partial_block))
        return self._route(
            self.final_attnres_query,
            self.final_attnres_norm,
            values,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.transformer.wte(input_ids)
        if self.embedding_scale != 1.0:
            embeddings = embeddings * self.embedding_scale
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("Pipeline parallelism is not supported")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            input_embeds = self.embed_input_ids(input_ids)
        else:
            input_embeds = inputs_embeds
        if input_ids is None and self.config.use_value_embeddings:
            raise ValueError("input_ids are required when value embeddings are enabled")

        full_mode = self.config.attnres_block_size == 1
        blocks = [input_embeds]
        partial_block = None if full_mode else input_embeds
        sublayer_idx = 0
        effective_layer_idx = 0
        layers_per_attnres_block = self.config.attnres_block_size // 2

        for _loop_idx in range(self.loop_count):
            for layer_idx, layer in enumerate(self.transformer.h):
                hidden_states = self._route_sublayer(
                    blocks, partial_block, sublayer_idx, layer.attn_norm
                )
                previous_partial = partial_block
                if (
                    not full_mode
                    and effective_layer_idx > 0
                    and effective_layer_idx % layers_per_attnres_block == 0
                ):
                    if partial_block is not None:
                        blocks.append(partial_block)
                    previous_partial = None

                value_embedding = None
                value_embedding_key = self.value_embedding_keys.get(layer_idx)
                if input_ids is not None and value_embedding_key is not None:
                    value_embedding = self.value_embeds[value_embedding_key](input_ids)
                query, key, value = layer.attn.project_qkv(
                    hidden_states, value_embedding
                )
                query, key = self._apply_rope(positions, query, key)
                query, key = layer.attn.normalize_qk(query, key)
                attention_output = self.runtime_attn[effective_layer_idx](
                    query.flatten(1), key.flatten(1), value.flatten(1)
                )
                attention_output = layer.attn.c_proj(attention_output)
                partial_block = (
                    attention_output
                    if previous_partial is None
                    else previous_partial + attention_output
                )
                if full_mode:
                    blocks.append(partial_block)
                    partial_block = None
                sublayer_idx += 1

                hidden_states = self._route_sublayer(
                    blocks, partial_block, sublayer_idx, layer.mlp_norm
                )
                mlp_output = layer.mlp(hidden_states)
                partial_block = (
                    mlp_output if partial_block is None else partial_block + mlp_output
                )
                if full_mode:
                    blocks.append(partial_block)
                    partial_block = None
                sublayer_idx += 1
                effective_layer_idx += 1

        hidden_states = self._route_final(blocks, partial_block)
        return self.transformer.ln_f(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tie_embeddings = bool(self.config.tie_word_embeddings)

        for source_name, source_weight in weights:
            name = source_name
            while name.startswith(("module.", "_forward_module.")):
                name = name.split(".", 1)[1]
            if name == "shared_value_embedding.weight":
                if not getattr(self.config, "share_value_embeddings", False):
                    raise ValueError("Shared HF value embeddings require share_value_embeddings")
                first_layer = (self.physical_layer_count - 1) % 2
                name = f"value_embeds.{first_layer}.weight"
            if name == "freqs_cis":
                continue
            if name == "lm_head.weight":
                if tie_embeddings:
                    continue
                source_weight = source_weight.transpose(0, 1).contiguous()
            shared_value_embeddings = bool(
                getattr(self.config, "share_value_embeddings", False)
            )
            if shared_value_embeddings and name.startswith("value_embeds."):
                parts = name.split(".")
                if len(parts) == 3 and parts[2] == "weight":
                    source_layer = int(parts[1])
                    shared_key = self.value_embedding_keys.get(source_layer)
                    if shared_key is not None:
                        name = f"value_embeds.{shared_key}.weight"

            param = params.get(name)
            if param is None:
                raise ValueError(f"Unexpected LargeScaleARLoop checkpoint weight: {name}")
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, source_weight)
            loaded.add(name)

        required = set(params)
        if tie_embeddings:
            required.discard("lm_head.weight")
        missing = sorted(required - loaded)
        if missing:
            raise ValueError(f"LargeScaleARLoop checkpoint is missing weights: {missing[:12]}")
        return loaded


__all__ = ["LargeScaleARLoopForCausalLM"]
