from typing import Any, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from loopedlm.models.arloop.config import ARLoopConfig
from loopedlm.modules.utils import precompute_freqs_cis
from loopedlm.modules.mixer import has_ve
from loopedlm.utils.cache import ARLoopDynamicCache


class ARLoopBase(torch.nn.Module):
    """Shared parameter layout, loss and generation interface for ARLoop."""

    _default_objective = {"ignore_index": -100, "z_regularization": 0.0}

    def __init__(
        self,
        config: ARLoopConfig,
        objective: Optional[dict[str, Any]] = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        objective = objective or self._default_objective
        assert config.padded_vocab_size is not None
        self.config = config
        self.emb_scale = config.init.embedding_scale
        self.transformer = torch.nn.ModuleDict(
            dict(
                wte=torch.nn.Embedding(config.padded_vocab_size, config.n_embd),
                abacus=(
                    torch.nn.Embedding(config.block_size, config.n_embd)
                    if self.config.use_abacus
                    else torch.nn.Identity()
                ),
                h=torch.nn.ModuleList(config.Block(config, layer_id=i) for i in range(config.n_layer)),
                ln_f=config.Norm(config.n_embd, eps=config.norm_eps),
            )
        )
        head_dim = config.n_embd // config.num_attention_heads
        kv_dim = config.num_key_value_heads * head_dim
        self.value_embeds = torch.nn.ModuleDict({
            str(i): torch.nn.Embedding(config.padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if config.use_value_embeddings and has_ve(i, config.n_layer)
        })
        self.attnres_queries = torch.nn.Parameter(torch.zeros(config.effective_sublayers, config.n_embd))
        self.attnres_norms = torch.nn.ModuleList(
            config.Norm(config.n_embd, eps=config.norm_eps) for _ in range(config.effective_sublayers)
        )
        self.final_attnres_query = torch.nn.Parameter(torch.zeros(config.n_embd))
        self.final_attnres_norm = config.Norm(config.n_embd, eps=config.norm_eps)
        self.objective = objective
        self.lm_head = config.Linear(
            config.padded_vocab_size,
            config.n_embd,
            bias=False,
            init_method=config.init.fn("head"),
        )

        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[Tensor] = None
        self.gradient_checkpointing = gradient_checkpointing

        self.step = 0
        self.monitoring = False
        self.latest_metrics: dict[str, torch.Tensor] = {}

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.reset_parameters()

    def _precompute_freqs_cis(self) -> torch.Tensor:
        dim = self.config.intermediate_size if self.transformer.h[0].expanded else self.config.n_embd
        max_length = self.config.randomize_positions_from or self.config.block_size
        return precompute_freqs_cis(
            dim // self.config.num_attention_heads,
            max_length,
            self.config.rope_settings.rope_base,
            self.config.rope_settings.rope_condense_ratio,
        )

    def reset_parameters(self) -> None:
        self.config.init.apply(self.transformer.wte, "embedding")
        self.config.init.apply(self.transformer.ln_f, "normalization")
        for norm in self.attnres_norms:
            self.config.init.apply(norm, "normalization")
        self.config.init.apply(self.final_attnres_norm, "normalization")
        torch.nn.init.zeros_(self.attnres_queries)
        torch.nn.init.zeros_(self.final_attnres_query)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_embeds = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale
        return input_embeds

    @staticmethod
    def _attn_res_values(
        blocks: list[torch.Tensor],
        partial_block: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        if partial_block is None:
            return tuple(blocks)
        return tuple([*blocks, partial_block])

    def _maybe_close_block(
        self,
        blocks: list[torch.Tensor],
        partial_block: Optional[torch.Tensor],
        effective_layer_idx: int,
    ) -> Optional[torch.Tensor]:
        layers_per_attnres_block = self.config.attnres_block_size // 2
        if effective_layer_idx > 0 and effective_layer_idx % layers_per_attnres_block == 0:
            if partial_block is not None:
                blocks.append(partial_block)
            return None
        return partial_block

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.tie_embeddings:
            logits = torch.matmul(x, self.transformer.wte.weight.T).float()
        else:
            logits = torch.matmul(x, self.lm_head.weight).float()
        if self.config.logit_softcap is not None:
            softcap = self.config.logit_softcap
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    def _ce_loss_from_hidden(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        logits = self._compute_logits(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=self.objective.get("ignore_index", -100),
        )
        return loss, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        return_hidden_states: bool = False,
        hidden_state_layers: Optional[list[int]] = None,
        return_loop_hidden_states: bool = False,
    ) -> dict[str, Optional[torch.Tensor]]:
        if self.config.randomize_positions_from is not None and self.training:
            position_ids = torch.sort(
                torch.randint(0, self.config.randomize_positions_from, (input_ids.shape[1],), device=input_ids.device)
            )[0]

        if position_ids is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)

        input_embeds = self._embed_tokens(input_ids)
        collect_loop_hidden_states = return_loop_hidden_states or self.monitoring
        x, hidden_states, loop_hidden_states = self._run_loops(
            input_ids,
            input_embeds,
            freqs_cis,
            attention_mask,
            return_hidden_states=return_hidden_states,
            hidden_state_layers=hidden_state_layers,
            return_loop_hidden_states=collect_loop_hidden_states,
        )
        recurrent_state = x
        previous_recurrent_state = input_embeds
        first_recurrent_block_state = recurrent_state
        if loop_hidden_states:
            first_recurrent_block_state = loop_hidden_states[0]
            if len(loop_hidden_states) > 1:
                previous_recurrent_state = loop_hidden_states[-2].detach()

        x = self.transformer.ln_f(recurrent_state)
        if self.monitoring:
            self.monitor_module(
                x,
                recurrent_state,
                previous_recurrent_state,
                first_recurrent_block_state,
            )

        if labels is not None:
            loss, outputs = self._ce_loss_from_hidden(x, labels)
        else:
            outputs = self._compute_logits(x)
            loss = torch.as_tensor(0.0, device=input_ids.device)

        return {
            "loss": loss,
            "logits": outputs if return_logits else None,
            "log_ppl": loss.detach(),
            "ce_loss": loss.detach(),
            "hidden_states": hidden_states if return_hidden_states else None,
            "loop_hidden_states": loop_hidden_states if return_loop_hidden_states else None,
        }

    def create_cache(self, lookup_strategy: str = "full") -> ARLoopDynamicCache:
        return ARLoopDynamicCache(
            lookup_strategy=lookup_strategy,
            core_step_range=(0, self.config.loop_count * self.config.n_layer),
            n_core=self.config.n_layer,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": kwargs.get("past_key_values"),
        }

    def forward_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[ARLoopDynamicCache] = None,
        **kwargs,
    ) -> dict[str, Any]:
        seq_len = input_ids.shape[1]
        cache_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        freqs_cis = self.freqs_cis[:, cache_len:cache_len + seq_len]

        input_embeds = self._embed_tokens(input_ids)
        x, _, _ = self._run_loops(
            input_ids,
            input_embeds,
            freqs_cis,
            attention_mask,
            past_key_values=past_key_values,
        )
        x = self.transformer.ln_f(x)
        logits = self._compute_logits(x)
        return {
            "logits": logits,
            "past_key_values": past_key_values,
        }

    @torch.no_grad()
    def monitor_module(
        self,
        x_out: torch.Tensor,
        x_rec: torch.Tensor,
        xk: torch.Tensor,
        first_recurrent_block: torch.Tensor,
    ):
        x_out_c = x_out - x_out.mean(dim=-1, keepdim=True)
        normed_x = x_out_c / x_out_c.norm(dim=-1, keepdim=True)
        token_corr = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x_out.shape[1]

        x_rec_c = x_rec - x_rec.mean(dim=-1, keepdim=True)
        normed_x = x_rec_c / x_rec_c.norm(dim=-1, keepdim=True)
        token_corr_rec = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x_rec.shape[1]

        residual = x_rec - xk
        metrics = {
            "last_hidden_token_corr": token_corr,
            "recurrent_state_token_corr": token_corr_rec,
            "last_hidden_norm": x_out.norm(dim=-1).mean(),
            "recurrent_state_norm": x_rec.norm(dim=-1).mean(),
            "recurrent_residual": residual.norm(dim=-1).mean(),
            "rel_residual": (residual.norm(dim=-1) / x_rec.norm(dim=-1)).mean(),
            "first_recurrent_block_norm": first_recurrent_block.norm(dim=-1).mean(),
        }
        self.latest_metrics = metrics
