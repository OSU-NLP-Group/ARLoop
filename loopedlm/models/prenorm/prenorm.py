from typing import Any, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from loopedlm.models.prenorm.config import PreNormConfig
from loopedlm.modules.utils import precompute_freqs_cis
from loopedlm.modules.mixer import has_ve
from loopedlm.utils.cache import ARLoopDynamicCache


class PreNorm(torch.nn.Module):
    """A fixed number of loops through the same pre-norm block stack."""

    _default_objective = {"ignore_index": -100, "z_regularization": 0.0}

    def __init__(
        self,
        config: PreNormConfig,
        objective: Optional[dict[str, Any]] = None,
        gradient_checkpointing=False,
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
        self.latest_metrics = {}

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.reset_parameters()

        if not config.shared_value_embeddings:
            return

        value_embedding_layers = tuple(self.value_embeds.keys())
        if not value_embedding_layers:
            raise ValueError("shared_value_embeddings requires at least one VE layer")

        shared_embedding = self.value_embeds[value_embedding_layers[0]]
        for layer_idx in value_embedding_layers[1:]:
            self.value_embeds[layer_idx] = shared_embedding
        self.shared_value_embedding_layers = value_embedding_layers


    def _precompute_freqs_cis(self):
        dim = self.config.intermediate_size if self.transformer.h[0].expanded else self.config.n_embd
        if self.config.randomize_positions_from is not None:
            max_length = self.config.randomize_positions_from
        else:
            max_length = self.config.block_size
        return precompute_freqs_cis(
            dim // self.config.num_attention_heads,
            max_length,
            self.config.rope_settings.rope_base,
            self.config.rope_settings.rope_condense_ratio,
        )

    def reset_parameters(self) -> None:
        self.config.init.apply(self.transformer.wte, "embedding")
        self.config.init.apply(self.transformer.ln_f, "normalization")


    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_embeds = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale
        return input_embeds


    def _run_block_stack(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        loop_idx: int,
        past_key_values: Optional[ARLoopDynamicCache] = None,
    ) -> torch.Tensor:
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            block_kwargs = {"ve": ve}
            if past_key_values is not None:
                block_kwargs["past_key_values"] = past_key_values
                block_kwargs["step_idx"] = torch.tensor(
                    loop_idx * self.config.n_layer + i,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            if self.gradient_checkpointing:
                x = self.config.checkpoint(block, x, freqs_cis, attention_mask, **block_kwargs)
            else:
                x = block(x, freqs_cis, attention_mask, **block_kwargs)
        return x

    def _eval_loop_count(self) -> int:
        return self.config.loop_count


    def _run_training_loop_once(self, x, input_ids, freqs_cis, attention_mask,
                                return_hidden_states, selected_layers, hidden_states, layer_counter):
        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            if not self.gradient_checkpointing:
                x = block(x, freqs_cis, attention_mask, ve=ve)
            else:
                x = self.config.checkpoint(block, x, freqs_cis, attention_mask, ve=ve)
            layer_counter += 1
            if return_hidden_states and (selected_layers is None or layer_counter in selected_layers):
                hidden_states.append(x)
        return x, layer_counter


    def _run_loops(self, input_ids, input_embeds, freqs_cis, attention_mask, past_key_values=None):
        x = input_embeds
        for loop_idx in range(self.config.loop_count):
            x = self._run_block_stack(x, input_ids, freqs_cis, attention_mask,
                                     loop_idx, past_key_values=past_key_values)
        return x

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.tie_embeddings:
            head_weight = self.transformer.wte.weight
            logits = torch.matmul(x, head_weight.T).float()
        else:
            head_weight = self.lm_head.weight
            logits = torch.matmul(x, head_weight).float()
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
        selected_layers = set(hidden_state_layers or []) if hidden_state_layers is not None else None
        hidden_states = []
        if return_hidden_states and (selected_layers is None or 0 in selected_layers):
            hidden_states.append(input_embeds)

        x = input_embeds
        collect_loop_hidden_states = return_loop_hidden_states or self.monitoring
        loop_hidden_states = [] if collect_loop_hidden_states else None
        layer_counter = 0
        for _ in range(self.config.loop_count):
            x, layer_counter = self._run_training_loop_once(
                x, input_ids, freqs_cis, attention_mask, return_hidden_states,
                selected_layers, hidden_states, layer_counter,
            )
            if loop_hidden_states is not None:
                loop_hidden_states.append(x)

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
            core_step_range=(0, self._eval_loop_count() * self.config.n_layer),
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
        x = self._run_loops(
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
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_cache: bool = True,
        streamer=None,
        **kwargs,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        past_key_values = self.create_cache() if use_cache else None
        generated = input_ids.clone()
        current_input = input_ids

        for _ in range(max_new_tokens):
            outputs = self.forward_for_generation(
                current_input,
                past_key_values=past_key_values,
            )
            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)
            past_key_values = outputs.get("past_key_values")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                logits[mask.scatter(1, sorted_indices, mask)] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) if do_sample else logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            current_input = next_token if use_cache else generated

            if streamer is not None:
                streamer.put(next_token.cpu())

        if streamer is not None:
            streamer.end()

        return generated

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
