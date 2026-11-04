from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from loopedlm.models.arloop.base import ARLoopBase
from loopedlm.utils.cache import ARLoopDynamicCache


def _load_fla_attnres():
    try:
        from fla.ops.attnres import fused_attnres
    except ImportError as exc:
        raise RuntimeError(
            "The fla-softmax path requires flash-linear-attention on PYTHONPATH. "
            "The pinned experiment source is third_party/flash-linear-attention."
        ) from exc
    return fused_attnres


class AttnResFLASoftmax(ARLoopBase):
    """ARLoop topology using FLA's fused softmax AttnRes operator."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fla_checkpoint_level = int(
            os.environ.get("ATTNRES_FLA_CHECKPOINT_LEVEL", "1")
        )
        if self.fla_checkpoint_level not in {0, 1}:
            raise ValueError("ATTNRES_FLA_CHECKPOINT_LEVEL must be 0 or 1")

    @staticmethod
    def _validate_fla_values(values: Sequence[torch.Tensor]) -> None:
        if not values:
            raise ValueError("FLA AttnRes requires at least one residual value")
        first = values[0]
        for index, value in enumerate(values):
            if value.shape != first.shape:
                raise ValueError(
                    f"FLA residual {index} shape {tuple(value.shape)} differs from "
                    f"{tuple(first.shape)}"
                )
            if value.dtype != first.dtype:
                raise ValueError(
                    "FLA residuals must have one dtype; use the benchmark's "
                    "all-bf16 embedding-output cast instead of per-call copies. "
                    f"Got value 0={first.dtype}, value {index}={value.dtype}."
                )
            if value.stride(-1) != 1:
                raise ValueError(f"FLA residual {index} is not contiguous in D")

    def _attn_res_from_values(
        self,
        query: torch.Tensor,
        norm: torch.nn.Module,
        *values: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_fla_values(values)
        fused_attnres = _load_fla_attnres()
        return fused_attnres(
            query=query,
            residuals=values,
            rms_weight=norm.weight,
            output_rms_weight=None,
            rms_eps=norm.eps,
            scale=1.0,
            return_weights=False,
            checkpoint_level=self.fla_checkpoint_level,
        )


class AttnResFLAFusedNorm(AttnResFLASoftmax):
    """FLA routing fused with the immediately following sublayer RMSNorm."""

    def _select_values(
        self,
        blocks: list[torch.Tensor],
        partial_block: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        values = self._attn_res_values(blocks, partial_block)
        if not self.config.attnres_sliding_window:
            return values
        recent_count = self.config.sw_blocks
        if recent_count is None:
            layers_per_block = self.config.attnres_block_size // 2
            recent_count = self.config.n_layer // layers_per_block + 1
        if len(values) <= recent_count + 1:
            return values
        return (values[0], *values[-recent_count:])

    def _route_normed(
        self,
        sublayer_idx: int,
        output_norm: torch.nn.Module,
        blocks: list[torch.Tensor],
        partial_block: Optional[torch.Tensor],
    ) -> torch.Tensor:
        values = self._select_values(blocks, partial_block)
        return self._route_normed_values(sublayer_idx, output_norm, values)

    def _route_normed_values(
        self, sublayer_idx: int, output_norm: torch.nn.Module, values: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        self._validate_fla_values(values)
        return _load_fla_attnres()(
            query=self.attnres_queries[sublayer_idx],
            residuals=values,
            rms_weight=self.attnres_norms[sublayer_idx].weight,
            output_rms_weight=output_norm.weight,
            rms_eps=self.attnres_norms[sublayer_idx].eps,
            scale=1.0,
            return_weights=False,
            checkpoint_level=self.fla_checkpoint_level,
        )

    def _final_attn_res(
        self,
        blocks: list[torch.Tensor],
        partial_block: Optional[torch.Tensor],
    ) -> torch.Tensor:
        values = self._select_values(blocks, partial_block)
        return self._attn_res_from_values(
            self.final_attnres_query,
            self.final_attnres_norm,
            *values,
        )

    def _sublayer_update(
        self,
        sublayer_idx: int,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[ARLoopDynamicCache],
    ) -> torch.Tensor:
        effective_layer_idx = sublayer_idx // 2
        loop_idx = effective_layer_idx // self.config.n_layer
        layer_idx = effective_layer_idx % self.config.n_layer
        block = self.transformer.h[layer_idx]
        if sublayer_idx % 2:
            return block.mlp(hidden)

        ve = (
            self.value_embeds[str(layer_idx)](input_ids)
            if str(layer_idx) in self.value_embeds
            else None
        )
        block_kwargs: dict[str, Any] = {"ve": ve}
        if past_key_values is not None:
            block_kwargs["past_key_values"] = past_key_values
            block_kwargs["step_idx"] = torch.tensor(
                loop_idx * self.config.n_layer + layer_idx,
                dtype=torch.long,
                device=input_ids.device,
            )
        return block.attn(hidden, freqs_cis, attention_mask, **block_kwargs)

    def _route_and_update(
        self,
        sublayer_idx: int,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[ARLoopDynamicCache],
        *values: torch.Tensor,
    ) -> torch.Tensor:
        block = self.transformer.h[(sublayer_idx // 2) % self.config.n_layer]
        output_norm = block.mlp_norm if sublayer_idx % 2 else block.attn_norm
        hidden = self._route_normed_values(sublayer_idx, output_norm, values)
        return self._sublayer_update(
            sublayer_idx, hidden, input_ids, freqs_cis, attention_mask, past_key_values,
        )

    def _run_loops(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[ARLoopDynamicCache] = None,
        return_hidden_states: bool = False,
        hidden_state_layers: Optional[list[int]] = None,
        return_loop_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[list[torch.Tensor]]]:
        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        if use_checkpoint and past_key_values is not None:
            raise ValueError("Activation checkpointing requires past_key_values=None during training")
        selected_layers = set(hidden_state_layers or []) if hidden_state_layers is not None else None
        hidden_states = [] if return_hidden_states else None
        if hidden_states is not None and (selected_layers is None or 0 in selected_layers):
            hidden_states.append(input_embeds)
        loop_hidden_states = [] if return_loop_hidden_states else None

        blocks = [input_embeds]
        partial_block: Optional[torch.Tensor] = input_embeds
        current_hidden = input_embeds
        sublayer_idx = 0
        effective_layer_idx = 0

        for _loop_idx in range(self.config.loop_count):
            for _block in self.transformer.h:
                for is_mlp in (False, True):
                    values = self._select_values(blocks, partial_block)
                    args = (sublayer_idx, input_ids, freqs_cis, attention_mask, past_key_values, *values)
                    if use_checkpoint:
                        update = checkpoint(self._route_and_update, *args, use_reentrant=False)
                    else:
                        update = self._route_and_update(*args)
                    if not is_mlp:
                        partial_block = self._maybe_close_block(blocks, partial_block, effective_layer_idx)
                    partial_block = update if partial_block is None else partial_block + update
                    current_hidden = partial_block
                    sublayer_idx += 1
                effective_layer_idx += 1

                if hidden_states is not None and (
                    selected_layers is None or effective_layer_idx in selected_layers
                ):
                    hidden_states.append(current_hidden)
            if loop_hidden_states is not None:
                loop_hidden_states.append(current_hidden)

        return self._final_attn_res(blocks, partial_block), hidden_states, loop_hidden_states


class AttnResFLAIndexedBackward(AttnResFLAFusedNorm):
    """FLA checkpoint-level-1 routing with sequential indexed dV writes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fla_checkpoint_level = 1

    def _select_indexed_values(
        self,
        blocks: list[torch.Tensor],
        block_rows: list[int],
        partial_block: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        values = tuple([*blocks, partial_block])
        rows = tuple([*block_rows, -1])
        if not self.config.attnres_sliding_window:
            return values, rows

        recent_count = self.config.sw_blocks
        if recent_count is None:
            layers_per_block = self.config.attnres_block_size // 2
            recent_count = self.config.n_layer // layers_per_block + 1
        if len(values) <= recent_count + 1:
            return values, rows
        selected = (0, *range(len(values) - recent_count, len(values)))
        return (
            tuple(values[index] for index in selected),
            tuple(rows[index] for index in selected),
        )

    def _route_normed_indexed(
        self,
        sublayer_idx: int,
        output_norm: torch.nn.Module,
        blocks: list[torch.Tensor],
        block_rows: list[int],
        partial_block: torch.Tensor,
        shared_grad: torch.Tensor,
        owner_row: int,
    ) -> torch.Tensor:
        from loopedlm.models.arloop.fla_indexed import (
            fla_attnres_indexed,
        )

        values, rows = self._select_indexed_values(
            blocks,
            block_rows,
            partial_block,
        )
        self._validate_fla_values(values)
        if rows[-1] != -1:
            raise RuntimeError("Indexed FLA expects the mutable partial last")
        return fla_attnres_indexed(
            query=self.attnres_queries[sublayer_idx],
            residuals=values,
            source_rows=rows[:-1],
            owner_row=owner_row,
            shared_grad=shared_grad,
            rms_weight=self.attnres_norms[sublayer_idx].weight,
            output_rms_weight=output_norm.weight,
            rms_eps=self.attnres_norms[sublayer_idx].eps,
        )

    def _final_attn_res_indexed(
        self,
        blocks: list[torch.Tensor],
        block_rows: list[int],
        partial_block: torch.Tensor,
        shared_grad: torch.Tensor,
    ) -> torch.Tensor:
        from loopedlm.models.arloop.fla_indexed import (
            fla_attnres_indexed,
        )

        values, rows = self._select_indexed_values(
            blocks,
            block_rows,
            partial_block,
        )
        return fla_attnres_indexed(
            query=self.final_attnres_query,
            residuals=values,
            source_rows=rows[:-1],
            owner_row=-1,
            shared_grad=shared_grad,
            rms_weight=self.final_attnres_norm.weight,
            output_rms_weight=None,
            rms_eps=self.final_attnres_norm.eps,
        )

    def _run_loops(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[ARLoopDynamicCache] = None,
        return_hidden_states: bool = False,
        hidden_state_layers: Optional[list[int]] = None,
        return_loop_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[list[torch.Tensor]]]:
        use_indexed_backward = (
            self.config.attnres_backend == "indexed"
            and self.training
            and torch.is_grad_enabled()
            and not self.gradient_checkpointing
        )
        if not use_indexed_backward:
            return AttnResFLAFusedNorm._run_loops(
                self,
                input_ids,
                input_embeds,
                freqs_cis,
                attention_mask,
                past_key_values,
                return_hidden_states,
                hidden_state_layers,
                return_loop_hidden_states,
            )

        block_size = self.config.attnres_block_size
        if block_size < 2 or block_size % 2:
            raise ValueError(
                "FLA indexed backward requires an even block size of at least 2"
            )
        layers_per_block = block_size // 2
        effective_layers = self.config.n_layer * self.config.loop_count
        shared_rows = 1 + (effective_layers - 1) // layers_per_block
        shared_grad = torch.zeros(
            (shared_rows, *input_embeds.shape),
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        selected_layers = (
            set(hidden_state_layers or [])
            if hidden_state_layers is not None
            else None
        )
        hidden_states = [] if return_hidden_states else None
        if hidden_states is not None and (
            selected_layers is None or 0 in selected_layers
        ):
            hidden_states.append(input_embeds)
        loop_hidden_states = [] if return_loop_hidden_states else None

        blocks = [input_embeds]
        block_rows = [0]
        partial_block = input_embeds
        next_block_row = 1
        sublayer_idx = 0

        for effective_layer_idx in range(effective_layers):
            close_block = (
                effective_layer_idx > 0
                and effective_layer_idx % layers_per_block == 0
            )
            owner_row = (
                0
                if sublayer_idx == 0
                else next_block_row
                if close_block
                else -1
            )
            layer_idx = effective_layer_idx % self.config.n_layer
            block = self.transformer.h[layer_idx]

            hidden = self._route_normed_indexed(
                sublayer_idx,
                block.attn_norm,
                blocks,
                block_rows,
                partial_block,
                shared_grad,
                owner_row,
            )
            if close_block:
                blocks.append(partial_block)
                block_rows.append(next_block_row)
                next_block_row += 1
                previous_partial: Optional[torch.Tensor] = None
            else:
                previous_partial = partial_block

            update = self._sublayer_update(
                sublayer_idx,
                hidden,
                input_ids,
                freqs_cis,
                attention_mask,
                past_key_values,
            )
            partial_block = (
                update if previous_partial is None else previous_partial + update
            )
            sublayer_idx += 1

            hidden = self._route_normed_indexed(
                sublayer_idx,
                block.mlp_norm,
                blocks,
                block_rows,
                partial_block,
                shared_grad,
                -1,
            )
            partial_block = partial_block + self._sublayer_update(
                sublayer_idx,
                hidden,
                input_ids,
                freqs_cis,
                attention_mask,
                past_key_values,
            )
            sublayer_idx += 1

            completed_layer = effective_layer_idx + 1
            if hidden_states is not None and (
                selected_layers is None or completed_layer in selected_layers
            ):
                hidden_states.append(partial_block)
            if (
                loop_hidden_states is not None
                and completed_layer % self.config.n_layer == 0
            ):
                loop_hidden_states.append(partial_block)

        return (
            self._final_attn_res_indexed(
                blocks,
                block_rows,
                partial_block,
                shared_grad,
            ),
            hidden_states,
            loop_hidden_states,
        )


class AttnResFLAIndexedOwnerBackward(AttnResFLAIndexedBackward):
    """Indexed ckpt1 backward with owner finalization fused into dV."""

    def _route_normed_indexed(
        self,
        sublayer_idx: int,
        output_norm: torch.nn.Module,
        blocks: list[torch.Tensor],
        block_rows: list[int],
        partial_block: torch.Tensor,
        shared_grad: torch.Tensor,
        owner_row: int,
    ) -> torch.Tensor:
        if owner_row < 0:
            return super()._route_normed_indexed(
                sublayer_idx,
                output_norm,
                blocks,
                block_rows,
                partial_block,
                shared_grad,
                owner_row,
            )

        from loopedlm.models.arloop.fla_indexed import (
            fla_attnres_indexed_owner,
        )

        values, rows = self._select_indexed_values(
            blocks,
            block_rows,
            partial_block,
        )
        self._validate_fla_values(values)
        if rows[-1] != -1:
            raise RuntimeError("Owner-fused indexed FLA expects current last")
        return fla_attnres_indexed_owner(
            query=self.attnres_queries[sublayer_idx],
            residuals=values,
            source_rows=rows[:-1],
            owner_row=owner_row,
            shared_grad=shared_grad,
            rms_weight=self.attnres_norms[sublayer_idx].weight,
            output_rms_weight=output_norm.weight,
            rms_eps=self.attnres_norms[sublayer_idx].eps,
        )


__all__ = [
    "AttnResFLASoftmax",
    "AttnResFLAFusedNorm",
    "AttnResFLAIndexedBackward",
    "AttnResFLAIndexedOwnerBackward",
]
