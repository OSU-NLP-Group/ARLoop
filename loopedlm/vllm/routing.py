"""Inference-only vLLM adapter for the vendored FLA AttnRes forward kernel."""
from collections.abc import Sequence
from typing import List, Optional

import torch


def _attnres_forward_impl(
    query: torch.Tensor,
    residuals: List[torch.Tensor],
    rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    # Installed from third_party/flash-linear-attention by the project setup command.
    # Keep Python pointer handling and Triton autotuning outside Dynamo tracing.
    from fla.ops.attnres.fused import _build_ptr_table, fused_attnres_fwd

    shape = residuals[0].shape
    values = [value.reshape(-1, shape[-1]).contiguous() for value in residuals]
    if values[0].numel() == 0:
        return torch.empty_like(residuals[0])
    output, _, _, _, _ = fused_attnres_fwd(
        q=query.contiguous(),
        residuals=values,
        res=_build_ptr_table(values),
        w=rms_weight.contiguous(),
        ow=None if output_rms_weight is None else output_rms_weight.contiguous(),
        eps=rms_eps,
        scale=1.0,
        checkpoint_level=1,
    )
    return output.view(shape)


@torch.library.custom_op("arloop::vllm_attnres", mutates_args=(), device_types="cuda")
def _attnres_forward(
    query: torch.Tensor,
    residuals: List[torch.Tensor],
    rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    return _attnres_forward_impl(query, residuals, rms_weight, rms_eps, output_rms_weight)


@_attnres_forward.register_fake
def _attnres_forward_fake(query, residuals, rms_weight, rms_eps, output_rms_weight):
    return torch.empty_like(residuals[0], memory_format=torch.contiguous_format)


def attnres_route(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not residuals:
        raise ValueError("AttnRes requires at least one residual value")
    return _attnres_forward(query, list(residuals), rms_weight, rms_eps, output_rms_weight)
