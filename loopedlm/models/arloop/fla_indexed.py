from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.attnres.fused import (
    _build_ptr_table,
    attnres_bwd_kernel_dqdw,
    fused_attnres_fwd,
)
from fla.ops.utils.op import exp


def _pad_ptr_table(
    tensors: Sequence[torch.Tensor], *, min_size: int = 0
) -> tuple[torch.Tensor, ...]:
    bucket = max(8, triton.next_power_of_2(max(len(tensors), min_size)))
    return tuple(tensors) + (tensors[0],) * (bucket - len(tensors))


@torch.library.custom_op(
    "parcae::fla_attnres_indexed_forward_ckpt1",
    mutates_args=(),
    device_types="cuda",
)
def _fla_attnres_indexed_forward_ckpt1(
    query: torch.Tensor,
    residuals: list[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    scale: float,
    has_output_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    res = _build_ptr_table(residuals)
    output, _output_pre, rstd, logit, lse = fused_attnres_fwd(
        q=query,
        residuals=residuals,
        res=res,
        w=rms_weight,
        ow=output_rms_weight if has_output_norm else None,
        eps=rms_eps,
        scale=scale,
        checkpoint_level=1,
    )
    return output, rstd, logit, lse


@_fla_attnres_indexed_forward_ckpt1.register_fake
def _fla_attnres_indexed_forward_ckpt1_fake(
    query: torch.Tensor,
    residuals: list[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    scale: float,
    has_output_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del query, rms_weight, output_rms_weight, rms_eps, scale, has_output_norm
    output = torch.empty_like(residuals[0])
    stats_shape = (len(residuals), *output.shape[:-1])
    rstd = output.new_empty(stats_shape, dtype=torch.float32)
    logit = output.new_empty(stats_shape, dtype=torch.float32)
    lse = output.new_empty(output.shape[:-1], dtype=torch.float32)
    return output, rstd, logit, lse


@triton.jit(do_not_specialize=["L"])
def _attnres_bwd_kernel_dv_indexed(
    q,
    res,
    w,
    ow,
    rstd,
    logit,
    lse,
    do,
    dres,
    grad_current_out,
    dqw,
    dow_partial,
    N,
    L,
    L2: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    FUSE_OWNER: tl.constexpr,
    OWNER_PTR_INDEX: tl.constexpr,
    OWNER_SOURCE_INDEX: tl.constexpr,
):
    """FLA level-1 dV backward with sequential indexed accumulation.

    Sources 0..L-2 point at persistent BF16 source-gradient rows and are
    accumulated in place. Source L-1 is the mutable partial residual and is
    written to a fresh output tensor returned to autograd.
    """
    i_n = tl.program_id(0).to(tl.int64)
    o_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    m_d = o_d < D
    b_qw = (
        tl.load(q + o_d, mask=m_d, other=0.0).to(tl.float32)
        * tl.load(w + o_d, mask=m_d, other=0.0).to(tl.float32)
    )
    b_lse = tl.load(lse + i_n).to(tl.float32)
    b_do = tl.load(do + i_n * D + o_d, mask=m_d, other=0.0).to(tl.float32)

    # Checkpoint level 1 recomputes the mixed residual rather than loading o_pre.
    b_o_pre = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        o_l = (i_l * BL).to(tl.int64) + tl.arange(0, BL)
        m_l = o_l < L
        p_v = res[0] + o_l * 0
        for i in tl.static_range(1, L2):
            p_v = tl.where(o_l == i, res[i], p_v)
        p_v = tl.multiple_of(p_v, 16)
        b_v = tl.load(
            tl.multiple_of(
                p_v[:, None] + (i_n * D + o_d[None, :]),
                (1, 16),
            ),
            mask=m_l[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        b_logit = tl.load(logit + i_n + o_l * N, mask=m_l, other=0.0).to(
            tl.float32
        )
        b_p = tl.where(m_l, exp(b_logit * scale - b_lse), 0.0)
        b_o_pre += tl.sum(b_p[:, None] * b_v, axis=0)

    if HAS_ONORM:
        b_o_rstd = tl.rsqrt(
            tl.sum(tl.where(m_d, b_o_pre * b_o_pre, 0.0), axis=0) / D + eps
        )
        b_ow = tl.load(ow + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_xhat = b_o_pre * b_o_rstd
        b_c1 = (
            tl.sum(tl.where(m_d, b_xhat * b_ow * b_do, 0.0), axis=0) / D
        )
        p_dow = dow_partial + i_n * D + o_d
        tl.store(p_dow, (b_xhat * b_do).to(p_dow.dtype.element_ty), mask=m_d)
        b_do = (b_ow * b_do - b_xhat * b_c1) * b_o_rstd

    if FUSE_OWNER:
        p_owner = dres[0]
        for i in tl.static_range(1, L2):
            if OWNER_PTR_INDEX == i:
                p_owner = dres[i]
        p_owner = tl.multiple_of(p_owner, 16)
        b_owner = tl.load(
            tl.multiple_of(p_owner + i_n * D + o_d, 16),
            mask=m_d,
            other=0.0,
        ).to(tl.float32)
        if OWNER_SOURCE_INDEX >= 0:
            b_current = tl.zeros([BD], dtype=tl.float32)

    b_delta = tl.sum(tl.where(m_d, b_do * b_o_pre, 0.0), axis=0)
    b_dqw = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        o_l = (i_l * BL).to(tl.int64) + tl.arange(0, BL)
        m_l = o_l < L
        m_v = m_l[:, None] & m_d[None, :]
        p_v = res[0] + o_l * 0
        p_dv = dres[0] + o_l * 0
        for i in tl.static_range(1, L2):
            p_v = tl.where(o_l == i, res[i], p_v)
            p_dv = tl.where(o_l == i, dres[i], p_dv)
        p_v = tl.multiple_of(p_v, 16)
        p_dv = tl.multiple_of(p_dv, 16)
        b_v = tl.load(
            tl.multiple_of(
                p_v[:, None] + (i_n * D + o_d[None, :]),
                (1, 16),
            ),
            mask=m_v,
            other=0.0,
        ).to(tl.float32)

        b_rstd = tl.load(rstd + i_n + o_l * N, mask=m_l, other=0.0).to(
            tl.float32
        )
        b_logit = tl.load(logit + i_n + o_l * N, mask=m_l, other=0.0).to(
            tl.float32
        )
        b_p = tl.where(m_l, exp(b_logit * scale - b_lse), 0.0)
        b_dp = tl.sum(b_v * b_do[None, :], axis=1)
        b_ds = b_p * (b_dp - b_delta) * scale
        b_k = b_v * b_rstd[:, None]
        b_dv = b_p[:, None] * b_do[None, :] + (b_ds * b_rstd)[:, None] * (
            b_qw[None, :] - b_k * (b_logit / D)[:, None]
        )

        # The final row is the mutable partial and is a normal autograd output.
        # Earlier rows are persistent source slots. CUDA launches these routing
        # backwards on one stream, so a non-atomic read/add/write is sufficient.
        m_accumulate = m_l & (o_l < L - 1)
        if FUSE_OWNER and OWNER_SOURCE_INDEX >= 0:
            m_owner = m_accumulate & (o_l == OWNER_SOURCE_INDEX)
            m_regular = m_accumulate & ~m_owner
            b_previous = tl.load(
                tl.multiple_of(
                    p_dv[:, None] + (i_n * D + o_d[None, :]),
                    (1, 16),
                ),
                mask=m_regular[:, None] & m_d[None, :],
                other=0.0,
            ).to(tl.float32)
            b_updated = (b_previous + b_dv).to(dres[0].dtype.element_ty)
            tl.store(
                tl.multiple_of(
                    p_dv[:, None] + (i_n * D + o_d[None, :]),
                    (1, 16),
                ),
                b_updated,
                mask=m_regular[:, None] & m_d[None, :],
            )
            b_owner_local = tl.sum(
                tl.where(m_owner[:, None], b_dv, 0.0), axis=0
            )
            b_owner = (b_owner + b_owner_local).to(tl.bfloat16).to(tl.float32)
            b_current += tl.sum(
                tl.where((o_l == L - 1)[:, None], b_dv, 0.0), axis=0
            )
        else:
            b_previous = tl.load(
                tl.multiple_of(
                    p_dv[:, None] + (i_n * D + o_d[None, :]),
                    (1, 16),
                ),
                mask=m_accumulate[:, None] & m_d[None, :],
                other=0.0,
            ).to(tl.float32)
            b_dv = tl.where(m_accumulate[:, None], b_previous + b_dv, b_dv)
            b_store = b_dv.to(dres[0].dtype.element_ty)
            if FUSE_OWNER:
                m_current = m_l & (o_l == L - 1)
                b_current_with_owner = (
                    b_store.to(tl.float32) + b_owner[None, :]
                ).to(dres[0].dtype.element_ty)
                b_store = tl.where(
                    m_current[:, None], b_current_with_owner, b_store
                )
            tl.store(
                tl.multiple_of(
                    p_dv[:, None] + (i_n * D + o_d[None, :]),
                    (1, 16),
                ),
                b_store,
                mask=m_v,
            )
        b_dqw += tl.sum(b_ds[:, None] * b_k, axis=0)

    if FUSE_OWNER and OWNER_SOURCE_INDEX >= 0:
        b_current = b_current.to(tl.bfloat16).to(tl.float32)
        b_current = (b_current + b_owner).to(tl.bfloat16)
        tl.store(
            grad_current_out + i_n * D + o_d,
            b_current,
            mask=m_d,
        )
        if OWNER_SOURCE_INDEX >= 0:
            tl.store(
                p_owner + i_n * D + o_d,
                b_owner.to(tl.bfloat16),
                mask=m_d,
            )
    tl.store(dqw + i_n * D + o_d, b_dqw, mask=m_d)


@torch.library.custom_op(
    "parcae::fla_attnres_backward_indexed",
    mutates_args={"grad_history"},
    device_types="cuda",
)
def _fla_attnres_backward_indexed(
    query: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    residuals: list[torch.Tensor],
    grad_history: list[torch.Tensor],
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    rms_eps: float,
    scale: float,
    has_output_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(residuals) < 2:
        raise ValueError("Indexed FLA backward requires history plus current partial")
    if len(grad_history) != len(residuals) - 1:
        raise ValueError(
            f"Expected {len(residuals) - 1} history gradients, got "
            f"{len(grad_history)}"
        )

    current = residuals[-1]
    l_count = len(residuals)
    n_tokens = current.numel() // current.shape[-1]
    width = current.shape[-1]
    grad_current = torch.empty_like(current)
    grad_targets = [*grad_history, grad_current]
    # Avoid FLA's eager-only data_ptr alignment assertions here: torch.library
    # traces this body once with FakeTensors before lowering the wrapped kernels.
    res = _pad_ptr_table(residuals)
    dres = _pad_ptr_table(grad_targets)

    if has_output_norm:
        dow_partial = torch.empty_like(current, dtype=torch.float32)
        grad_output_rms = torch.empty_like(output_rms_weight)
    else:
        dow_partial = torch.empty_like(current, dtype=torch.float32)
        grad_output_rms = torch.empty_like(output_rms_weight)
    dqw = torch.empty_like(current, dtype=torch.float32)
    grad_query = torch.empty_like(query)
    grad_rms_weight = torch.empty_like(rms_weight)

    l_bucket = max(8, triton.next_power_of_2(l_count))
    indexed_bl = int(os.environ.get("ATTNRES_FLA_INDEXED_BL", "2"))
    indexed_warps = int(os.environ.get("ATTNRES_FLA_INDEXED_WARPS", "4"))
    indexed_stages = int(os.environ.get("ATTNRES_FLA_INDEXED_STAGES", "3"))
    if indexed_bl not in {1, 2, 4, 8}:
        raise ValueError("ATTNRES_FLA_INDEXED_BL must be one of 1, 2, 4, 8")
    if indexed_warps not in {4, 8, 16}:
        raise ValueError("ATTNRES_FLA_INDEXED_WARPS must be one of 4, 8, 16")
    _attnres_bwd_kernel_dv_indexed[(n_tokens,)](
        q=query,
        res=res,
        w=rms_weight,
        ow=output_rms_weight,
        rstd=rstd,
        logit=logit,
        lse=lse,
        do=grad_output,
        dres=dres,
        grad_current_out=grad_current,
        dqw=dqw,
        dow_partial=dow_partial,
        N=n_tokens,
        L=l_count,
        L2=l_bucket,
        D=width,
        eps=rms_eps,
        scale=scale,
        BL=indexed_bl,
        BD=triton.next_power_of_2(width),
        HAS_ONORM=has_output_norm,
        FUSE_OWNER=False,
        OWNER_PTR_INDEX=0,
        OWNER_SOURCE_INDEX=-1,
        num_warps=indexed_warps,
        num_stages=indexed_stages,
    )

    def grid(meta):
        return (triton.cdiv(width, meta["BD"]),)

    attnres_bwd_kernel_dqdw[grid](
        q=query,
        w=rms_weight,
        dqw=dqw,
        dow_partial=dow_partial,
        dq=grad_query,
        dw=grad_rms_weight,
        dow=grad_output_rms,
        N=n_tokens,
        D=width,
        HAS_ONORM=has_output_norm,
    )
    return grad_current, grad_query, grad_rms_weight, grad_output_rms


@_fla_attnres_backward_indexed.register_fake
def _fla_attnres_backward_indexed_fake(
    query: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    residuals: list[torch.Tensor],
    grad_history: list[torch.Tensor],
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    rms_eps: float,
    scale: float,
    has_output_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_history, rstd, logit, lse, grad_output, rms_eps, scale
    grad_current = torch.empty_like(residuals[-1])
    grad_query = torch.empty_like(query)
    grad_rms_weight = torch.empty_like(rms_weight)
    grad_output_rms = (
        torch.empty_like(output_rms_weight)
        if has_output_norm
        else output_rms_weight.new_empty(0)
    )
    return grad_current, grad_query, grad_rms_weight, grad_output_rms


@torch.library.custom_op(
    "parcae::fla_attnres_backward_indexed_owner",
    mutates_args={"grad_slots"},
    device_types="cuda",
)
def _fla_attnres_backward_indexed_owner(
    query: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    residuals: list[torch.Tensor],
    grad_slots: list[torch.Tensor],
    owner_source_index: int,
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    rms_eps: float,
    scale: float,
    has_output_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(residuals) < 2:
        raise ValueError("Owner-fused indexed FLA requires history plus current")
    history_count = len(residuals) - 1
    expected_slots = history_count + int(owner_source_index < 0)
    if len(grad_slots) != expected_slots:
        raise ValueError(
            f"Expected {expected_slots} gradient slots, got {len(grad_slots)}"
        )
    if owner_source_index >= history_count:
        raise ValueError(
            f"Owner source index {owner_source_index} exceeds history count "
            f"{history_count}"
        )

    current = residuals[-1]
    l_count = len(residuals)
    n_tokens = current.numel() // current.shape[-1]
    width = current.shape[-1]
    grad_current = torch.empty_like(current)
    grad_targets = [*grad_slots[:history_count], grad_current]
    owner_ptr_index = owner_source_index
    if owner_source_index < 0:
        owner_ptr_index = len(grad_targets)
        grad_targets.append(grad_slots[-1])
    # The owner-only slot can push dres into a larger power-of-two bucket.
    # Both tables are indexed by the same L2 loop in the Triton kernel.
    res = _pad_ptr_table(residuals, min_size=len(grad_targets))
    dres = _pad_ptr_table(grad_targets)

    dow_partial = torch.empty_like(current, dtype=torch.float32)
    grad_output_rms = torch.empty_like(output_rms_weight)
    dqw = torch.empty_like(current, dtype=torch.float32)
    grad_query = torch.empty_like(query)
    grad_rms_weight = torch.empty_like(rms_weight)

    l_bucket = max(8, triton.next_power_of_2(len(grad_targets)))
    indexed_bl = int(os.environ.get("ATTNRES_FLA_INDEXED_BL", "2"))
    indexed_warps = int(os.environ.get("ATTNRES_FLA_INDEXED_WARPS", "4"))
    indexed_stages = int(os.environ.get("ATTNRES_FLA_INDEXED_STAGES", "3"))
    if indexed_bl not in {1, 2, 4, 8}:
        raise ValueError("ATTNRES_FLA_INDEXED_BL must be one of 1, 2, 4, 8")
    if indexed_warps not in {4, 8, 16}:
        raise ValueError("ATTNRES_FLA_INDEXED_WARPS must be one of 4, 8, 16")
    _attnres_bwd_kernel_dv_indexed[(n_tokens,)](
        q=query,
        res=res,
        w=rms_weight,
        ow=output_rms_weight,
        rstd=rstd,
        logit=logit,
        lse=lse,
        do=grad_output,
        dres=dres,
        grad_current_out=grad_current,
        dqw=dqw,
        dow_partial=dow_partial,
        N=n_tokens,
        L=l_count,
        L2=l_bucket,
        D=width,
        eps=rms_eps,
        scale=scale,
        BL=indexed_bl,
        BD=triton.next_power_of_2(width),
        HAS_ONORM=has_output_norm,
        FUSE_OWNER=True,
        OWNER_PTR_INDEX=owner_ptr_index,
        OWNER_SOURCE_INDEX=owner_source_index,
        num_warps=indexed_warps,
        num_stages=indexed_stages,
    )

    def grid(meta):
        return (triton.cdiv(width, meta["BD"]),)

    attnres_bwd_kernel_dqdw[grid](
        q=query,
        w=rms_weight,
        dqw=dqw,
        dow_partial=dow_partial,
        dq=grad_query,
        dw=grad_rms_weight,
        dow=grad_output_rms,
        N=n_tokens,
        D=width,
        HAS_ONORM=has_output_norm,
    )
    return grad_current, grad_query, grad_rms_weight, grad_output_rms


@_fla_attnres_backward_indexed_owner.register_fake
def _fla_attnres_backward_indexed_owner_fake(
    query: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    residuals: list[torch.Tensor],
    grad_slots: list[torch.Tensor],
    owner_source_index: int,
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    rms_eps: float,
    scale: float,
    has_output_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_slots, owner_source_index, rstd, logit, lse
    del grad_output, rms_eps, scale
    grad_current = torch.empty_like(residuals[-1])
    grad_query = torch.empty_like(query)
    grad_rms_weight = torch.empty_like(rms_weight)
    grad_output_rms = (
        torch.empty_like(output_rms_weight)
        if has_output_norm
        else output_rms_weight.new_empty(0)
    )
    return grad_current, grad_query, grad_rms_weight, grad_output_rms


class _FLAIndexedBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        shared_grad: torch.Tensor,
        source_rows: tuple[int, ...],
        owner_row: int,
        rms_eps: float,
        scale: float,
        has_output_norm: bool,
        *residuals: torch.Tensor,
    ) -> torch.Tensor:
        output, rstd, logit, lse = _fla_attnres_indexed_forward_ckpt1(
            query,
            list(residuals),
            rms_weight,
            output_rms_weight,
            rms_eps,
            scale,
            has_output_norm,
        )
        ctx.save_for_backward(
            query,
            rms_weight,
            output_rms_weight,
            rstd,
            logit,
            lse,
            *residuals,
        )
        ctx.shared_grad = shared_grad
        ctx.source_rows = tuple(int(row) for row in source_rows)
        ctx.owner_row = int(owner_row)
        ctx.rms_eps = float(rms_eps)
        ctx.scale = float(scale)
        ctx.has_output_norm = bool(has_output_norm)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            query,
            rms_weight,
            output_rms_weight,
            rstd,
            logit,
            lse,
            *residuals,
        ) = ctx.saved_tensors
        shared_grad = ctx.shared_grad
        grad_current, grad_query, grad_rms_weight, grad_output_rms = (
            _fla_attnres_backward_indexed(
                query,
                rms_weight,
                output_rms_weight,
                list(residuals),
                [shared_grad[row] for row in ctx.source_rows],
                rstd,
                logit,
                lse,
                grad_output.contiguous(),
                ctx.rms_eps,
                ctx.scale,
                ctx.has_output_norm,
            )
        )

        if ctx.owner_row >= 0:
            # The former fanout kernel only received these two gradients.
            # Preserve its FP32 addition followed by BF16 rounding.
            grad_current = (
                grad_current.float() + shared_grad[ctx.owner_row].float()
            ).to(torch.bfloat16).to(grad_current.dtype)

        grad_residuals = [None] * (len(residuals) - 1) + [grad_current]
        return (
            grad_query,
            grad_rms_weight,
            grad_output_rms if ctx.has_output_norm else None,
            None,
            None,
            None,
            None,
            None,
            None,
            *grad_residuals,
        )


class _FLAIndexedOwnerBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        shared_grad: torch.Tensor,
        source_rows: tuple[int, ...],
        owner_row: int,
        rms_eps: float,
        scale: float,
        has_output_norm: bool,
        *residuals: torch.Tensor,
    ) -> torch.Tensor:
        if owner_row < 0:
            raise ValueError("Owner-fused indexed FLA requires owner_row >= 0")
        output, rstd, logit, lse = _fla_attnres_indexed_forward_ckpt1(
            query,
            list(residuals),
            rms_weight,
            output_rms_weight,
            rms_eps,
            scale,
            has_output_norm,
        )
        ctx.save_for_backward(
            query,
            rms_weight,
            output_rms_weight,
            rstd,
            logit,
            lse,
            *residuals,
        )
        ctx.shared_grad = shared_grad
        ctx.source_rows = tuple(int(row) for row in source_rows)
        ctx.owner_row = int(owner_row)
        ctx.rms_eps = float(rms_eps)
        ctx.scale = float(scale)
        ctx.has_output_norm = bool(has_output_norm)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            query,
            rms_weight,
            output_rms_weight,
            rstd,
            logit,
            lse,
            *residuals,
        ) = ctx.saved_tensors
        grad_slots = [ctx.shared_grad[row] for row in ctx.source_rows]
        try:
            owner_source_index = ctx.source_rows.index(ctx.owner_row)
        except ValueError:
            owner_source_index = -1
            grad_slots.append(ctx.shared_grad[ctx.owner_row])

        grad_current, grad_query, grad_rms_weight, grad_output_rms = (
            _fla_attnres_backward_indexed_owner(
                query,
                rms_weight,
                output_rms_weight,
                list(residuals),
                grad_slots,
                owner_source_index,
                rstd,
                logit,
                lse,
                grad_output.contiguous(),
                ctx.rms_eps,
                ctx.scale,
                ctx.has_output_norm,
            )
        )
        grad_residuals = [None] * (len(residuals) - 1) + [grad_current]
        return (
            grad_query,
            grad_rms_weight,
            grad_output_rms if ctx.has_output_norm else None,
            None,
            None,
            None,
            None,
            None,
            None,
            *grad_residuals,
        )


def fla_attnres_indexed(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    source_rows: Sequence[int],
    owner_row: int,
    shared_grad: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: Optional[torch.Tensor],
    rms_eps: float,
    scale: float = 1.0,
) -> torch.Tensor:
    if len(residuals) < 2:
        raise ValueError("Indexed FLA requires at least history and current partial")
    if len(source_rows) != len(residuals) - 1:
        raise ValueError(
            f"Expected {len(residuals) - 1} source rows, got {len(source_rows)}"
        )
    if shared_grad.dtype != residuals[0].dtype or not shared_grad.is_contiguous():
        raise ValueError("shared_grad must be contiguous and match residual dtype")
    if any(row < 0 or row >= shared_grad.shape[0] for row in source_rows):
        raise ValueError(f"Invalid source rows {tuple(source_rows)}")
    if owner_row < -1 or owner_row >= shared_grad.shape[0]:
        raise ValueError(f"Invalid owner row {owner_row}")

    output_shape = residuals[0].shape
    width = output_shape[-1]
    flat_residuals = tuple(
        residual.reshape(-1, width).contiguous() for residual in residuals
    )
    empty_output_weight = rms_weight.new_empty(0)
    output = _FLAIndexedBackwardFunction.apply(
        query.reshape(-1).contiguous(),
        rms_weight.reshape(-1).contiguous(),
        (
            output_rms_weight.reshape(-1).contiguous()
            if output_rms_weight is not None
            else empty_output_weight
        ),
        shared_grad.reshape(shared_grad.shape[0], -1, width),
        tuple(int(row) for row in source_rows),
        int(owner_row),
        float(rms_eps),
        float(scale),
        output_rms_weight is not None,
        *flat_residuals,
    )
    return output.view(output_shape)


def fla_attnres_indexed_owner(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    source_rows: Sequence[int],
    owner_row: int,
    shared_grad: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: Optional[torch.Tensor],
    rms_eps: float,
    scale: float = 1.0,
) -> torch.Tensor:
    if len(residuals) < 2:
        raise ValueError(
            "Owner-fused indexed FLA requires history and current partial"
        )
    if len(source_rows) != len(residuals) - 1:
        raise ValueError(
            f"Expected {len(residuals) - 1} source rows, got {len(source_rows)}"
        )
    if shared_grad.dtype != residuals[0].dtype or not shared_grad.is_contiguous():
        raise ValueError("shared_grad must be contiguous and match residual dtype")
    if any(row < 0 or row >= shared_grad.shape[0] for row in source_rows):
        raise ValueError(f"Invalid source rows {tuple(source_rows)}")
    if owner_row < 0 or owner_row >= shared_grad.shape[0]:
        raise ValueError(f"Invalid owner row {owner_row}")

    output_shape = residuals[0].shape
    width = output_shape[-1]
    flat_residuals = tuple(
        residual.reshape(-1, width).contiguous() for residual in residuals
    )
    empty_output_weight = rms_weight.new_empty(0)
    output = _FLAIndexedOwnerBackwardFunction.apply(
        query.reshape(-1).contiguous(),
        rms_weight.reshape(-1).contiguous(),
        (
            output_rms_weight.reshape(-1).contiguous()
            if output_rms_weight is not None
            else empty_output_weight
        ),
        shared_grad.reshape(shared_grad.shape[0], -1, width),
        tuple(int(row) for row in source_rows),
        int(owner_row),
        float(rms_eps),
        float(scale),
        output_rms_weight is not None,
        *flat_residuals,
    )
    return output.view(output_shape)


__all__ = [
    "fla_attnres_indexed",
    "fla_attnres_indexed_owner",
]
