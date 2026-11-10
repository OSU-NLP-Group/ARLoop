"""Deterministic blending for independently pre-packed binary datasets.

This module is intentionally isolated from the existing training/data-loading
implementation. It builds one global source/sample plan and resolves every
sample through a global, lazy-mmap packed reader. Data-parallel ranks consume
disjoint positions from that shared plan.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from recpre.data_loading_utils import collate_packed_rows
from recpre.global_packed_dataset import GlobalPackedBinaryDataset
from recpre.path_utils import PROJECT_ROOT


logger = logging.getLogger(__name__)

BLEND_INDEX_VERSION = "global-plan-global-packed-ordinal-v3"
SAMPLE_REPEAT_POLICY = "global-source-ordinal-modulo-v3"
DEFAULT_BUILD_BLOCK_SIZE = 1 << 20
DATASET_INDEX_DTYPE = np.dtype("<u2")
DATASET_SAMPLE_INDEX_DTYPE = np.dtype("<u4")


@dataclass(frozen=True)
class BlendIndexPaths:
    root: Path
    dataset_index: Path
    dataset_sample_index: Path
    description: Path
    success: Path


@dataclass(frozen=True)
class BlendIndex:
    dataset_index: np.ndarray
    dataset_sample_index: np.ndarray
    fingerprint: str
    paths: BlendIndexPaths
    cache_hit: bool
    source_counts: tuple[int, ...]


@dataclass(frozen=True)
class BlendStage:
    """One contiguous optimizer-step-aligned section of a blend plan."""

    name: str
    num_samples: int
    weights: tuple[float, ...]


def _normalized_weights(weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Blend weights must be a non-empty one-dimensional list")
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.any(values > 0)
    ):
        raise ValueError(
            "Blend weights must be finite and non-negative, with at least one "
            f"positive value, got {weights}"
        )
    return values / values.sum()


def _apportion_with_capacity(
    expected: np.ndarray,
    total: int,
    capacity: np.ndarray,
) -> np.ndarray:
    """Largest-remainder apportionment capped by remaining capacity."""

    if total < 0:
        raise ValueError(f"Cannot apportion a negative total: {total}")
    capacity = np.asarray(capacity, dtype=np.int64)
    expected = np.asarray(expected, dtype=np.float64)
    if expected.shape != capacity.shape:
        raise ValueError("Expected values and capacity must have the same shape")
    if total > int(capacity.sum()):
        raise ValueError(
            f"Cannot apportion {total} slots into capacity {int(capacity.sum())}"
        )

    counts = np.minimum(np.floor(expected).astype(np.int64), capacity)
    remaining = int(total - counts.sum())
    if remaining == 0:
        return counts

    fractional = expected - np.floor(expected)
    # Stable ordering makes equal fractional remainders deterministic by source id.
    order = np.argsort(-fractional, kind="stable")
    for source_id in order:
        if remaining == 0:
            break
        if counts[source_id] < capacity[source_id]:
            counts[source_id] += 1
            remaining -= 1

    # This is only needed for extreme capacity clipping. It remains deterministic.
    if remaining:
        for source_id in range(len(counts)):
            available = int(capacity[source_id] - counts[source_id])
            take = min(available, remaining)
            counts[source_id] += take
            remaining -= take
            if remaining == 0:
                break
    if remaining:
        raise RuntimeError(f"Failed to apportion {remaining} blend slots")
    return counts


def largest_remainder_counts(total: int, weights: Sequence[float]) -> np.ndarray:
    normalized = _normalized_weights(weights)
    expected = normalized * int(total)
    capacity = np.full(len(normalized), int(total), dtype=np.int64)
    return _apportion_with_capacity(expected, int(total), capacity)


def rank_position_counts(num_samples: int, world_size: int) -> np.ndarray:
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    base, remainder = divmod(int(num_samples), int(world_size))
    counts = np.full(world_size, base, dtype=np.int64)
    counts[:remainder] += 1
    return counts


def validate_source_lengths(
    source_counts: np.ndarray,
    source_lengths: np.ndarray,
    source_names: Sequence[str],
) -> None:
    source_counts = np.asarray(source_counts, dtype=np.int64)
    source_lengths = np.asarray(source_lengths, dtype=np.int64)
    if source_counts.shape != source_lengths.shape:
        raise ValueError(
            "Source plan shape does not match available packed data: "
            f"need {source_counts.shape}, have {source_lengths.shape}"
        )
    required_but_empty = np.flatnonzero((source_counts > 0) & (source_lengths <= 0))
    if required_but_empty.size:
        source_id = int(required_but_empty[0])
        raise ValueError(
            "Blend plan cannot repeat an empty packed source: "
            f"source={source_names[source_id]!r}, "
            f"required={int(source_counts[source_id]):,}, "
            f"available={int(source_lengths[source_id]):,}"
        )
    too_many_planned = np.flatnonzero(
        source_counts > np.iinfo(DATASET_SAMPLE_INDEX_DTYPE).max + 1
    )
    if too_many_planned.size:
        source_id = int(too_many_planned[0])
        raise ValueError(
            "Planned global source ordinals exceed uint32 capacity: "
            f"source={source_names[source_id]!r}, "
            f"required={int(source_counts[source_id]):,}"
        )


def _paths_for_cache(cache_root: Path, fingerprint: str) -> BlendIndexPaths:
    root = cache_root / fingerprint
    return BlendIndexPaths(
        root=root,
        dataset_index=root / "dataset_index.npy",
        dataset_sample_index=root / "dataset_sample_index.npy",
        description=root / "description.json",
        success=root / "_SUCCESS",
    )


def _is_complete_cache(
    paths: BlendIndexPaths,
    fingerprint: str,
    num_samples: int,
) -> bool:
    if not all(
        path.is_file()
        for path in (
            paths.dataset_index,
            paths.dataset_sample_index,
            paths.description,
            paths.success,
        )
    ):
        return False
    try:
        success = json.loads(paths.success.read_text())
        description = json.loads(paths.description.read_text())
        if (
            success.get("version") != BLEND_INDEX_VERSION
            or success.get("fingerprint") != fingerprint
            or description.get("version") != BLEND_INDEX_VERSION
        ):
            return False
        dataset_index = np.load(paths.dataset_index, mmap_mode="r", allow_pickle=False)
        sample_index = np.load(
            paths.dataset_sample_index,
            mmap_mode="r",
            allow_pickle=False,
        )
        return (
            dataset_index.shape == (num_samples,)
            and sample_index.shape == (num_samples,)
            and dataset_index.dtype == DATASET_INDEX_DTYPE
            and sample_index.dtype == DATASET_SAMPLE_INDEX_DTYPE
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _atomic_replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _build_index_files(
    paths: BlendIndexPaths,
    *,
    fingerprint: str,
    description: dict[str, Any],
    num_samples: int,
    stage_weights: Sequence[Sequence[float]],
    stage_source_counts: np.ndarray,
    stage_num_samples: Sequence[int],
    seed: int,
    build_block_size: int,
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp-{os.getpid()}"
    dataset_tmp = Path(f"{paths.dataset_index}{suffix}")
    sample_tmp = Path(f"{paths.dataset_sample_index}{suffix}")
    description_tmp = Path(f"{paths.description}{suffix}")
    success_tmp = Path(f"{paths.success}{suffix}")
    temporary_paths = (dataset_tmp, sample_tmp, description_tmp, success_tmp)
    for path in temporary_paths:
        if path.exists():
            path.unlink()

    normalized_stages = [_normalized_weights(weights) for weights in stage_weights]
    stage_counts_array = np.asarray(stage_source_counts, dtype=np.int64)
    stage_samples_array = np.asarray(stage_num_samples, dtype=np.int64)
    if stage_counts_array.shape != (
        len(normalized_stages),
        len(normalized_stages[0]),
    ):
        raise ValueError("Stage source counts have an unexpected shape")
    if stage_samples_array.shape != (len(normalized_stages),):
        raise ValueError("Stage sample counts have an unexpected shape")
    if int(stage_samples_array.sum()) != int(num_samples):
        raise ValueError("Stage sample counts do not cover the full blend plan")
    rng = np.random.default_rng(seed)
    effective_block_size = max(int(build_block_size), 1)

    dataset_index = np.lib.format.open_memmap(
        dataset_tmp,
        mode="w+",
        dtype=DATASET_INDEX_DTYPE,
        shape=(num_samples,),
    )
    sample_index = np.lib.format.open_memmap(
        sample_tmp,
        mode="w+",
        dtype=DATASET_SAMPLE_INDEX_DTYPE,
        shape=(num_samples,),
    )

    total_source_counts = stage_counts_array.sum(axis=0)
    source_cursors = np.zeros_like(total_source_counts, dtype=np.int64)
    source_ids = np.arange(len(normalized_stages[0]), dtype=DATASET_INDEX_DTYPE)

    try:
        stage_global_start = 0
        for stage_id, stage_size_value in enumerate(stage_samples_array):
            stage_size = int(stage_size_value)
            remaining_counts = stage_counts_array[stage_id].copy()
            remaining_slots = stage_size
            stage_global_end = stage_global_start + stage_size

            # A block never crosses a stage boundary. This makes the requested
            # distribution change exact at the corresponding optimizer step.
            for global_start in range(
                stage_global_start,
                stage_global_end,
                effective_block_size,
            ):
                global_end = min(
                    global_start + effective_block_size,
                    stage_global_end,
                )
                block_length = global_end - global_start
                if block_length == remaining_slots:
                    block_counts = remaining_counts.copy()
                else:
                    expected = remaining_counts.astype(np.float64) * (
                        block_length / remaining_slots
                    )
                    block_counts = _apportion_with_capacity(
                        expected,
                        block_length,
                        remaining_counts,
                    )

                block_dataset = np.repeat(source_ids, block_counts)
                if len(block_dataset) != block_length:
                    raise RuntimeError("Internal blend block size mismatch")
                block_dataset = block_dataset[rng.permutation(block_length)]
                block_sample = np.empty(
                    block_length,
                    dtype=DATASET_SAMPLE_INDEX_DTYPE,
                )

                # Source ordinals remain global across stages and ranks. A
                # stage switch changes selection ratios, never source position.
                for source_id, count_value in enumerate(block_counts):
                    count = int(count_value)
                    if count == 0:
                        continue
                    start = int(source_cursors[source_id])
                    source_positions = np.flatnonzero(block_dataset == source_id)
                    block_sample[source_positions] = np.arange(
                        start,
                        start + count,
                        dtype=DATASET_SAMPLE_INDEX_DTYPE,
                    )
                source_cursors += block_counts
                remaining_counts -= block_counts
                remaining_slots -= block_length

                dataset_index[global_start:global_end] = block_dataset
                sample_index[global_start:global_end] = block_sample

            if np.any(remaining_counts) or remaining_slots:
                raise RuntimeError(
                    f"Blend index construction did not consume stage {stage_id}"
                )
            stage_global_start = stage_global_end

        if stage_global_start != num_samples:
            raise RuntimeError("Blend index construction did not cover its plan")
        if not np.array_equal(source_cursors, total_source_counts):
            raise RuntimeError("Blend sample cursors do not match planned source counts")
        dataset_index.flush()
        sample_index.flush()
        del dataset_index
        del sample_index

        description_tmp.write_text(json.dumps(description, indent=2, sort_keys=True))
        success_tmp.write_text(
            json.dumps(
                {
                    "complete": True,
                    "version": BLEND_INDEX_VERSION,
                    "fingerprint": fingerprint,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        _atomic_replace(dataset_tmp, paths.dataset_index)
        _atomic_replace(sample_tmp, paths.dataset_sample_index)
        _atomic_replace(description_tmp, paths.description)
        _atomic_replace(success_tmp, paths.success)
    except BaseException:
        for path in temporary_paths:
            if path.exists():
                path.unlink()
        raise


def prepare_blend_index(
    *,
    cache_root: str | Path,
    weights: Sequence[float],
    num_samples: int,
    source_lengths: Sequence[int],
    source_names: Sequence[str],
    seed: int,
    sequence_length: int,
    source_descriptions: Sequence[dict[str, Any]],
    source_fingerprints: Sequence[str],
    build_block_size: int = DEFAULT_BUILD_BLOCK_SIZE,
) -> BlendIndex:
    if num_samples < 1:
        raise ValueError(f"Blend plan must contain at least one sample, got {num_samples}")
    if len(source_names) > np.iinfo(DATASET_INDEX_DTYPE).max:
        raise ValueError("Too many sources for uint16 dataset_index")
    if len(weights) != len(source_names):
        raise ValueError("Each blended source must have exactly one weight")

    source_lengths_array = np.asarray(source_lengths, dtype=np.int64)
    if source_lengths_array.shape != (len(source_names),):
        raise ValueError(
            "source_lengths has unexpected shape: "
            f"expected {(len(source_names),)}, got {source_lengths_array.shape}"
        )
    if len(source_fingerprints) != len(source_names):
        raise ValueError("Each blended source must have one catalog fingerprint")

    normalized = _normalized_weights(weights)
    source_counts = largest_remainder_counts(num_samples, normalized)
    validate_source_lengths(
        source_counts,
        source_lengths_array,
        source_names,
    )

    source_wrap_counts = np.zeros_like(source_counts, dtype=np.int64)
    required = source_counts > 0
    source_wrap_counts[required] = (
        source_counts[required] - 1
    ) // source_lengths_array[required]

    description: dict[str, Any] = {
        "version": BLEND_INDEX_VERSION,
        "num_samples": int(num_samples),
        "sequence_length": int(sequence_length),
        "seed": int(seed),
        "weights": [float(value) for value in normalized],
        "source_names": list(source_names),
        "source_counts": [int(value) for value in source_counts],
        "source_lengths": [int(value) for value in source_lengths_array],
        "sample_repeat_policy": SAMPLE_REPEAT_POLICY,
        "source_wrap_counts": [int(value) for value in source_wrap_counts],
        "source_descriptions": list(source_descriptions),
        "source_fingerprints": list(source_fingerprints),
        "dataset_index_dtype": DATASET_INDEX_DTYPE.str,
        "dataset_sample_index_dtype": DATASET_SAMPLE_INDEX_DTYPE.str,
        "build_block_size": int(build_block_size),
    }
    serialized = json.dumps(description, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
    paths = _paths_for_cache(Path(cache_root), fingerprint)
    cache_hit = _is_complete_cache(paths, fingerprint, num_samples)
    if not cache_hit:
        _build_index_files(
            paths,
            fingerprint=fingerprint,
            description=description,
            num_samples=num_samples,
            stage_weights=[normalized],
            stage_source_counts=source_counts[np.newaxis, :],
            stage_num_samples=[num_samples],
            seed=seed,
            build_block_size=build_block_size,
        )
    if not _is_complete_cache(paths, fingerprint, num_samples):
        raise RuntimeError(f"Blend index cache is incomplete under {paths.root}")

    dataset_index = np.load(paths.dataset_index, mmap_mode="r", allow_pickle=False)
    dataset_sample_index = np.load(
        paths.dataset_sample_index,
        mmap_mode="r",
        allow_pickle=False,
    )
    return BlendIndex(
        dataset_index=dataset_index,
        dataset_sample_index=dataset_sample_index,
        fingerprint=fingerprint,
        paths=paths,
        cache_hit=cache_hit,
        source_counts=tuple(int(value) for value in source_counts),
    )


def prepare_staged_blend_index(
    *,
    cache_root: str | Path,
    stages: Sequence[BlendStage],
    source_lengths: Sequence[int],
    source_names: Sequence[str],
    seed: int,
    sequence_length: int,
    source_descriptions: Sequence[dict[str, Any]],
    source_fingerprints: Sequence[str],
    build_block_size: int = DEFAULT_BUILD_BLOCK_SIZE,
) -> BlendIndex:
    """Build one global plan containing contiguous, exact mixture stages."""

    if not stages:
        raise ValueError("A staged blend plan requires at least one stage")
    if len(source_names) > np.iinfo(DATASET_INDEX_DTYPE).max:
        raise ValueError("Too many sources for uint16 dataset_index")
    if len(source_fingerprints) != len(source_names):
        raise ValueError("Each blended source must have one catalog fingerprint")

    source_lengths_array = np.asarray(source_lengths, dtype=np.int64)
    if source_lengths_array.shape != (len(source_names),):
        raise ValueError(
            "source_lengths has unexpected shape: "
            f"expected {(len(source_names),)}, got {source_lengths_array.shape}"
        )

    normalized_stages: list[np.ndarray] = []
    stage_num_samples: list[int] = []
    stage_source_counts: list[np.ndarray] = []
    seen_names: set[str] = set()
    for stage_id, stage in enumerate(stages):
        if not stage.name:
            raise ValueError(f"Blend stage {stage_id} must have a non-empty name")
        if stage.name in seen_names:
            raise ValueError(f"Duplicate blend stage name: {stage.name!r}")
        seen_names.add(stage.name)
        if int(stage.num_samples) < 1:
            raise ValueError(
                f"Blend stage {stage.name!r} must contain at least one sample"
            )
        if len(stage.weights) != len(source_names):
            raise ValueError(
                f"Blend stage {stage.name!r} must provide one weight per source"
            )
        normalized = _normalized_weights(stage.weights)
        normalized_stages.append(normalized)
        stage_num_samples.append(int(stage.num_samples))
        stage_source_counts.append(
            largest_remainder_counts(int(stage.num_samples), normalized)
        )

    stage_counts_array = np.stack(stage_source_counts)
    source_counts = stage_counts_array.sum(axis=0)
    num_samples = int(sum(stage_num_samples))
    validate_source_lengths(source_counts, source_lengths_array, source_names)

    source_wrap_counts = np.zeros_like(source_counts, dtype=np.int64)
    required = source_counts > 0
    source_wrap_counts[required] = (
        source_counts[required] - 1
    ) // source_lengths_array[required]

    stage_descriptions: list[dict[str, Any]] = []
    global_sample_cursor = 0
    source_ordinal_cursor = np.zeros(len(source_names), dtype=np.int64)
    for stage, normalized, counts in zip(
        stages,
        normalized_stages,
        stage_counts_array,
    ):
        global_sample_end = global_sample_cursor + int(stage.num_samples)
        ordinal_end = source_ordinal_cursor + counts
        stage_descriptions.append(
            {
                "name": stage.name,
                "start_sample": int(global_sample_cursor),
                "end_sample": int(global_sample_end),
                "num_samples": int(stage.num_samples),
                "weights": [float(value) for value in normalized],
                "source_counts": [int(value) for value in counts],
                "source_ordinal_start": [
                    int(value) for value in source_ordinal_cursor
                ],
                "source_ordinal_end": [int(value) for value in ordinal_end],
            }
        )
        global_sample_cursor = global_sample_end
        source_ordinal_cursor = ordinal_end

    description: dict[str, Any] = {
        "version": BLEND_INDEX_VERSION,
        "plan_kind": "staged",
        "num_samples": num_samples,
        "sequence_length": int(sequence_length),
        "seed": int(seed),
        "source_names": list(source_names),
        "source_counts": [int(value) for value in source_counts],
        "source_lengths": [int(value) for value in source_lengths_array],
        "sample_repeat_policy": SAMPLE_REPEAT_POLICY,
        "source_wrap_counts": [int(value) for value in source_wrap_counts],
        "stages": stage_descriptions,
        "source_descriptions": list(source_descriptions),
        "source_fingerprints": list(source_fingerprints),
        "dataset_index_dtype": DATASET_INDEX_DTYPE.str,
        "dataset_sample_index_dtype": DATASET_SAMPLE_INDEX_DTYPE.str,
        "build_block_size": int(build_block_size),
    }
    serialized = json.dumps(description, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
    paths = _paths_for_cache(Path(cache_root), fingerprint)
    cache_hit = _is_complete_cache(paths, fingerprint, num_samples)
    if not cache_hit:
        _build_index_files(
            paths,
            fingerprint=fingerprint,
            description=description,
            num_samples=num_samples,
            stage_weights=normalized_stages,
            stage_source_counts=stage_counts_array,
            stage_num_samples=stage_num_samples,
            seed=seed,
            build_block_size=build_block_size,
        )
    if not _is_complete_cache(paths, fingerprint, num_samples):
        raise RuntimeError(f"Blend index cache is incomplete under {paths.root}")

    dataset_index = np.load(paths.dataset_index, mmap_mode="r", allow_pickle=False)
    dataset_sample_index = np.load(
        paths.dataset_sample_index,
        mmap_mode="r",
        allow_pickle=False,
    )
    return BlendIndex(
        dataset_index=dataset_index,
        dataset_sample_index=dataset_sample_index,
        fingerprint=fingerprint,
        paths=paths,
        cache_hit=cache_hit,
        source_counts=tuple(int(value) for value in source_counts),
    )


def compile_blend_stages(
    data_config: Sequence[Any],
    *,
    max_steps: int,
    world_batch_size: int,
) -> tuple[BlendStage, ...]:
    """Compile DataEntry piecewise-constant schedulers into global stages.

    A source may keep a static ``weight`` or use ``scheduler_ratios``:
    ``[[0.6, ["const", 0.7]], [0.8, ["const", 0.6]]]``. Ratios are
    fractions of total optimizer steps, rounded down to step boundaries. All source change
    points are unioned, so every resulting stage has a complete weight vector.
    """

    if max_steps < 1:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if world_batch_size < 1:
        raise ValueError(
            f"world_batch_size must be positive, got {world_batch_size}"
        )

    schedules: list[list[tuple[int, float]]] = []
    boundaries = {0, int(max_steps)}
    for entry in data_config:
        default_weight = float(entry.weight)
        scheduler = entry.scheduler_ratios
        if scheduler is None:
            schedules.append([(0, default_weight)])
            continue

        if not isinstance(scheduler, (list, tuple)) or not scheduler:
            raise ValueError(
                f"Invalid blend scheduler for source {entry.prefix!r}: {scheduler!r}"
            )
        source_schedule: list[tuple[int, float]] = [(0, default_weight)]
        previous_ratio = -1.0
        for interval in scheduler:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(
                    f"Invalid piecewise blend interval for {entry.prefix!r}: {interval!r}"
                )
            ratio, rule = interval
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio < 1:
                raise ValueError(
                    f"Blend stage ratio must be in [0, 1), got {ratio!r}"
                )
            if ratio <= previous_ratio:
                raise ValueError(
                    f"Blend stage ratios must be strictly increasing for {entry.prefix!r}"
                )
            start = int(ratio * max_steps)
            if (
                not isinstance(rule, (list, tuple))
                or len(rule) != 2
                or rule[0] != "const"
            ):
                raise ValueError(
                    "Global blended plans only support piecewise const rules, "
                    f"got {rule!r} for {entry.prefix!r}"
                )
            if start == source_schedule[-1][0]:
                source_schedule[-1] = (start, float(rule[1]))
            else:
                source_schedule.append((start, float(rule[1])))
            boundaries.add(start)
            previous_ratio = ratio
        schedules.append(source_schedule)

    ordered_boundaries = sorted(boundaries)
    stages: list[BlendStage] = []
    for start_step, end_step in zip(ordered_boundaries, ordered_boundaries[1:]):
        weights: list[float] = []
        for source_schedule in schedules:
            active_weight = source_schedule[0][1]
            for schedule_start, value in source_schedule:
                if schedule_start > start_step:
                    break
                active_weight = value
            weights.append(active_weight)
        _normalized_weights(weights)
        stages.append(
            BlendStage(
                name=f"steps-{start_step}-to-{end_step}",
                num_samples=(end_step - start_step) * int(world_batch_size),
                weights=tuple(weights),
            )
        )
    return tuple(stages)


def _planned_optimizer_steps(cfg: Any) -> int:
    if cfg.max_steps is not None:
        max_steps = int(cfg.max_steps)
        if max_steps < 1:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        return max_steps

    max_tokens = int(cfg.max_tokens)
    if max_tokens > 0:
        tokens_per_global_step = int(cfg.world_batch_size) * int(cfg.block_size)
        optimizer_steps = max_tokens // tokens_per_global_step
        if optimizer_steps < 1:
            raise ValueError(
                "max_tokens is too small for one blended global batch: "
                f"max_tokens={max_tokens:,}, "
                f"tokens_per_global_step={tokens_per_global_step:,}"
            )
        return optimizer_steps

    raise RuntimeError(
        "Blended index initialization requires max_steps or max_tokens. "
        "max_flops/token_to_param_ratio are resolved after dataloader "
        "construction by the existing training startup path."
    )


def _distributed_rank_world(default_rank: int, default_world_size: int) -> tuple[int, int]:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(default_rank), int(default_world_size)


def _gather_rank_metadata(
    local_metadata: list[dict[str, Any]],
    world_size: int,
) -> list[list[dict[str, Any]]]:
    if torch.distributed.is_initialized():
        gathered: list[list[dict[str, Any]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_metadata)
        return [row for row in gathered if row is not None]
    if world_size != 1:
        raise RuntimeError(
            "A multi-rank blended dataset requires initialized torch.distributed"
        )
    return [local_metadata]


class BlendedPackedDataset(IterableDataset):
    """Blend globally addressable packed sources using one shared cached plan."""

    requires_resume_batch_skip = False

    def __init__(
        self,
        sources: Sequence[GlobalPackedBinaryDataset],
        *,
        source_names: Sequence[str],
        weights: Sequence[float],
        source_descriptions: Sequence[dict[str, Any]],
        cfg: Any,
        process_rank: int,
        num_processes: int,
        seed: int,
        cache_root: str | Path,
        build_block_size: int = DEFAULT_BUILD_BLOCK_SIZE,
        stages: Sequence[BlendStage] | None = None,
    ) -> None:
        super().__init__()
        if not sources:
            raise ValueError("BlendedPackedDataset requires at least one source")
        if not (len(sources) == len(source_names) == len(weights)):
            raise ValueError("Sources, names, and weights must have matching lengths")
        self.sources = list(sources)
        self.source_names = list(source_names)
        self.weights = [float(weight) for weight in weights]
        self.source_descriptions = list(source_descriptions)
        self.cfg = cfg
        self.process_rank = int(process_rank)
        self.num_processes = int(num_processes)
        self.seed = int(seed)
        self.cache_root = Path(cache_root)
        self.build_block_size = int(build_block_size)
        self.stages = tuple(stages) if stages is not None else None
        if self.stages is not None:
            staged_samples = sum(stage.num_samples for stage in self.stages)
            planned_samples = self._planned_global_samples()
            if staged_samples != planned_samples:
                raise ValueError(
                    "Blend stages do not cover the complete training plan: "
                    f"staged={staged_samples:,}, planned={planned_samples:,}"
                )
            if any(len(stage.weights) != len(self.sources) for stage in self.stages):
                raise ValueError("Every blend stage must provide one weight per source")
        self._blend_index: BlendIndex | None = None
        self._local_cursor = 0

    def _planned_global_samples(self) -> int:
        # train.startup() resolves max_tokens into max_steps only after it has
        # created the dataloader. Mirror that calculation here so a global
        # plan can be built during dataloader construction without changing
        # the existing startup order.
        return _planned_optimizer_steps(self.cfg) * int(self.cfg.world_batch_size)

    def _ensure_indices(self) -> None:
        if self._blend_index is not None:
            return
        rank, world_size = _distributed_rank_world(
            self.process_rank,
            self.num_processes,
        )
        if rank != self.process_rank or world_size != self.num_processes:
            raise RuntimeError(
                "Blended dataset rank/world changed after construction: "
                f"expected {self.process_rank}/{self.num_processes}, "
                f"got {rank}/{world_size}"
            )

        local_metadata = [
            {
                "length": int(len(source)),
                "fingerprint": str(source.catalog_fingerprint),
            }
            for source in self.sources
        ]
        gathered = _gather_rank_metadata(local_metadata, world_size)
        if len(gathered) != world_size:
            raise RuntimeError(
                f"Expected metadata from {world_size} ranks, got {len(gathered)}"
            )
        if any(rank_metadata != gathered[0] for rank_metadata in gathered[1:]):
            raise RuntimeError(
                "Global packed source catalogs differ across ranks; every rank "
                "must see the same files, row counts, and ordering"
            )
        source_lengths = [int(source["length"]) for source in gathered[0]]
        source_fingerprints = [
            str(source["fingerprint"]) for source in gathered[0]
        ]

        common_kwargs = dict(
            cache_root=self.cache_root,
            source_lengths=source_lengths,
            source_names=self.source_names,
            seed=self.seed,
            sequence_length=int(self.cfg.block_size),
            source_descriptions=self.source_descriptions,
            source_fingerprints=source_fingerprints,
            build_block_size=self.build_block_size,
        )
        if self.stages is None:
            prepare_index = prepare_blend_index
            kwargs = dict(
                common_kwargs,
                weights=self.weights,
                num_samples=self._planned_global_samples(),
            )
        else:
            prepare_index = prepare_staged_blend_index
            kwargs = dict(common_kwargs, stages=self.stages)

        error: str | None = None
        blend_index: BlendIndex | None = None
        if rank == 0:
            try:
                blend_index = prepare_index(**kwargs)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
        if torch.distributed.is_initialized():
            status = [error]
            torch.distributed.broadcast_object_list(status, src=0)
            error = status[0]
        if error is not None:
            raise RuntimeError(f"Failed to prepare blended dataset index: {error}")
        if rank != 0:
            blend_index = prepare_index(**kwargs)
        if blend_index is None:
            raise RuntimeError("Blend index preparation returned no result")
        self._blend_index = blend_index

        logger.info(
            "Blended plan rank=%s/%s samples=%s counts=%s weights=%s "
            "cache_hit=%s cache=%s fingerprint=%s",
            rank,
            world_size,
            len(blend_index.dataset_index),
            blend_index.source_counts,
            self.weights if self.stages is None else [stage.weights for stage in self.stages],
            blend_index.cache_hit,
            blend_index.paths.root,
            blend_index.fingerprint,
        )

    @property
    def blend_index(self) -> BlendIndex:
        self._ensure_indices()
        assert self._blend_index is not None
        return self._blend_index

    def __len__(self) -> int:
        num_samples = self._planned_global_samples()
        return int(rank_position_counts(num_samples, self.num_processes)[self.process_rank])

    def __iter__(self):
        blend_index = self.blend_index
        total_global = len(blend_index.dataset_index)
        while True:
            global_position = self.process_rank + self._local_cursor * self.num_processes
            if global_position >= total_global:
                return
            source_id = int(blend_index.dataset_index[global_position])
            source_ordinal = int(blend_index.dataset_sample_index[global_position])
            row = self.sources[source_id][source_ordinal]

            self._local_cursor += 1
            yield row

    def state_dict(self) -> dict[str, Any]:
        blend_index = self.blend_index
        rank, world_size = _distributed_rank_world(
            self.process_rank,
            self.num_processes,
        )
        use_cuda_collective = (
            torch.distributed.is_initialized()
            and torch.distributed.get_backend() == "nccl"
        )
        device = torch.device("cuda", torch.cuda.current_device()) if use_cuda_collective else "cpu"
        local_state = torch.tensor(
            [self._local_cursor],
            dtype=torch.long,
            device=device,
        )
        if torch.distributed.is_initialized():
            gathered = [torch.zeros_like(local_state) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, local_state)
        else:
            gathered = [local_state]
        return {
            "version": BLEND_INDEX_VERSION,
            "world_size": world_size,
            "plan_fingerprint": blend_index.fingerprint,
            "local_cursor": [int(state[0].item()) for state in gathered],
            "rank": rank,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        blend_index = self.blend_index
        rank, world_size = _distributed_rank_world(
            self.process_rank,
            self.num_processes,
        )
        saved_version = state_dict.get("version")
        if saved_version != BLEND_INDEX_VERSION:
            raise ValueError(
                "Only v3 blended data state is supported: "
                f"saved={saved_version!r}, current={BLEND_INDEX_VERSION!r}. "
                "Use resume_data_state=false to start from the v3 plan."
            )
        saved_world_size = int(state_dict.get("world_size", -1))
        if saved_world_size != world_size:
            raise ValueError(
                "Cannot restore blended data state with a different world size: "
                f"saved={saved_world_size}, current={world_size}. "
                "Use resume_data_state=false to start a new data plan."
            )
        saved_fingerprint = state_dict.get("plan_fingerprint")
        if saved_fingerprint != blend_index.fingerprint:
            raise ValueError(
                "Cannot restore blended data state from a different plan: "
                f"saved={saved_fingerprint}, current={blend_index.fingerprint}"
            )
        cursors = state_dict.get("local_cursor", [])
        if len(cursors) != world_size:
            raise ValueError("Blended checkpoint does not contain every rank's cursor")

        self._local_cursor = int(cursors[rank])


def _packed_manifest_description(data_dir: str, source_name: str) -> dict[str, Any]:
    split_path = Path(data_dir).resolve()
    manifest_path = split_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing packed manifest for {source_name}: {manifest_path}")
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    tokenizer_sha256 = manifest.get("source_tokenizer_sha256") or manifest.get(
        "tokenizer_sha256"
    )
    return {
        "name": source_name,
        "data_dir": str(split_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "capacity": int(manifest.get("capacity", -1)),
        "dtype": manifest.get("dtype"),
        "tokenizer_sha256": tokenizer_sha256,
        "algorithm": manifest.get("algorithm"),
    }


def create_blended_dataloader(
    data_config: Sequence[Any],
    batch_size: int,
    fabric: Any,
    *,
    cfg: Any,
    tokenizer: Any,
):
    """Build a stateful global packed loader for multi-source training."""

    if any(entry.type != "packed-bin" for entry in data_config):
        raise ValueError("Blended training currently requires every source to be type=packed-bin")
    if cfg.dataloader_num_workers != 0:
        raise ValueError("Blended packed training requires dataloader_num_workers=0")
    if cfg.batch_size_ramp != 0:
        raise ValueError("Blended packed training does not support batch_size_ramp")
    model_parallel_sizes = (
        int(cfg.fabric.row_tensor_parallel_size),
        int(cfg.fabric.col_tensor_parallel_size),
        int(cfg.fabric.depth_tensor_parallel_size),
    )
    if model_parallel_sizes != (1, 1, 1):
        raise ValueError(
            "Global blended packed training currently requires pure data "
            "parallelism; TP/depth-parallel runs need explicit DP rank/group "
            f"mapping, got sizes={model_parallel_sizes}"
        )
    for entry in data_config:
        if float(entry.weight) < 0:
            raise ValueError(
                f"Blend weight must be non-negative for source {entry.prefix!r}"
            )
        if not entry.data_dir:
            raise ValueError(f"data_dir is required for data source {entry.prefix!r}")

    stages: tuple[BlendStage, ...] | None = None
    if any(entry.scheduler_ratios is not None for entry in data_config):
        stages = compile_blend_stages(
            data_config,
            max_steps=_planned_optimizer_steps(cfg),
            world_batch_size=int(cfg.world_batch_size),
        )

    sources: list[GlobalPackedBinaryDataset] = []
    source_names: list[str] = []
    weights: list[float] = []
    descriptions: list[dict[str, Any]] = []
    permutation_cache_root = PROJECT_ROOT / ".runtime" / "data-cache" / "packed-sequence-shuffle"
    for source_id, entry in enumerate(data_config):
        source_dir = entry.data_dir
        source_name = entry.prefix or f"source-{source_id}"
        source = GlobalPackedBinaryDataset(
            dataset_folder_path=source_dir,
            block_size=cfg.loader_block_size,
            seed=cfg.seed,
            shuffle_filenames=cfg.shuffle_filenames,
            shuffle_sequences=cfg.shuffle_blocks,
            sequence_shuffle_chunk_size=cfg.packed_sequence_shuffle_chunk_size,
            permutation_cache_root=permutation_cache_root,
        )
        description = _packed_manifest_description(source_dir, source_name)
        description.update(
            {
                "catalog_fingerprint": source.catalog_fingerprint,
                "global_rows": len(source),
                "num_files": len(source.files),
            }
        )
        sources.append(source)
        descriptions.append(description)
        source_names.append(source_name)
        weights.append(float(entry.weight))

    catalog_error: str | None = None
    if fabric.global_rank == 0:
        try:
            for source in sources:
                source.validate_files()
        except BaseException as exc:
            catalog_error = f"{type(exc).__name__}: {exc}"
    if torch.distributed.is_initialized():
        status = [catalog_error]
        torch.distributed.broadcast_object_list(status, src=0)
        catalog_error = status[0]
    if catalog_error is not None:
        raise RuntimeError(f"Global packed catalog validation failed: {catalog_error}")

    capacities = {description["capacity"] for description in descriptions}
    dtypes = {description["dtype"] for description in descriptions}
    if capacities != {int(cfg.loader_block_size)}:
        raise ValueError(
            f"Blended sources must all have capacity={cfg.loader_block_size}, got {capacities}"
        )
    if dtypes != {"uint16-le"}:
        raise ValueError(f"Blended sources must all use uint16-le, got {dtypes}")
    cache_root = PROJECT_ROOT / ".runtime" / "data-cache" / "blends"
    combined_dataset = BlendedPackedDataset(
        sources,
        source_names=source_names,
        weights=weights,
        source_descriptions=descriptions,
        cfg=cfg,
        process_rank=fabric.global_rank,
        num_processes=fabric.world_size,
        seed=cfg.seed,
        cache_root=cache_root,
        stages=stages,
    )
    return StatefulDataLoader(
        combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        collate_fn=partial(collate_packed_rows, tokenizer=tokenizer),
        num_workers=0, prefetch_factor=None,
    )
