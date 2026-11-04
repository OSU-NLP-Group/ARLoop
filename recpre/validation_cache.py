"""Materialize and persist repeatable validation batches."""
import fcntl
import os
from pathlib import Path
import torch

_VALIDATION_BATCH_CACHE_ATTR = "_arloop_validation_batch_cache"
_VALIDATION_BATCH_CACHE_SOURCE_ATTR = "_arloop_validation_batch_cache_source"
_VALIDATION_BATCH_CACHE_VERSION = 2


def rank_sequence_budget(total_sequences: int, rank: int, world_size: int) -> int:
    base, remainder = divmod(total_sequences, world_size)
    return base + int(rank < remainder)


def _materialize_validation_batches(
    dataloader, eval_iters: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    iterator = iter(dataloader)
    for batch_index in range(eval_iters):
        try:
            input_ids, labels, _ = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Validation dataloader exhausted after {batch_index} of {eval_iters} requested batches"
            ) from exc
        batches.append((input_ids, labels))
    return batches


def _materialize_validation_sequences(
    dataloader, num_sequences: int
) -> tuple[torch.Tensor, torch.Tensor]:
    input_batches = []
    label_batches = []
    remaining = num_sequences
    iterator = iter(dataloader)
    while remaining > 0:
        try:
            input_ids, labels, _ = next(iterator)
        except StopIteration as exc:
            materialized = num_sequences - remaining
            raise RuntimeError(
                f"Validation dataloader exhausted after {materialized} of "
                f"{num_sequences} requested sequences"
            ) from exc
        take = min(remaining, input_ids.shape[0])
        input_batches.append(input_ids[:take])
        label_batches.append(labels[:take])
        remaining -= take
    return torch.cat(input_batches), torch.cat(label_batches)


def _split_validation_sequences(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (input_ids[start : start + batch_size], labels[start : start + batch_size])
        for start in range(0, input_ids.shape[0], batch_size)
    ]


def _load_persistent_validation_cache(
    cache_path: Path,
    eval_iters: int,
    num_sequences: int | None,
) -> dict | None:
    if not cache_path.is_file():
        return None

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"Ignoring unreadable validation cache {cache_path}: {exc}", flush=True)
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _VALIDATION_BATCH_CACHE_VERSION:
        return None
    if num_sequences is not None:
        if payload.get("num_sequences") != num_sequences:
            return None
        input_ids = payload.get("input_ids")
        labels = payload.get("labels")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return None
        if input_ids.shape[0] != num_sequences or labels.shape[0] != num_sequences:
            return None
        return {"input_ids": input_ids, "labels": labels}

    if payload.get("eval_iters") != eval_iters:
        return None
    batches = payload.get("batches")
    if not isinstance(batches, list) or len(batches) != eval_iters:
        return None
    if any(
        not isinstance(batch, (tuple, list))
        or len(batch) != 2
        or not all(isinstance(tensor, torch.Tensor) for tensor in batch)
        for batch in batches
    ):
        return None
    return {"batches": [(input_ids, labels) for input_ids, labels in batches]}


def _save_persistent_validation_cache(
    cache_path: Path,
    eval_iters: int,
    cache_payload: dict,
    num_sequences: int | None,
) -> None:
    temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "version": _VALIDATION_BATCH_CACHE_VERSION,
                "eval_iters": eval_iters,
                "num_sequences": num_sequences,
                **cache_payload,
            },
            temporary_path,
        )
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def get_validation_batch_cache_source(dataloader) -> str | None:
    return getattr(dataloader, _VALIDATION_BATCH_CACHE_SOURCE_ATTR, None)


def get_cached_validation_batches(
    dataloader,
    eval_iters: int,
    persistent_cache_path: str | Path | None = None,
    *,
    num_sequences: int | None = None,
    batch_size: int | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], bool]:
    """Reuse a fixed validation prefix in memory and, optionally, across runs."""
    if num_sequences is not None and (batch_size is None or batch_size < 1):
        raise ValueError("batch_size must be positive when num_sequences is set")

    cache_key = (eval_iters, num_sequences)
    cached = getattr(dataloader, _VALIDATION_BATCH_CACHE_ATTR, None)
    if cached is not None and cached["cache_key"] == cache_key:
        setattr(dataloader, _VALIDATION_BATCH_CACHE_SOURCE_ATTR, "memory")
        cache_payload = cached["payload"]
        if num_sequences is not None:
            return (
                _split_validation_sequences(
                    cache_payload["input_ids"], cache_payload["labels"], batch_size
                ),
                False,
            )
        return cache_payload["batches"], False

    cache_path = Path(persistent_cache_path) if persistent_cache_path is not None else None
    source = "materialized"
    if cache_path is None:
        if num_sequences is not None:
            input_ids, labels = _materialize_validation_sequences(dataloader, num_sequences)
            cache_payload = {"input_ids": input_ids, "labels": labels}
        else:
            cache_payload = {"batches": _materialize_validation_batches(dataloader, eval_iters)}
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            cache_payload = _load_persistent_validation_cache(
                cache_path, eval_iters, num_sequences
            )
            if cache_payload is None:
                if num_sequences is not None:
                    input_ids, labels = _materialize_validation_sequences(
                        dataloader, num_sequences
                    )
                    cache_payload = {"input_ids": input_ids, "labels": labels}
                else:
                    cache_payload = {
                        "batches": _materialize_validation_batches(dataloader, eval_iters)
                    }
                _save_persistent_validation_cache(
                    cache_path, eval_iters, cache_payload, num_sequences
                )
            else:
                source = "disk"

    setattr(
        dataloader,
        _VALIDATION_BATCH_CACHE_ATTR,
        {"cache_key": cache_key, "payload": cache_payload},
    )
    setattr(dataloader, _VALIDATION_BATCH_CACHE_SOURCE_ATTR, source)
    if num_sequences is not None:
        batches = _split_validation_sequences(
            cache_payload["input_ids"], cache_payload["labels"], batch_size
        )
    else:
        batches = cache_payload["batches"]
    return batches, source == "materialized"
