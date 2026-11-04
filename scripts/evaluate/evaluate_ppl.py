#!/usr/bin/env python3
"""Evaluate native checkpoints on an explicitly supplied validation cache or packed rows."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from loopedlm.evaluation.checkpoint import load_checkpoint


def validation_batches(
    cache_paths, packed_paths, sequence_length, batch_size, num_sequences
):
    remaining = num_sequences
    for path in cache_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        pairs = (
            [(payload["input_ids"], payload["labels"])]
            if "input_ids" in payload
            else payload["batches"]
        )
        for inputs, labels in pairs:
            if (
                inputs.shape != labels.shape
                or inputs.ndim != 2
                or inputs.shape[1] != sequence_length
            ):
                raise ValueError(f"Invalid validation cache shape in {path}")
            for start in range(0, len(inputs), batch_size):
                count = min(batch_size, len(inputs) - start, remaining)
                if count == 0:
                    return
                yield (
                    inputs[start : start + count].long(),
                    labels[start : start + count].long(),
                )
                remaining -= count
    for path in packed_paths:
        capacity = sequence_length + 1
        if path.stat().st_size % (2 * capacity):
            raise ValueError(
                f"{path} does not contain complete uint16 rows of width {capacity}"
            )
        rows = np.memmap(path, dtype="<u2", mode="r").reshape(-1, capacity)
        for start in range(0, len(rows), batch_size):
            count = min(batch_size, len(rows) - start, remaining)
            if count == 0:
                return
            packed = torch.from_numpy(
                np.array(rows[start : start + count], dtype=np.int64)
            )
            yield packed[:, :-1], packed[:, 1:]
            remaining -= count
    if remaining:
        raise ValueError(f"Validation input is short by {remaining} sequences")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--validation-cache",
        type=Path,
        nargs="+",
        help="All rank cache files for the original validation set, in fixed order.",
    )
    inputs.add_argument(
        "--packed-files",
        type=Path,
        nargs="+",
        help="Ordered .bin files; this is a new prefix evaluation, not the historical shuffled cache.",
    )
    parser.add_argument("--num-sequences", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ignore-index",
        type=int,
        help="Default: training recipe's loss_ignore_index, otherwise tokenizer PAD.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/arloop_ppl.json"))
    args = parser.parse_args()
    if min(args.batch_size, args.num_sequences) < 1:
        parser.error("batch-size and num-sequences must be positive")
    raw = yaml.safe_load(Path(args.config).read_text())
    loaded = load_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
    )
    ignore = (
        args.ignore_index
        if args.ignore_index is not None
        else raw.get("loss_ignore_index")
    )
    if ignore is None:
        ignore = (
            loaded.tokenizer.pad_id if loaded.tokenizer.pad_id is not None else -100
        )
    device = torch.device(args.device)
    loss_sum, tokens, sequences = 0.0, 0, 0
    with torch.inference_mode():
        for inputs, labels in validation_batches(
            args.validation_cache or [],
            args.packed_files or [],
            int(raw["block_size"]),
            args.batch_size,
            args.num_sequences,
        ):
            sequences += len(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            context = (
                torch.autocast(device.type, dtype=loaded.compute_dtype)
                if loaded.compute_dtype != torch.float32
                else nullcontext()
            )
            with context:
                logits = loaded.model(input_ids=inputs, return_logits=True)["logits"]
                loss = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=ignore,
                    reduction="sum",
                )
            loss_sum += loss.item()
            tokens += labels.ne(ignore).sum().item()
    if not tokens:
        raise ValueError("No scored tokens in validation input")
    payload = {
        "checkpoint": str(loaded.checkpoint_path),
        "config": args.config,
        "sequences": sequences,
        "scored_tokens": tokens,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "ignore_index": ignore,
        "validation_mode": "cache" if args.validation_cache else "packed_prefix",
        "input_files": [str(p) for p in (args.validation_cache or args.packed_files)],
        "loss": loss_sum / tokens,
        "perplexity": math.exp(loss_sum / tokens),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
