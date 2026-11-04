#!/usr/bin/env python3
"""Download OpenMathReasoning, globally shuffle all splits, and write Parquet."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi


EXP_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = EXP_DIR / "data" / "openmathreasoning_shuffled_s42"
DEFAULT_REPO_ID = "nvidia/OpenMathReasoning"
DEFAULT_REVISION = "d3d08664755704f422af97d43a7ff0ded4bd95df"
DEFAULT_SPLITS = ("cot", "tir", "genselect", "additional_problems")

from . import hf_shards


from .common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-num-proc", type=int, default=16)
    parser.add_argument("--write-workers", type=int, default=24)
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    output_root = args.output_root.resolve()
    parquet_root = output_root / "parquet"
    staging_root = output_root / ".staging"
    cache_root = output_root / ".hf_home" / "datasets"
    success_path = output_root / "_SUCCESS"

    output_root.mkdir(parents=True, exist_ok=True)
    parquet_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    if success_path.is_file():
        print(f"SKIP completed output: {output_root}", flush=True)
        return

    repo_info = HfApi().dataset_info(args.repo_id, revision=args.revision)
    if repo_info.sha != args.revision:
        raise ValueError(
            f"Pinned revision did not resolve to itself: {args.revision} -> {repo_info.sha}"
        )
    print(
        f"repo={args.repo_id}@{repo_info.sha} seed={args.seed} output={output_root}",
        flush=True,
    )

    download_started = time.time()
    source = load_dataset(
        args.repo_id,
        revision=repo_info.sha,
        cache_dir=str(cache_root),
        num_proc=args.download_num_proc,
    )
    if not isinstance(source, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(source)!r}")
    missing_splits = [name for name in DEFAULT_SPLITS if name not in source]
    unexpected_splits = [name for name in source if name not in DEFAULT_SPLITS]
    if missing_splits or unexpected_splits:
        raise ValueError(
            f"Unexpected splits: missing={missing_splits}, unexpected={unexpected_splits}"
        )
    split_rows = {name: len(source[name]) for name in DEFAULT_SPLITS}
    print(
        "download_complete "
        + " ".join(f"{name}={rows:,}" for name, rows in split_rows.items())
        + f" elapsed_seconds={time.time() - download_started:.1f}",
        flush=True,
    )

    feature_signatures = {name: str(source[name].features) for name in DEFAULT_SPLITS}
    if len(set(feature_signatures.values())) != 1:
        raise ValueError(f"Split schemas differ: {feature_signatures}")

    combined = concatenate_datasets([source[name] for name in DEFAULT_SPLITS])
    total_rows = sum(split_rows.values())
    if len(combined) != total_rows:
        raise ValueError(f"Combined rows {len(combined)} != expected {total_rows}")
    source_first_row = hf_shards.row_value(combined[0])

    indices_path = staging_root / f"shuffle_indices_seed{args.seed}.arrow"
    shuffle_started = time.time()
    shuffled = combined.shuffle(
        seed=args.seed,
        keep_in_memory=False,
        indices_cache_file_name=str(indices_path),
    )
    shuffled_first_row = hf_shards.row_value(shuffled[0])
    if source_first_row == shuffled_first_row:
        raise RuntimeError("Shuffle unexpectedly retained the original first row")
    print(
        f"shuffle_index_complete rows={len(shuffled):,} "
        f"indices_bytes={indices_path.stat().st_size:,} "
        f"elapsed_seconds={time.time() - shuffle_started:.1f}",
        flush=True,
    )

    hf_shards._SHUFFLED_DATASET = shuffled
    tasks = []
    for shard_id in range(args.num_shards):
        output_path = parquet_root / (
            f"train-{shard_id:05d}-of-{args.num_shards:05d}.parquet"
        )
        tasks.append(
            (
                shard_id,
                args.num_shards,
                str(output_path),
                args.batch_rows,
                args.compression,
                args.compression_level,
            )
        )

    write_started = time.time()
    results: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.write_workers,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(hf_shards.write_one_shard, task): task[0] for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"shard_complete id={result['shard_id']:03d} "
                f"rows={result['rows']:,} bytes={result['size_bytes']:,} "
                f"reused={result['reused']} completed={len(results)}/{len(tasks)}",
                flush=True,
            )

    results.sort(key=lambda item: item["shard_id"])
    written_rows = sum(item["rows"] for item in results)
    if written_rows != total_rows:
        raise ValueError(f"Output rows {written_rows} != source rows {total_rows}")
    for result in results:
        expected = hf_shards.expected_shard_rows(
            total_rows,
            args.num_shards,
            result["shard_id"],
        )
        if result["rows"] != expected:
            raise ValueError(
                f"Shard {result['shard_id']} rows={result['rows']} expected={expected}"
            )

    first_output = pq.read_table(results[0]["path"]).slice(0, 1).to_pylist()[0]
    output_first_row = hf_shards.row_value(first_output)
    if output_first_row != shuffled_first_row:
        raise ValueError("First physical output row does not match shuffled row zero")

    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": repo_info.sha,
        "seed": args.seed,
        "shuffle": "datasets.Dataset.shuffle over concatenated cot, tir, genselect, and additional_problems splits",
        "physical_materialization": "Contiguous shards of the shuffled Dataset written as Parquet; no indices mapping is required to read output",
        "split_order_before_shuffle": list(DEFAULT_SPLITS),
        "split_rows": split_rows,
        "rows": total_rows,
        "columns": combined.column_names,
        "features": str(combined.features),
        "num_shards": args.num_shards,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "shuffle_indices_path": str(indices_path),
        "shuffle_indices_size_bytes": indices_path.stat().st_size,
        "output_size_bytes": sum(item["size_bytes"] for item in results),
        "download_and_load_seconds": shuffle_started - download_started,
        "shuffle_index_seconds": write_started - shuffle_started,
        "parquet_write_seconds": time.time() - write_started,
        "elapsed_seconds": time.time() - started,
        "files": results,
    }
    write_json(output_root / "manifest.json", manifest)
    success_path.write_text(
        f"OpenMathReasoning global shuffle complete revision={repo_info.sha} "
        f"seed={args.seed} rows={total_rows}\n"
    )
    print(
        f"SUCCESS rows={total_rows:,} files={len(results)} "
        f"bytes={manifest['output_size_bytes']:,} "
        f"elapsed_seconds={manifest['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
