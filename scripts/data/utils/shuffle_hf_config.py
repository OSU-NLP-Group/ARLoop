#!/usr/bin/env python3
"""Globally shuffle one Hugging Face dataset config and write physical Parquet."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


from . import hf_shards


from .common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-num-proc", type=int, default=16)
    parser.add_argument("--write-workers", type=int, default=24)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--batch-rows", type=int, default=1000)
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
        f"repo={args.repo_id}@{repo_info.sha} config={args.config_name} "
        f"split={args.split} seed={args.seed} output={output_root}",
        flush=True,
    )

    load_started = time.time()
    source = load_dataset(
        args.repo_id,
        args.config_name,
        split=args.split,
        revision=repo_info.sha,
        cache_dir=str(cache_root),
        num_proc=args.download_num_proc,
    )
    if not isinstance(source, Dataset):
        raise TypeError(f"Expected Dataset, got {type(source)!r}")
    total_rows = len(source)
    source_first_row = hf_shards.row_value(source[0])
    print(
        f"load_complete rows={total_rows:,} columns={source.column_names} "
        f"elapsed_seconds={time.time() - load_started:.1f}",
        flush=True,
    )

    indices_path = staging_root / f"shuffle_indices_seed{args.seed}.arrow"
    shuffle_started = time.time()
    shuffled = source.shuffle(
        seed=args.seed,
        keep_in_memory=False,
        indices_cache_file_name=str(indices_path),
    )
    shuffled_first_row = hf_shards.row_value(shuffled[0])
    if total_rows > 1 and source_first_row == shuffled_first_row:
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
    with ProcessPoolExecutor(
        max_workers=args.write_workers,
        mp_context=mp.get_context("fork"),
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
        "config_name": args.config_name,
        "split": args.split,
        "requested_revision": args.revision,
        "resolved_revision": repo_info.sha,
        "seed": args.seed,
        "shuffle": "datasets.Dataset.shuffle",
        "physical_materialization": "Contiguous shards of the shuffled Dataset written as Parquet",
        "rows": total_rows,
        "columns": source.column_names,
        "features": str(source.features),
        "num_shards": args.num_shards,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "shuffle_indices_path": str(indices_path),
        "shuffle_indices_size_bytes": indices_path.stat().st_size,
        "output_size_bytes": sum(item["size_bytes"] for item in results),
        "load_seconds": shuffle_started - load_started,
        "shuffle_index_seconds": write_started - shuffle_started,
        "parquet_write_seconds": time.time() - write_started,
        "elapsed_seconds": time.time() - started,
        "files": results,
    }
    write_json(output_root / "manifest.json", manifest)
    success_path.write_text(
        f"HF Dataset shuffle complete repo={args.repo_id} "
        f"revision={repo_info.sha} config={args.config_name} "
        f"split={args.split} seed={args.seed} rows={total_rows}\n"
    )
    print(
        f"SUCCESS rows={total_rows:,} files={len(results)} "
        f"bytes={manifest['output_size_bytes']:,} "
        f"elapsed_seconds={manifest['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
