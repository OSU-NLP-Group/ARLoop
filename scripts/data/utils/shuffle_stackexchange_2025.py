#!/usr/bin/env python3
"""Download all StackExchange 2025 sites, globally shuffle, and write Parquet."""

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
from huggingface_hub import HfApi, snapshot_download

from . import hf_shards as common
from .common import write_json


EXP_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = EXP_DIR / "data" / "stackexchange_2025_md" / "raw_shuffled_s42"
DEFAULT_REPO_ID = "HuggingFaceTB/stackexchange_2025_md"
DEFAULT_REVISION = "fe382c48856f4eae5d8402cd4aa9be55ee1e1ee0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--load-num-proc", type=int, default=32)
    parser.add_argument("--write-workers", type=int, default=24)
    parser.add_argument("--num-shards", type=int, default=256)
    parser.add_argument("--batch-rows", type=int, default=1000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    output_root = args.output_root.resolve()
    source_root = output_root / ".source_snapshot"
    cache_root = output_root / ".hf_home" / "datasets"
    staging_root = output_root / ".staging"
    parquet_root = output_root / "parquet"
    success_path = output_root / "_SUCCESS"
    for path in (source_root, cache_root, staging_root, parquet_root):
        path.mkdir(parents=True, exist_ok=True)
    if success_path.is_file():
        print(f"SKIP completed output: {output_root}", flush=True)
        return

    repo_info = HfApi().dataset_info(
        args.repo_id,
        revision=args.revision,
        files_metadata=True,
    )
    if repo_info.sha != args.revision:
        raise ValueError(
            f"Pinned revision did not resolve to itself: {args.revision} -> {repo_info.sha}"
        )
    expected_files = sorted(
        sibling.rfilename
        for sibling in (repo_info.siblings or [])
        if "/" in sibling.rfilename and sibling.rfilename.endswith(".parquet")
    )
    if not expected_files:
        raise RuntimeError("No site Parquet files found in the remote repository")
    print(
        f"repo={args.repo_id}@{repo_info.sha} files={len(expected_files)} "
        f"seed={args.seed} output={output_root}",
        flush=True,
    )

    download_started = time.time()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=repo_info.sha,
        allow_patterns=["*/*.parquet", "README.md"],
        local_dir=source_root,
        max_workers=args.download_workers,
    )
    source_files = sorted(source_root.glob("*/*.parquet"))
    relative_files = [str(path.relative_to(source_root)) for path in source_files]
    if relative_files != expected_files:
        missing = sorted(set(expected_files) - set(relative_files))
        unexpected = sorted(set(relative_files) - set(expected_files))
        raise ValueError(
            f"Snapshot mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    domains = sorted({path.parent.name for path in source_files})
    print(
        f"download_complete files={len(source_files)} domains={len(domains)} "
        f"elapsed_seconds={time.time() - download_started:.1f}",
        flush=True,
    )

    load_started = time.time()
    source = load_dataset(
        "parquet",
        data_files=[str(path) for path in source_files],
        split="train",
        cache_dir=str(cache_root),
        num_proc=args.load_num_proc,
    )
    if not isinstance(source, Dataset):
        raise TypeError(f"Expected Dataset, got {type(source)!r}")
    total_rows = len(source)
    source_first_row = common.row_value(source[0])
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
    shuffled_first_row = common.row_value(shuffled[0])
    if total_rows > 1 and source_first_row == shuffled_first_row:
        raise RuntimeError("Shuffle unexpectedly retained the original first row")
    print(
        f"shuffle_index_complete rows={len(shuffled):,} "
        f"indices_bytes={indices_path.stat().st_size:,} "
        f"elapsed_seconds={time.time() - shuffle_started:.1f}",
        flush=True,
    )

    common._SHUFFLED_DATASET = shuffled
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
            executor.submit(common.write_one_shard, task): task[0] for task in tasks
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
        expected = common.expected_shard_rows(
            total_rows,
            args.num_shards,
            result["shard_id"],
        )
        if result["rows"] != expected:
            raise ValueError(
                f"Shard {result['shard_id']} rows={result['rows']} expected={expected}"
            )

    first_output = pq.read_table(results[0]["path"]).slice(0, 1).to_pylist()[0]
    output_first_row = common.row_value(first_output)
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
        "shuffle": "datasets.Dataset.shuffle over all StackExchange site Parquet files",
        "physical_materialization": "Contiguous shards of the shuffled Dataset written as Parquet",
        "source_file_count": len(source_files),
        "domain_count": len(domains),
        "domains": domains,
        "rows": total_rows,
        "columns": source.column_names,
        "features": str(source.features),
        "num_shards": args.num_shards,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "shuffle_indices_path": str(indices_path),
        "shuffle_indices_size_bytes": indices_path.stat().st_size,
        "output_size_bytes": sum(item["size_bytes"] for item in results),
        "download_seconds": load_started - download_started,
        "load_seconds": shuffle_started - load_started,
        "shuffle_index_seconds": write_started - shuffle_started,
        "parquet_write_seconds": time.time() - write_started,
        "elapsed_seconds": time.time() - started,
        "files": results,
    }
    write_json(output_root / "manifest.json", manifest)
    success_path.write_text(
        f"StackExchange 2025 global shuffle complete revision={repo_info.sha} "
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
