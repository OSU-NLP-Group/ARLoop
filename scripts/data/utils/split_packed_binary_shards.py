#!/usr/bin/env python3
"""Split one fixed-row packed binary into balanced row-aligned shards."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


from .common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = args.source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    capacity = int(source_manifest["capacity"])
    row_bytes = capacity * 2
    source_files = sorted((args.source_root / args.split).glob("*.bin"))
    if len(source_files) != 1:
        raise ValueError(f"Expected one source binary, found {len(source_files)}")
    source = source_files[0]
    if source.stat().st_size % row_bytes:
        raise ValueError(f"Source is not row aligned: {source}")
    rows = source.stat().st_size // row_bytes
    quotient, remainder = divmod(rows, args.num_shards)
    rows_per_shard = [
        quotient + int(index < remainder) for index in range(args.num_shards)
    ]
    if min(rows_per_shard) < 1:
        raise ValueError("More shards than source rows")

    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_root / "_SUCCESS").unlink(missing_ok=True)
    shards = []
    with source.open("rb", buffering=32 * 1024 * 1024) as source_handle:
        for index, shard_rows in enumerate(rows_per_shard):
            output = output_dir / f"lane-{index:03d}-of-{args.num_shards:03d}.bin"
            expected_bytes = shard_rows * row_bytes
            temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
            remaining = expected_bytes
            with temporary.open("wb", buffering=32 * 1024 * 1024) as output_handle:
                while remaining:
                    chunk = source_handle.read(min(16 * 1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError(f"Unexpected EOF while writing {output}")
                    output_handle.write(chunk)
                    remaining -= len(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
            os.replace(temporary, output)
            shards.append(
                {
                    "shard_id": index,
                    "num_shards": args.num_shards,
                    "rows": shard_rows,
                    "size_bytes": expected_bytes,
                    "path": str(output.resolve()),
                }
            )
        if source_handle.read(1):
            raise ValueError("Source has trailing bytes after shard split")

    manifest = dict(source_manifest)
    manifest.update(
        {
            "complete": True,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "algorithm": f"{source_manifest['algorithm']}_row_split_r{args.num_shards}",
            "source_root": str(args.source_root.resolve()),
            "num_lanes": args.num_shards,
            "output_rows": rows,
            "output_size_bytes": source.stat().st_size,
            "shards": shards,
            "split": args.split,
        }
    )
    manifest.pop("output", None)
    manifest.pop("lane_stats", None)
    write_json(args.output_root / "manifest.json", manifest)
    write_json(
        args.output_root / "_SUCCESS",
        {
            "manifest": "manifest.json",
            "output_rows": rows,
            "num_lanes": args.num_shards,
        },
    )
    if args.source_root.resolve() == args.output_root.resolve():
        source.unlink()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
