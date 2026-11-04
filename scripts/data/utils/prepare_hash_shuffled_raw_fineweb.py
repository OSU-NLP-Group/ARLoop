#!/usr/bin/env python3
"""Deterministically shuffle raw FineWeb-Edu Parquet with external hash partitions.

Each source row receives a unique document ID derived from its sorted source-file
index and row index. SplitMix64 maps that ID to a deterministic shuffle key. The
high key bits route rows into temporary buckets; sorting each bucket by the full
key and reading final bucket files in numeric order is a global document shuffle.

Temporary fragments contain ``shuffle_key`` and ``document_id`` plus every raw
source column. Final Parquet files contain exactly the raw source columns, with
shuffle provenance recorded as schema metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


from .common import write_json
from .shuffle_utils import (
    aggregate,
    bucket_ids,
    require_power_of_two,
    splitmix64,
)


def discover_files(source_root: Path) -> list[Path]:
    paths = sorted((source_root / "data").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files below {source_root / 'data'}")
    return paths


def partition_schema(raw_schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [pa.field("shuffle_key", pa.uint64()), pa.field("document_id", pa.uint64())]
        + list(raw_schema),
    )


def fragment_path(output_root: Path, bucket_id: int, file_index: int) -> Path:
    return (
        output_root
        / "_partitions"
        / f"bucket-{bucket_id:05d}"
        / f"source-{file_index:05d}.parquet"
    )


def partition_stats_path(output_root: Path, file_index: int) -> Path:
    return output_root / ".stats" / "partition" / f"source-{file_index:05d}.json"


def bucket_path(output_root: Path, bucket_id: int, num_buckets: int) -> Path:
    return output_root / "data" / f"part-{bucket_id:05d}-of-{num_buckets:05d}.parquet"


def bucket_stats_path(output_root: Path, bucket_id: int) -> Path:
    return output_root / ".stats" / "buckets" / f"bucket-{bucket_id:05d}.json"


class PartitionWriters:
    def __init__(
        self,
        *,
        output_root: Path,
        file_index: int,
        schema: pa.Schema,
        num_buckets: int,
        buffer_bytes: int,
        compression: str,
    ) -> None:
        self.output_root = output_root
        self.file_index = file_index
        self.schema = schema
        self.num_buckets = num_buckets
        self.buffer_bytes = buffer_bytes
        self.compression = compression
        self.buffers: dict[int, list[pa.Table]] = {}
        self.buffer_sizes: dict[int, int] = {}
        self.writers: dict[int, pq.ParquetWriter] = {}
        self.temporary_paths: dict[int, Path] = {}
        self.documents = np.zeros(num_buckets, dtype=np.int64)
        self.row_groups = np.zeros(num_buckets, dtype=np.int64)

    def _writer(self, bucket_id: int) -> pq.ParquetWriter:
        writer = self.writers.get(bucket_id)
        if writer is not None:
            return writer
        final = fragment_path(self.output_root, bucket_id, self.file_index)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(f".{final.name}.tmp.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        writer = pq.ParquetWriter(
            temporary,
            self.schema,
            compression=self.compression,
            use_dictionary=True,
            write_statistics=True,
        )
        self.writers[bucket_id] = writer
        self.temporary_paths[bucket_id] = temporary
        return writer

    def append(self, bucket_id: int, table: pa.Table) -> None:
        self.buffers.setdefault(bucket_id, []).append(table)
        self.buffer_sizes[bucket_id] = (
            self.buffer_sizes.get(bucket_id, 0) + table.nbytes
        )
        self.documents[bucket_id] += len(table)
        if self.buffer_sizes[bucket_id] >= self.buffer_bytes:
            self.flush(bucket_id)

    def flush(self, bucket_id: int) -> None:
        tables = self.buffers.pop(bucket_id, [])
        self.buffer_sizes.pop(bucket_id, None)
        if not tables:
            return
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        self._writer(bucket_id).write_table(table, row_group_size=len(table))
        self.row_groups[bucket_id] += 1

    def finish(self) -> list[dict]:
        for bucket_id in sorted(self.buffers):
            self.flush(bucket_id)
        for writer in self.writers.values():
            writer.close()
        records = []
        for bucket_id, temporary in sorted(self.temporary_paths.items()):
            final = fragment_path(self.output_root, bucket_id, self.file_index)
            os.replace(temporary, final)
            records.append(
                {
                    "bucket_id": bucket_id,
                    "documents": int(self.documents[bucket_id]),
                    "row_groups": int(self.row_groups[bucket_id]),
                    "size_bytes": final.stat().st_size,
                }
            )
        return records

    def abort(self) -> None:
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for temporary in self.temporary_paths.values():
            temporary.unlink(missing_ok=True)


def partition_file(args: argparse.Namespace) -> None:
    require_power_of_two(args.num_buckets)
    files = discover_files(args.source_root)
    if args.file_index < 0 or args.file_index >= len(files):
        raise IndexError(f"file index {args.file_index} outside [0, {len(files)})")
    source = files[args.file_index]
    stats_path = partition_stats_path(args.output_root, args.file_index)
    if stats_path.is_file() and not args.force:
        stats = json.loads(stats_path.read_text())
        expected = {
            "source_path": str(source.resolve()),
            "seed": args.seed,
            "num_buckets": args.num_buckets,
        }
        if all(stats.get(key) == value for key, value in expected.items()):
            print(f"SKIP valid partition: {source}", flush=True)
            return
        raise ValueError(f"Existing stats conflict with requested run: {stats_path}")

    parquet_file = pq.ParquetFile(source)
    raw_schema = parquet_file.schema_arrow
    if "text" not in raw_schema.names or "id" not in raw_schema.names:
        raise ValueError(f"Unexpected FineWeb-Edu schema: {raw_schema}")
    schema = partition_schema(raw_schema.remove_metadata())
    num_row_groups = parquet_file.num_row_groups
    if args.max_row_groups is not None:
        num_row_groups = min(num_row_groups, args.max_row_groups)
    writers = PartitionWriters(
        output_root=args.output_root,
        file_index=args.file_index,
        schema=schema,
        num_buckets=args.num_buckets,
        buffer_bytes=args.buffer_kib * 1024,
        compression=args.compression,
    )
    started = time.time()
    row_offset = 0
    documents = 0
    token_count = 0
    try:
        for row_group_index in range(num_row_groups):
            raw = parquet_file.read_row_group(row_group_index)
            rows = len(raw)
            if row_offset + rows >= (1 << 32):
                raise ValueError(f"Source has too many rows for document IDs: {source}")
            document_ids = np.arange(
                row_offset, row_offset + rows, dtype=np.uint64
            ) | np.uint64(args.file_index << 32)
            keys = splitmix64(document_ids, args.seed)
            ids = bucket_ids(keys, args.num_buckets)
            order = np.argsort(ids, kind="stable")
            sorted_ids = ids[order]
            boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [rows]))
            for start, end in zip(starts, ends):
                take_indices = order[start:end]
                bucket_id = int(sorted_ids[start])
                indices = pa.array(take_indices, type=pa.int64())
                table = pa.Table.from_arrays(
                    [
                        pa.array(keys[take_indices], type=pa.uint64()),
                        pa.array(document_ids[take_indices], type=pa.uint64()),
                    ]
                    + [pc.take(raw.column(name), indices) for name in raw_schema.names],
                    schema=schema,
                )
                writers.append(bucket_id, table)

            documents += rows
            if "token_count" in raw_schema.names:
                value = pc.sum(raw.column("token_count")).as_py()
                token_count += int(value or 0)
            row_offset += rows
            if (row_group_index + 1) % 100 == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"partition file={args.file_index + 1}/{len(files)} "
                    f"rg={row_group_index + 1}/{num_row_groups} "
                    f"docs={documents:,} input_MB/s="
                    f"{source.stat().st_size / 1e6 * (row_group_index + 1) / parquet_file.num_row_groups / elapsed:,.1f}",
                    flush=True,
                )
        bucket_records = writers.finish()
    except Exception:
        writers.abort()
        raise

    elapsed = time.time() - started
    stats = {
        "complete": True,
        "file_index": args.file_index,
        "source_path": str(source.resolve()),
        "source_size_bytes": source.stat().st_size,
        "source_row_groups": parquet_file.num_row_groups,
        "processed_row_groups": num_row_groups,
        "documents": documents,
        "token_count": token_count,
        "seed": args.seed,
        "num_buckets": args.num_buckets,
        "compression": args.compression,
        "partition_size_bytes": sum(record["size_bytes"] for record in bucket_records),
        "elapsed_seconds": elapsed,
        "buckets": bucket_records,
    }
    write_json(stats_path, stats)
    print(
        f"DONE partition file={args.file_index} docs={documents:,} "
        f"source_GB={source.stat().st_size / 1e9:.3f} "
        f"partition_GB={stats['partition_size_bytes'] / 1e9:.3f} "
        f"elapsed={elapsed:.1f}s source_MB/s={source.stat().st_size / 1e6 / elapsed:.1f}",
        flush=True,
    )


def sort_bucket(args: argparse.Namespace) -> None:
    require_power_of_two(args.num_buckets)
    if args.bucket_id < 0 or args.bucket_id >= args.num_buckets:
        raise IndexError(f"bucket {args.bucket_id} outside [0, {args.num_buckets})")
    output = bucket_path(args.output_root, args.bucket_id, args.num_buckets)
    stats_path = bucket_stats_path(args.output_root, args.bucket_id)
    if output.is_file() and stats_path.is_file() and not args.force:
        print(f"SKIP valid bucket: {output}", flush=True)
        return
    fragments = sorted(
        (args.output_root / "_partitions" / f"bucket-{args.bucket_id:05d}").glob(
            "source-*.parquet"
        )
    )
    if args.num_source_files is not None:
        fragments = [
            path
            for path in fragments
            if int(path.stem.rsplit("-", 1)[1]) < args.num_source_files
        ]
    if not fragments:
        raise FileNotFoundError(f"No fragments for bucket {args.bucket_id}")

    started = time.time()
    tables = [pq.read_table(path) for path in fragments]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    keys = table.column("shuffle_key").combine_chunks().to_numpy(zero_copy_only=False)
    ids = bucket_ids(keys, args.num_buckets)
    if len(ids) and not np.all(ids == args.bucket_id):
        raise ValueError(f"Misrouted rows in bucket {args.bucket_id}")
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if len(sorted_keys) > 1 and not np.all(sorted_keys[1:] > sorted_keys[:-1]):
        raise ValueError(f"Duplicate or unsorted keys in bucket {args.bucket_id}")

    raw_names = table.schema.names[2:]
    indices = pa.array(order, type=pa.int64())
    raw_columns = [pc.take(table.column(name), indices) for name in raw_names]
    source_schema = pq.ParquetFile(fragments[0]).schema_arrow
    raw_schema = pa.schema([source_schema.field(name) for name in raw_names])
    metadata = {
        b"shuffle_method": b"splitmix64-bijective-hash-partition",
        b"shuffle_seed": str(args.seed).encode(),
        b"shuffle_num_buckets": str(args.num_buckets).encode(),
        b"shuffle_bucket_id": str(args.bucket_id).encode(),
        b"global_order": b"ascending-shuffle-key-across-ascending-bucket-id",
    }
    output_table = pa.Table.from_arrays(
        raw_columns, schema=raw_schema.with_metadata(metadata)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    pq.write_table(
        output_table,
        temporary,
        compression=args.compression,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=args.output_row_group_rows,
    )
    os.replace(temporary, output)
    elapsed = time.time() - started
    stats = {
        "complete": True,
        "bucket_id": args.bucket_id,
        "seed": args.seed,
        "num_buckets": args.num_buckets,
        "fragments": len(fragments),
        "partition_size_bytes": sum(path.stat().st_size for path in fragments),
        "output_path": str(output.resolve()),
        "output_size_bytes": output.stat().st_size,
        "documents": len(table),
        "token_count": int(pc.sum(table.column("token_count")).as_py() or 0),
        "first_shuffle_key": int(sorted_keys[0]) if len(sorted_keys) else None,
        "last_shuffle_key": int(sorted_keys[-1]) if len(sorted_keys) else None,
        "elapsed_seconds": elapsed,
    }
    write_json(stats_path, stats)
    if args.delete_partitions:
        for path in fragments:
            path.unlink()
    print(
        f"DONE sort bucket={args.bucket_id}/{args.num_buckets} "
        f"docs={len(table):,} output_MB={output.stat().st_size / 1e6:.1f} "
        f"elapsed={elapsed:.1f}s output_MB/s={output.stat().st_size / 1e6 / elapsed:.1f}",
        flush=True,
    )


def summarize(args: argparse.Namespace) -> None:
    partition_records = [
        json.loads(partition_stats_path(args.output_root, index).read_text())
        for index in range(args.num_source_files)
    ]
    bucket_records = [
        json.loads(bucket_stats_path(args.output_root, bucket).read_text())
        for bucket in range(args.num_buckets)
    ]
    partition_totals = aggregate(partition_records)
    bucket_totals = aggregate(bucket_records)
    if partition_totals != bucket_totals:
        raise ValueError(
            f"Partition/output count mismatch: {partition_totals} != {bucket_totals}"
        )
    previous_last = None
    for record in bucket_records:
        first = record["first_shuffle_key"]
        if first is not None and previous_last is not None and first <= previous_last:
            raise ValueError("Final bucket boundaries are not globally increasing")
        if record["last_shuffle_key"] is not None:
            previous_last = record["last_shuffle_key"]
    payload = {
        "complete": True,
        "benchmark": True,
        "method": "splitmix64-bijective-hash-partition",
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "seed": args.seed,
        "num_source_files": args.num_source_files,
        "num_buckets": args.num_buckets,
        **bucket_totals,
        "source_size_bytes": sum(r["source_size_bytes"] for r in partition_records),
        "partition_size_bytes": sum(
            r["partition_size_bytes"] for r in partition_records
        ),
        "output_size_bytes": sum(r["output_size_bytes"] for r in bucket_records),
        "partition_cpu_elapsed_seconds": sum(
            r["elapsed_seconds"] for r in partition_records
        ),
        "sort_cpu_elapsed_seconds": sum(r["elapsed_seconds"] for r in bucket_records),
    }
    (args.output_root / "_SUCCESS").write_text("raw hash-shuffle benchmark complete\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-buckets", type=int, default=256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    partition = subparsers.add_parser("partition-file")
    partition.add_argument("--file-index", type=int, required=True)
    partition.add_argument("--max-row-groups", type=int)
    partition.add_argument("--buffer-kib", type=int, default=256)
    partition.add_argument(
        "--compression", choices=("snappy", "zstd"), default="snappy"
    )
    partition.add_argument("--force", action="store_true")
    partition.set_defaults(func=partition_file)

    sort = subparsers.add_parser("sort-bucket")
    sort.add_argument("--bucket-id", type=int, required=True)
    sort.add_argument("--num-source-files", type=int)
    sort.add_argument("--output-row-group-rows", type=int, default=1000)
    sort.add_argument("--compression", choices=("snappy", "zstd"), default="snappy")
    sort.add_argument("--delete-partitions", action="store_true")
    sort.add_argument("--force", action="store_true")
    sort.set_defaults(func=sort_bucket)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--num-source-files", type=int, required=True)
    summary.set_defaults(func=summarize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
