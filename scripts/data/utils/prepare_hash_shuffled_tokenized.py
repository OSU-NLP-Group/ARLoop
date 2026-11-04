#!/usr/bin/env python3
"""Globally shuffle tokenized documents with an external hash partition.

The source dataset is read sequentially.  Each document gets a stable, unique
64-bit shuffle key derived from ``(split, source_file_index, row_index)``.  The
high key bits select a temporary partition and each partition is then sorted by
the complete key.  Reading the final partition files in numeric order is thus
equivalent to globally sorting every document by its shuffle key.

Temporary files retain the key and document identity for validation.  Final
Parquet files contain only ``input_ids`` so they remain compatible with the
existing tokenized-data loader.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


EXP_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = EXP_DIR / "data" / "100bt_smol2_tokenized"
DEFAULT_OUTPUT_ROOT = EXP_DIR / "data" / "100bt_smol2_tokenized_hashshuffle_s42"
TOKEN_TYPE = pa.list_(pa.int32())
FINAL_SCHEMA = pa.schema([pa.field("input_ids", TOKEN_TYPE)])
PARTITION_SCHEMA = pa.schema(
    [
        pa.field("shuffle_key", pa.uint64()),
        pa.field("document_id", pa.uint64()),
        pa.field("input_ids", TOKEN_TYPE),
    ]
)
SPLIT_SALTS = {
    "train": 0x243F6A8885A308D3,
    "val": 0x13198A2E03707344,
}


@dataclass(frozen=True)
class SourceFile:
    split: str
    index: int
    path: Path


from .common import write_json
from .shuffle_utils import (
    bucket_ids,
    require_power_of_two,
    splitmix64 as mix_document_ids,
)


def discover_files(source_root: Path, split: str) -> list[SourceFile]:
    paths = sorted((source_root / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files under {source_root / split}")
    if len(paths) >= (1 << 32):
        raise ValueError("The document ID encoding supports fewer than 2^32 files")
    return [SourceFile(split, index, path) for index, path in enumerate(paths)]


def source_file(source_root: Path, split: str, index: int) -> SourceFile:
    files = discover_files(source_root, split)
    if index < 0 or index >= len(files):
        raise IndexError(f"{split} file index {index} outside [0, {len(files)})")
    return files[index]


def splitmix64(values: np.ndarray, seed: int, split: str) -> np.ndarray:
    return mix_document_ids(values, seed, SPLIT_SALTS[split])


def partition_dir(output_root: Path, split: str, bucket_id: int) -> Path:
    return output_root / "_partitions" / split / f"bucket-{bucket_id:05d}"


def partition_path(
    output_root: Path, split: str, bucket_id: int, file_index: int
) -> Path:
    return (
        partition_dir(output_root, split, bucket_id)
        / f"source-{file_index:05d}.parquet"
    )


def partition_stats_path(output_root: Path, source: SourceFile) -> Path:
    return (
        output_root
        / ".stats"
        / "partition"
        / source.split
        / f"source-{source.index:05d}.json"
    )


def bucket_output_path(
    output_root: Path, split: str, bucket_id: int, num_buckets: int
) -> Path:
    return output_root / split / f"part-{bucket_id:05d}-of-{num_buckets:05d}.parquet"


def bucket_stats_path(output_root: Path, split: str, bucket_id: int) -> Path:
    return output_root / ".stats" / "buckets" / split / f"bucket-{bucket_id:05d}.json"


def token_count(column: pa.ChunkedArray | pa.Array) -> int:
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    return sum(len(chunk.values) for chunk in chunks)


def token_sum(column: pa.ChunkedArray | pa.Array) -> int:
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    total = 0
    for chunk in chunks:
        if len(chunk.values):
            total += int(pc.sum(chunk.values).as_py())
    return total


def validate_source_schema(path: Path, parquet_file: pq.ParquetFile) -> None:
    schema = parquet_file.schema_arrow
    if schema.names != ["input_ids"] or schema.field("input_ids").type != TOKEN_TYPE:
        raise ValueError(f"{path}: expected input_ids: {TOKEN_TYPE}, got {schema}")


def validate_partition_stats(
    stats: dict,
    *,
    source: SourceFile,
    seed: int,
    num_buckets: int,
    require_complete_source: bool,
) -> None:
    expected = {
        "split": source.split,
        "file_index": source.index,
        "source_path": str(source.path.resolve()),
        "seed": seed,
        "num_buckets": num_buckets,
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(
                f"Partition stats mismatch for {source.path}: "
                f"{key}={stats.get(key)!r}, expected {value!r}"
            )
    if require_complete_source and not stats.get("complete_source"):
        raise ValueError(f"Partial partition stats cannot be reused: {source.path}")
    for entry in stats.get("buckets", []):
        output_root = Path(stats["output_root"])
        bucket_id = int(entry["bucket_id"])
        path = partition_path(output_root, source.split, bucket_id, source.index)
        if path.is_file():
            continue
        completed_stats = bucket_stats_path(output_root, source.split, bucket_id)
        completed_output = bucket_output_path(
            output_root, source.split, bucket_id, num_buckets
        )
        if not completed_stats.is_file() or not completed_output.is_file():
            raise FileNotFoundError(path)


class PartitionWriters:
    def __init__(
        self,
        *,
        output_root: Path,
        source: SourceFile,
        num_buckets: int,
        compression: str,
        compression_level: int,
        buffer_bytes: int,
    ):
        self.output_root = output_root
        self.source = source
        self.num_buckets = num_buckets
        self.compression = compression
        self.compression_level = compression_level
        self.buffer_bytes = buffer_bytes
        self.buffers: dict[int, list[pa.Table]] = {}
        self.buffer_sizes: dict[int, int] = {}
        self.writers: dict[int, pq.ParquetWriter] = {}
        self.temporary_paths: dict[int, Path] = {}
        self.documents = np.zeros(num_buckets, dtype=np.int64)
        self.tokens = np.zeros(num_buckets, dtype=np.int64)
        self.row_groups = np.zeros(num_buckets, dtype=np.int64)

    def _writer(self, bucket_id: int) -> pq.ParquetWriter:
        writer = self.writers.get(bucket_id)
        if writer is not None:
            return writer
        final_path = partition_path(
            self.output_root, self.source.split, bucket_id, self.source.index
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        writer = pq.ParquetWriter(
            temporary,
            PARTITION_SCHEMA,
            compression=self.compression,
            compression_level=self.compression_level,
            use_dictionary=False,
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
        self.tokens[bucket_id] += token_count(table.column("input_ids"))
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
            final_path = partition_path(
                self.output_root, self.source.split, bucket_id, self.source.index
            )
            os.replace(temporary, final_path)
            records.append(
                {
                    "bucket_id": bucket_id,
                    "documents": int(self.documents[bucket_id]),
                    "tokens": int(self.tokens[bucket_id]),
                    "row_groups": int(self.row_groups[bucket_id]),
                    "size_bytes": final_path.stat().st_size,
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
    source = source_file(args.source_root, args.split, args.file_index)
    stats_path = partition_stats_path(args.output_root, source)
    if stats_path.is_file() and not args.force and args.max_row_groups is None:
        stats = json.loads(stats_path.read_text())
        validate_partition_stats(
            stats,
            source=source,
            seed=args.seed,
            num_buckets=args.num_buckets,
            require_complete_source=True,
        )
        print(f"SKIP valid partition: {source.path}", flush=True)
        return

    old_stats = json.loads(stats_path.read_text()) if stats_path.is_file() else None
    if old_stats:
        for entry in old_stats.get("buckets", []):
            partition_path(
                args.output_root,
                source.split,
                int(entry["bucket_id"]),
                source.index,
            ).unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)

    parquet_file = pq.ParquetFile(source.path)
    validate_source_schema(source.path, parquet_file)
    source_count = len(discover_files(args.source_root, source.split))
    num_row_groups = parquet_file.num_row_groups
    if args.max_row_groups is not None:
        num_row_groups = min(num_row_groups, args.max_row_groups)

    writers = PartitionWriters(
        output_root=args.output_root,
        source=source,
        num_buckets=args.num_buckets,
        compression=args.compression,
        compression_level=args.compression_level,
        buffer_bytes=args.buffer_mib * 1024 * 1024,
    )
    started = time.time()
    row_offset = 0
    documents = 0
    tokens = 0
    token_id_sum = 0
    try:
        for row_group_index in range(num_row_groups):
            input_ids = (
                parquet_file.read_row_group(row_group_index, columns=["input_ids"])
                .column("input_ids")
                .combine_chunks()
            )
            if input_ids.null_count or input_ids.values.null_count:
                raise ValueError(
                    f"{source.path} row group {row_group_index} contains null tokens"
                )
            rows = len(input_ids)
            if row_offset + rows >= (1 << 32):
                raise ValueError(
                    f"{source.path} contains too many rows for document IDs"
                )
            document_ids = np.arange(
                row_offset, row_offset + rows, dtype=np.uint64
            ) | np.uint64(source.index << 32)
            keys = splitmix64(document_ids, args.seed, source.split)
            ids = bucket_ids(keys, args.num_buckets)
            order = np.argsort(ids, kind="stable")
            sorted_ids = ids[order]
            boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [rows]))
            for start, end in zip(starts, ends):
                take_indices = order[start:end]
                bucket_id = int(sorted_ids[start])
                table = pa.Table.from_arrays(
                    [
                        pa.array(keys[take_indices], type=pa.uint64()),
                        pa.array(document_ids[take_indices], type=pa.uint64()),
                        pc.take(input_ids, pa.array(take_indices, type=pa.int64())),
                    ],
                    schema=PARTITION_SCHEMA,
                )
                writers.append(bucket_id, table)

            row_tokens = token_count(input_ids)
            documents += rows
            tokens += row_tokens
            token_id_sum += token_sum(input_ids)
            row_offset += rows
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"partition split={source.split} file={source.index + 1}/"
                f"{source_count} "
                f"rg={row_group_index + 1}/{num_row_groups} docs={documents:,} "
                f"tokens={tokens:,} tok/s={tokens / elapsed:,.0f}",
                flush=True,
            )
        bucket_records = writers.finish()
    except Exception:
        writers.abort()
        raise

    stats = {
        "complete": True,
        "complete_source": num_row_groups == parquet_file.num_row_groups,
        "method": "splitmix64-bijective-hash-partition",
        "split": source.split,
        "file_index": source.index,
        "source_path": str(source.path.resolve()),
        "source_size_bytes": source.path.stat().st_size,
        "source_row_groups": parquet_file.num_row_groups,
        "processed_row_groups": num_row_groups,
        "output_root": str(args.output_root.resolve()),
        "seed": args.seed,
        "num_buckets": args.num_buckets,
        "documents": documents,
        "tokens": tokens,
        "token_id_sum": token_id_sum,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "buffer_mib_per_bucket": args.buffer_mib,
        "elapsed_seconds": time.time() - started,
        "buckets": bucket_records,
    }
    write_json(stats_path, stats)
    print(
        f"DONE partition {source.path.name}: docs={documents:,} tokens={tokens:,} "
        f"elapsed={stats['elapsed_seconds']:.1f}s",
        flush=True,
    )


def read_partition_fragments(paths: list[Path]) -> pa.Table:
    tables: list[pa.Table] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        if parquet_file.schema_arrow != PARTITION_SCHEMA:
            raise ValueError(f"{path}: unexpected partition schema")
        tables.append(parquet_file.read())
    if not tables:
        return pa.Table.from_arrays(
            [
                pa.array([], type=pa.uint64()),
                pa.array([], type=pa.uint64()),
                pa.array([], type=TOKEN_TYPE),
            ],
            schema=PARTITION_SCHEMA,
        )
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def validate_bucket_stats(stats: dict, args: argparse.Namespace) -> None:
    expected = {
        "split": args.split,
        "bucket_id": args.bucket_id,
        "seed": args.seed,
        "num_buckets": args.num_buckets,
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(
                f"Bucket stats mismatch: {key}={stats.get(key)!r}, expected {value!r}"
            )
    output_path = bucket_output_path(
        args.output_root, args.split, args.bucket_id, args.num_buckets
    )
    if (
        not output_path.is_file()
        or output_path.stat().st_size != stats["output_size_bytes"]
    ):
        raise FileNotFoundError(f"Invalid completed bucket output: {output_path}")


def sort_bucket(args: argparse.Namespace) -> None:
    require_power_of_two(args.num_buckets)
    if args.bucket_id < 0 or args.bucket_id >= args.num_buckets:
        raise IndexError(f"Bucket {args.bucket_id} outside [0, {args.num_buckets})")
    output_path = bucket_output_path(
        args.output_root, args.split, args.bucket_id, args.num_buckets
    )
    stats_path = bucket_stats_path(args.output_root, args.split, args.bucket_id)
    if output_path.is_file() and stats_path.is_file() and not args.force:
        stats = json.loads(stats_path.read_text())
        validate_bucket_stats(stats, args)
        if args.delete_partitions:
            fragments = sorted(
                partition_dir(args.output_root, args.split, args.bucket_id).glob(
                    "*.parquet"
                )
            )
            for path in fragments:
                path.unlink()
            try:
                partition_dir(args.output_root, args.split, args.bucket_id).rmdir()
            except OSError:
                pass
        print(f"SKIP valid bucket: {output_path}", flush=True)
        return

    fragments = sorted(
        partition_dir(args.output_root, args.split, args.bucket_id).glob("*.parquet")
    )
    if not fragments:
        raise FileNotFoundError(
            f"No partition fragments for {args.split} bucket {args.bucket_id}"
        )
    started = time.time()
    table = read_partition_fragments(fragments)
    keys = table.column("shuffle_key").combine_chunks().to_numpy(zero_copy_only=False)
    expected_bucket_ids = bucket_ids(keys, args.num_buckets)
    if len(keys) and not np.all(expected_bucket_ids == args.bucket_id):
        raise ValueError(
            f"Misrouted keys found in {args.split} bucket {args.bucket_id}"
        )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if len(sorted_keys) > 1 and not np.all(sorted_keys[1:] > sorted_keys[:-1]):
        raise ValueError(
            f"Duplicate or unsorted keys in {args.split} bucket {args.bucket_id}"
        )

    sorted_input_ids = pc.take(
        table.column("input_ids"), pa.array(order, type=pa.int64())
    )
    metadata = {
        b"shuffle_method": b"splitmix64-bijective-hash-partition",
        b"shuffle_seed": str(args.seed).encode(),
        b"shuffle_num_buckets": str(args.num_buckets).encode(),
        b"shuffle_bucket_id": str(args.bucket_id).encode(),
        b"shuffle_split": args.split.encode(),
        b"global_order": b"ascending-shuffle-key-across-ascending-bucket-id",
    }
    schema = FINAL_SCHEMA.with_metadata(metadata)
    output_table = pa.Table.from_arrays([sorted_input_ids], schema=schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    pq.write_table(
        output_table,
        temporary,
        compression=args.compression,
        compression_level=args.compression_level,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=args.output_row_group_rows,
    )
    os.replace(temporary, output_path)

    stats = {
        "complete": True,
        "method": "splitmix64-bijective-hash-partition",
        "split": args.split,
        "bucket_id": args.bucket_id,
        "seed": args.seed,
        "num_buckets": args.num_buckets,
        "fragments": len(fragments),
        "partition_size_bytes": sum(path.stat().st_size for path in fragments),
        "output_path": str(output_path.resolve()),
        "output_size_bytes": output_path.stat().st_size,
        "documents": len(table),
        "tokens": token_count(table.column("input_ids")),
        "token_id_sum": token_sum(table.column("input_ids")),
        "first_shuffle_key": int(sorted_keys[0]) if len(sorted_keys) else None,
        "last_shuffle_key": int(sorted_keys[-1]) if len(sorted_keys) else None,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "output_row_group_rows": args.output_row_group_rows,
        "elapsed_seconds": time.time() - started,
    }
    write_json(stats_path, stats)
    if args.delete_partitions:
        for path in fragments:
            path.unlink()
        try:
            partition_dir(args.output_root, args.split, args.bucket_id).rmdir()
        except OSError:
            pass
    print(
        f"DONE sort split={args.split} bucket={args.bucket_id}/"
        f"{args.num_buckets} docs={len(table):,} tokens={stats['tokens']:,} "
        f"elapsed={stats['elapsed_seconds']:.1f}s",
        flush=True,
    )


def aggregate_stats(records: list[dict]) -> dict:
    result = {
        "documents": 0,
        "tokens": 0,
        "token_id_sum": 0,
    }
    for record in records:
        result["documents"] += int(record["documents"])
        result["tokens"] += int(record["tokens"])
        result["token_id_sum"] += int(record["token_id_sum"])
    return result


def finalize(args: argparse.Namespace) -> None:
    require_power_of_two(args.num_buckets)
    split_manifests = {}
    for split in ("train", "val"):
        sources = discover_files(args.source_root, split)
        partition_records = []
        for source in sources:
            path = partition_stats_path(args.output_root, source)
            if not path.is_file():
                if args.allow_partial:
                    continue
                raise FileNotFoundError(path)
            record = json.loads(path.read_text())
            if (
                record.get("seed") != args.seed
                or record.get("num_buckets") != args.num_buckets
            ):
                raise ValueError(f"Parameter mismatch in {path}")
            if not args.allow_partial and not record.get("complete_source"):
                raise ValueError(f"Partial source found in full finalize: {path}")
            partition_records.append(record)
        if not partition_records:
            raise RuntimeError(f"No partition records found for {split}")

        bucket_records = []
        previous_last = None
        for bucket_id in range(args.num_buckets):
            path = bucket_stats_path(args.output_root, split, bucket_id)
            if not path.is_file():
                raise FileNotFoundError(path)
            record = json.loads(path.read_text())
            if (
                record.get("seed") != args.seed
                or record.get("num_buckets") != args.num_buckets
            ):
                raise ValueError(f"Parameter mismatch in {path}")
            output_path = bucket_output_path(
                args.output_root, split, bucket_id, args.num_buckets
            )
            if not output_path.is_file():
                raise FileNotFoundError(output_path)
            first_key = record["first_shuffle_key"]
            last_key = record["last_shuffle_key"]
            if first_key is not None:
                if previous_last is not None and first_key <= previous_last:
                    raise ValueError(f"Non-increasing global key order before {path}")
                previous_last = last_key
            bucket_records.append(record)

        partition_totals = aggregate_stats(partition_records)
        bucket_totals = aggregate_stats(bucket_records)
        if partition_totals != bucket_totals:
            raise ValueError(
                f"{split} partition/output count mismatch:\n"
                f"partition={partition_totals}\noutput={bucket_totals}"
            )
        split_manifests[split] = {
            "source_files": len(partition_records),
            "buckets": args.num_buckets,
            **bucket_totals,
            "output_size_bytes": sum(
                int(record["output_size_bytes"]) for record in bucket_records
            ),
            "first_shuffle_key": next(
                (
                    record["first_shuffle_key"]
                    for record in bucket_records
                    if record["first_shuffle_key"] is not None
                ),
                None,
            ),
            "last_shuffle_key": next(
                (
                    record["last_shuffle_key"]
                    for record in reversed(bucket_records)
                    if record["last_shuffle_key"] is not None
                ),
                None,
            ),
        }

    source_manifest = args.source_root / "manifest.json"
    manifest = {
        "complete": not args.allow_partial,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "splitmix64-bijective-hash-partition",
        "description": (
            "Documents are globally ordered within each split by a stable unique "
            "64-bit shuffle key; final files are ordered by bucket ID."
        ),
        "seed": args.seed,
        "num_buckets": args.num_buckets,
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "source_manifest": str(source_manifest.resolve())
        if source_manifest.is_file()
        else None,
        "schema": "input_ids: list<int32>",
        "splits": split_manifests,
        "documents": sum(record["documents"] for record in split_manifests.values()),
        "tokens": sum(record["tokens"] for record in split_manifests.values()),
        "output_size_bytes": sum(
            record["output_size_bytes"] for record in split_manifests.values()
        ),
    }
    write_json(args.output_root / "manifest.json", manifest)
    if not args.allow_partial:
        (args.output_root / "_SUCCESS").write_text(
            f"hash shuffle complete seed={args.seed} buckets={args.num_buckets}\n"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-buckets", type=int, default=256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    partition = subparsers.add_parser("partition-file")
    partition.add_argument("--split", choices=("train", "val"), required=True)
    partition.add_argument("--file-index", type=int, required=True)
    partition.add_argument("--max-row-groups", type=int)
    partition.add_argument("--buffer-mib", type=int, default=8)
    partition.add_argument("--compression", default="zstd")
    partition.add_argument("--compression-level", type=int, default=3)
    partition.add_argument("--force", action="store_true")
    partition.set_defaults(func=partition_file)

    sort = subparsers.add_parser("sort-bucket")
    sort.add_argument("--split", choices=("train", "val"), required=True)
    sort.add_argument("--bucket-id", type=int, required=True)
    sort.add_argument("--output-row-group-rows", type=int, default=20_000)
    sort.add_argument("--compression", default="zstd")
    sort.add_argument("--compression-level", type=int, default=3)
    sort.add_argument("--delete-partitions", action="store_true")
    sort.add_argument("--force", action="store_true")
    sort.set_defaults(func=sort_bucket)

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--allow-partial", action="store_true")
    finish.set_defaults(func=finalize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
