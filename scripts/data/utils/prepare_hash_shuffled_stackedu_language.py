#!/usr/bin/env python3
"""Globally shuffle one downloaded Stack-Edu language into text-only Parquet.

Successful, non-empty documents receive unique IDs derived from their source
file and row position. SplitMix64 maps those IDs bijectively to deterministic
shuffle keys. Reading final bucket files in numeric order is therefore an
exact global permutation of the retained language documents.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SOURCE_COLUMNS = ("blob_id", "text", "download_success")
FINAL_SCHEMA = pa.schema([pa.field("text", pa.large_string(), nullable=False)])
TEMP_SCHEMA = pa.schema(
    [
        pa.field("shuffle_key", pa.uint64(), nullable=False),
        pa.field("document_id", pa.uint64(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
    ]
)


from .common import write_json
from .shuffle_utils import (
    bucket_ids,
    require_power_of_two,
    splitmix64,
)


def discover_files(source_dir: Path) -> list[Path]:
    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {source_dir}")
    if len(files) >= (1 << 32):
        raise ValueError("Document IDs support fewer than 2^32 source files")
    return files


def validate_source_schema(schema: pa.Schema, source: Path) -> None:
    missing = set(SOURCE_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"{source}: missing Stack-Edu columns {sorted(missing)}")
    if not (
        pa.types.is_string(schema.field("text").type)
        or pa.types.is_large_string(schema.field("text").type)
    ):
        raise ValueError(f"{source}: text is {schema.field('text').type}")
    if not pa.types.is_boolean(schema.field("download_success").type):
        raise ValueError(
            f"{source}: download_success is {schema.field('download_success').type}"
        )


def fragment_path(output_root: Path, bucket_id: int, lane_id: int) -> Path:
    return (
        output_root
        / "_partitions"
        / f"bucket-{bucket_id:05d}"
        / f"lane-{lane_id:05d}.parquet"
    )


def lane_stats_path(output_root: Path, lane_id: int) -> Path:
    return output_root / ".stats" / "lanes" / f"lane-{lane_id:05d}.json"


def bucket_output_path(output_root: Path, bucket_id: int, count: int) -> Path:
    return output_root / "parquet" / f"part-{bucket_id:05d}-of-{count:05d}.parquet"


def bucket_stats_path(output_root: Path, bucket_id: int) -> Path:
    return output_root / ".stats" / "buckets" / f"bucket-{bucket_id:05d}.json"


def validate_common(record: dict, args: argparse.Namespace) -> None:
    expected = {
        "language": args.language,
        "seed": args.seed,
        "num_buckets": args.num_buckets,
        "num_lanes": args.num_lanes,
        "source_dir": str(args.source_dir.resolve()),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"Stats mismatch {key}={record.get(key)!r}, expected {value!r}"
            )


class PartitionWriters:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.buffer_bytes = args.buffer_mib * 1024 * 1024
        self.buffers: dict[int, list[pa.Table]] = {}
        self.buffer_sizes: dict[int, int] = {}
        self.writers: dict[int, pq.ParquetWriter] = {}
        self.temporaries: dict[int, Path] = {}
        self.documents = np.zeros(args.num_buckets, dtype=np.int64)
        self.row_groups = np.zeros(args.num_buckets, dtype=np.int64)

    def writer(self, bucket_id: int) -> pq.ParquetWriter:
        writer = self.writers.get(bucket_id)
        if writer is not None:
            return writer
        final_path = fragment_path(self.args.output_root, bucket_id, self.args.lane_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_path.with_name(
            f".{final_path.name}.tmp.{os.getenv('SLURM_JOB_ID', 'local')}.{os.getpid()}"
        )
        temporary.unlink(missing_ok=True)
        writer = pq.ParquetWriter(
            temporary,
            TEMP_SCHEMA,
            compression=self.args.compression,
            compression_level=self.args.compression_level,
            use_dictionary=True,
            write_statistics=True,
        )
        self.writers[bucket_id] = writer
        self.temporaries[bucket_id] = temporary
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
        self.writer(bucket_id).write_table(table, row_group_size=len(table))
        self.row_groups[bucket_id] += 1

    def finish(self) -> list[dict]:
        for bucket_id in sorted(self.buffers):
            self.flush(bucket_id)
        for writer in self.writers.values():
            writer.close()
        records = []
        for bucket_id, temporary in sorted(self.temporaries.items()):
            final_path = fragment_path(
                self.args.output_root, bucket_id, self.args.lane_id
            )
            os.replace(temporary, final_path)
            records.append(
                {
                    "bucket_id": bucket_id,
                    "documents": int(self.documents[bucket_id]),
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
        for temporary in self.temporaries.values():
            temporary.unlink(missing_ok=True)


def validate_completed_lane(args: argparse.Namespace, record: dict) -> None:
    validate_common(record, args)
    if record.get("lane_id") != args.lane_id or not record.get("complete"):
        raise ValueError(f"Invalid lane stats for lane {args.lane_id}")
    for bucket in record["buckets"]:
        bucket_id = int(bucket["bucket_id"])
        fragment = fragment_path(args.output_root, bucket_id, args.lane_id)
        if fragment.is_file() and fragment.stat().st_size == int(bucket["size_bytes"]):
            continue
        completed = bucket_output_path(args.output_root, bucket_id, args.num_buckets)
        completed_stats = bucket_stats_path(args.output_root, bucket_id)
        if not completed.is_file() or not completed_stats.is_file():
            raise FileNotFoundError(fragment)


def partition_lane(args: argparse.Namespace) -> None:
    require_power_of_two(args.num_buckets)
    files = discover_files(args.source_dir)
    if args.num_lanes > len(files):
        raise ValueError(f"num_lanes={args.num_lanes} exceeds files={len(files)}")
    if args.lane_id < 0 or args.lane_id >= args.num_lanes:
        raise IndexError(f"lane {args.lane_id} outside [0, {args.num_lanes})")
    assigned = list(range(args.lane_id, len(files), args.num_lanes))
    stats_path = lane_stats_path(args.output_root, args.lane_id)
    if stats_path.is_file() and not args.force:
        record = json.loads(stats_path.read_text())
        validate_completed_lane(args, record)
        print(f"SKIP valid lane={args.lane_id}", flush=True)
        return

    stats_path.unlink(missing_ok=True)
    for bucket_id in range(args.num_buckets):
        fragment_path(args.output_root, bucket_id, args.lane_id).unlink(missing_ok=True)

    writers = PartitionWriters(args)
    started = time.time()
    source_rows = documents = failed_downloads = empty_texts = 0
    source_size_bytes = 0
    try:
        for lane_file_position, file_index in enumerate(assigned):
            source = files[file_index]
            parquet_file = pq.ParquetFile(source)
            validate_source_schema(parquet_file.schema_arrow, source)
            row_offset = 0
            for row_group_index in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(
                    row_group_index, columns=list(SOURCE_COLUMNS)
                )
                rows = len(table)
                if row_offset + rows >= (1 << 32):
                    raise ValueError(f"Too many rows for document IDs: {source}")
                source_rows += rows

                success = table.column("download_success").combine_chunks()
                success_values = pc.fill_null(success, False).to_numpy(
                    zero_copy_only=False
                )
                success_mask = np.asarray(success_values, dtype=bool)
                failed_downloads += int(rows - success_mask.sum())

                text = table.column("text").combine_chunks()
                valid_text = np.asarray(
                    pc.is_valid(text).to_numpy(zero_copy_only=False), dtype=bool
                )
                lengths = np.asarray(
                    pc.fill_null(pc.utf8_length(text), 0).to_numpy(),
                    dtype=np.int64,
                )
                retained_mask = success_mask & valid_text & (lengths > 0)
                empty_texts += int(success_mask.sum() - retained_mask.sum())
                retained_positions = np.flatnonzero(retained_mask)
                row_offset_after = row_offset + rows
                if len(retained_positions):
                    original_rows = retained_positions.astype(np.uint64) + np.uint64(
                        row_offset
                    )
                    document_ids = original_rows | np.uint64(file_index << 32)
                    keys = splitmix64(document_ids, args.seed)
                    ids = bucket_ids(keys, args.num_buckets)
                    order = np.argsort(ids, kind="stable")
                    sorted_ids = ids[order]
                    boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
                    starts = np.concatenate(([0], boundaries))
                    ends = np.concatenate((boundaries, [len(order)]))
                    retained_text = pc.filter(text, pa.array(retained_mask)).cast(
                        pa.large_string()
                    )
                    for start, end in zip(starts, ends):
                        take_indices = order[start:end]
                        bucket_id = int(sorted_ids[start])
                        partition = pa.Table.from_arrays(
                            [
                                pa.array(keys[take_indices], type=pa.uint64()),
                                pa.array(document_ids[take_indices], type=pa.uint64()),
                                pc.take(
                                    retained_text,
                                    pa.array(take_indices, type=pa.int64()),
                                ),
                            ],
                            schema=TEMP_SCHEMA,
                        )
                        writers.append(bucket_id, partition)

                    documents += len(retained_positions)
                row_offset = row_offset_after

            source_size_bytes += source.stat().st_size
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"lane={args.lane_id}/{args.num_lanes} "
                f"file={lane_file_position + 1}/{len(assigned)} "
                f"docs={documents:,} source_GB={source_size_bytes / 1e9:.2f} "
                f"source_MB/s={source_size_bytes / 1e6 / elapsed:,.1f}",
                flush=True,
            )
        bucket_records = writers.finish()
    except Exception:
        writers.abort()
        raise

    if len(bucket_records) != args.num_buckets:
        raise ValueError(
            f"lane {args.lane_id} populated "
            f"{len(bucket_records)}/{args.num_buckets} buckets"
        )
    elapsed = time.time() - started
    record = {
        "complete": True,
        "language": args.language,
        "lane_id": args.lane_id,
        "num_lanes": args.num_lanes,
        "seed": args.seed,
        "num_buckets": args.num_buckets,
        "source_dir": str(args.source_dir.resolve()),
        "source_file_indices": assigned,
        "source_files": len(assigned),
        "source_rows": source_rows,
        "documents": documents,
        "failed_downloads": failed_downloads,
        "empty_texts": empty_texts,
        "source_size_bytes": source_size_bytes,
        "partition_size_bytes": sum(x["size_bytes"] for x in bucket_records),
        "elapsed_seconds": elapsed,
        "buckets": bucket_records,
    }
    write_json(stats_path, record)
    print(
        f"DONE partition language={args.language} lane={args.lane_id} "
        f"docs={documents:,} elapsed={elapsed:.1f}s",
        flush=True,
    )


def delete_fragments(args: argparse.Namespace, fragments: list[Path]) -> None:
    for fragment in fragments:
        fragment.unlink(missing_ok=True)
    bucket_dir = args.output_root / "_partitions" / f"bucket-{args.bucket_id:05d}"
    try:
        bucket_dir.rmdir()
    except OSError:
        pass


def sort_bucket(args: argparse.Namespace) -> None:
    require_power_of_two(args.num_buckets)
    if args.bucket_id < 0 or args.bucket_id >= args.num_buckets:
        raise IndexError(f"bucket {args.bucket_id} outside [0, {args.num_buckets})")
    output = bucket_output_path(args.output_root, args.bucket_id, args.num_buckets)
    stats_path = bucket_stats_path(args.output_root, args.bucket_id)
    fragments = [
        fragment_path(args.output_root, args.bucket_id, lane_id)
        for lane_id in range(args.num_lanes)
    ]
    if output.is_file() and stats_path.is_file() and not args.force:
        record = json.loads(stats_path.read_text())
        validate_common(record, args)
        if output.stat().st_size != int(record["output_size_bytes"]):
            raise ValueError(f"Output size mismatch: {output}")
        if args.delete_partitions:
            delete_fragments(args, fragments)
        print(f"SKIP valid bucket={args.bucket_id}", flush=True)
        return
    output.unlink(missing_ok=True)
    stats_path.unlink(missing_ok=True)
    missing = [path for path in fragments if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"bucket {args.bucket_id} missing {len(missing)}/{args.num_lanes} fragments"
        )

    started = time.time()
    tables = []
    for fragment in fragments:
        parquet_file = pq.ParquetFile(fragment)
        if parquet_file.schema_arrow.remove_metadata() != TEMP_SCHEMA:
            raise ValueError(f"Partition schema mismatch: {fragment}")
        tables.append(parquet_file.read())
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    del tables
    keys = table.column("shuffle_key").combine_chunks().to_numpy(zero_copy_only=False)
    ids = bucket_ids(keys, args.num_buckets)
    if len(ids) and not np.all(ids == args.bucket_id):
        raise ValueError(f"Misrouted rows in bucket {args.bucket_id}")
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if len(sorted_keys) > 1 and not np.all(sorted_keys[1:] > sorted_keys[:-1]):
        raise ValueError(f"Duplicate or unsorted keys in bucket {args.bucket_id}")

    metadata = {
        b"shuffle_method": b"splitmix64-bijective-hash-partition",
        b"shuffle_seed": str(args.seed).encode(),
        b"shuffle_num_buckets": str(args.num_buckets).encode(),
        b"shuffle_bucket_id": str(args.bucket_id).encode(),
        b"shuffle_num_lanes": str(args.num_lanes).encode(),
        b"language": args.language.encode(),
        b"global_order": b"ascending-shuffle-key-across-ascending-bucket-id",
    }
    final_schema = FINAL_SCHEMA.with_metadata(metadata)
    sorted_text = pc.take(
        table.column("text").combine_chunks(), pa.array(order, type=pa.int64())
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getenv('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        temporary,
        final_schema,
        compression=args.compression,
        compression_level=args.compression_level,
        use_dictionary=False,
        write_statistics=True,
    )
    try:
        for start in range(0, len(order), args.output_row_group_rows):
            chunk = sorted_text.slice(start, args.output_row_group_rows)
            writer.write_table(
                pa.Table.from_arrays([chunk], schema=final_schema),
                row_group_size=len(chunk),
            )
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(temporary, output)

    record = {
        "complete": True,
        "language": args.language,
        "bucket_id": args.bucket_id,
        "num_buckets": args.num_buckets,
        "num_lanes": args.num_lanes,
        "seed": args.seed,
        "source_dir": str(args.source_dir.resolve()),
        "documents": len(table),
        "partition_size_bytes": sum(path.stat().st_size for path in fragments),
        "output_path": str(output.resolve()),
        "output_size_bytes": output.stat().st_size,
        "first_shuffle_key": int(sorted_keys[0]) if len(sorted_keys) else None,
        "last_shuffle_key": int(sorted_keys[-1]) if len(sorted_keys) else None,
        "elapsed_seconds": time.time() - started,
    }
    write_json(stats_path, record)
    if args.delete_partitions:
        delete_fragments(args, fragments)
    print(
        f"DONE sort language={args.language} bucket={args.bucket_id}/"
        f"{args.num_buckets} docs={len(table):,}",
        flush=True,
    )


def aggregate(records: list[dict]) -> dict:
    result = {
        "source_rows": 0,
        "documents": 0,
        "failed_downloads": 0,
        "empty_texts": 0,
    }
    for record in records:
        for key in ("source_rows", "documents", "failed_downloads", "empty_texts"):
            result[key] += int(record.get(key, 0))
    return result


def finalize(args: argparse.Namespace) -> None:
    require_power_of_two(args.num_buckets)
    files = discover_files(args.source_dir)
    lane_records = []
    source_indices: list[int] = []
    for lane_id in range(args.num_lanes):
        path = lane_stats_path(args.output_root, lane_id)
        record = json.loads(path.read_text())
        validate_common(record, args)
        lane_records.append(record)
        source_indices.extend(int(index) for index in record["source_file_indices"])
    if sorted(source_indices) != list(range(len(files))):
        raise ValueError("Lane source-file assignment is incomplete or duplicated")

    bucket_records = []
    previous_last = None
    footer_documents = 0
    for bucket_id in range(args.num_buckets):
        stats_path = bucket_stats_path(args.output_root, bucket_id)
        record = json.loads(stats_path.read_text())
        validate_common(record, args)
        output = bucket_output_path(args.output_root, bucket_id, args.num_buckets)
        if not output.is_file() or output.stat().st_size != record["output_size_bytes"]:
            raise FileNotFoundError(output)
        parquet_file = pq.ParquetFile(output)
        if parquet_file.schema_arrow.remove_metadata() != FINAL_SCHEMA:
            raise ValueError(f"Final schema mismatch: {output}")
        if parquet_file.metadata.num_rows != record["documents"]:
            raise ValueError(f"Footer row-count mismatch: {output}")
        footer_documents += parquet_file.metadata.num_rows
        first_key = record["first_shuffle_key"]
        last_key = record["last_shuffle_key"]
        if (
            first_key is not None
            and previous_last is not None
            and first_key <= previous_last
        ):
            raise ValueError(f"Global key order fails before bucket {bucket_id}")
        if last_key is not None:
            previous_last = last_key
        bucket_records.append(record)

    lane_totals = aggregate(lane_records)
    bucket_totals = aggregate(bucket_records)
    if lane_totals["documents"] != bucket_totals["documents"]:
        raise ValueError(
            f"Lane/output count mismatch: lanes={lane_totals} "
            f"buckets={bucket_totals}"
        )
    if footer_documents != lane_totals["documents"]:
        raise ValueError("Final footer document total differs from lane stats")
    if (
        lane_totals["documents"]
        + lane_totals["failed_downloads"]
        + lane_totals["empty_texts"]
        != lane_totals["source_rows"]
    ):
        raise ValueError("Retained/skipped Stack-Edu row accounting mismatch")

    source_manifest = args.source_manifest.resolve()
    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "language": args.language,
        "method": "splitmix64-bijective-hash-partition",
        "description": (
            "Successful non-empty documents are globally ordered by a unique "
            "64-bit shuffle key; final files are read in ascending bucket ID."
        ),
        "source_dir": str(args.source_dir.resolve()),
        "source_manifest": str(source_manifest),
        "output_root": str(args.output_root.resolve()),
        "seed": args.seed,
        "num_lanes": args.num_lanes,
        "num_buckets": args.num_buckets,
        "source_files": len(files),
        "final_files": len(bucket_records),
        "schema": str(FINAL_SCHEMA),
        **lane_totals,
        "source_size_bytes": sum(x["source_size_bytes"] for x in lane_records),
        "partition_size_bytes": sum(x["partition_size_bytes"] for x in lane_records),
        "output_size_bytes": sum(x["output_size_bytes"] for x in bucket_records),
        "first_shuffle_key": next(
            (
                x["first_shuffle_key"]
                for x in bucket_records
                if x["first_shuffle_key"] is not None
            ),
            None,
        ),
        "last_shuffle_key": next(
            (
                x["last_shuffle_key"]
                for x in reversed(bucket_records)
                if x["last_shuffle_key"] is not None
            ),
            None,
        ),
    }
    write_json(args.output_root / "manifest.json", manifest)
    (args.output_root / "_SUCCESS").write_text(
        f"Stack-Edu language global shuffle complete language={args.language} "
        f"seed={args.seed} documents={manifest['documents']}\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-buckets", type=int, default=128)
    parser.add_argument("--num-lanes", type=int, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    partition = commands.add_parser("partition-lane")
    partition.add_argument("--lane-id", type=int, required=True)
    partition.add_argument("--buffer-mib", type=float, default=0.25)
    partition.add_argument("--compression", default="zstd")
    partition.add_argument("--compression-level", type=int, default=3)
    partition.add_argument("--force", action="store_true")
    partition.set_defaults(func=partition_lane)
    sort = commands.add_parser("sort-bucket")
    sort.add_argument("--bucket-id", type=int, required=True)
    sort.add_argument("--output-row-group-rows", type=int, default=2048)
    sort.add_argument("--compression", default="zstd")
    sort.add_argument("--compression-level", type=int, default=3)
    sort.add_argument("--delete-partitions", action="store_true")
    sort.add_argument("--force", action="store_true")
    sort.set_defaults(func=sort_bucket)
    finish = commands.add_parser("finalize")
    finish.set_defaults(func=finalize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_manifest.is_file():
        raise FileNotFoundError(args.source_manifest)
    args.func(args)


if __name__ == "__main__":
    main()
