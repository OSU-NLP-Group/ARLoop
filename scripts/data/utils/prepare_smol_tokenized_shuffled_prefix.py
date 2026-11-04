#!/usr/bin/env python3
"""Tokenize a globally shuffled FineWeb-Edu prefix with EOS-only boundaries.

The prefix is defined by the raw FineWeb-Edu ``token_count`` column in global
shuffle order. Whole documents are selected until the requested source-token
target is reached; the final included document may overshoot the target. Text
is encoded with ``add_special_tokens=False`` and exactly one tokenizer EOS ID
is appended to every document. No BOS or padding token is added.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


TOKEN_SCHEMA = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])


from .common import write_json
from .tokenization import tokenizer_properties


def load_shuffle_manifest(source_root: Path) -> dict:
    success = source_root / "_SUCCESS"
    manifest_path = source_root / "manifest.json"
    if not success.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Shuffled source is incomplete: {source_root}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError(f"Shuffled source manifest is incomplete: {manifest_path}")
    return manifest


def bucket_stats_path(source_root: Path, bucket_id: int) -> Path:
    return source_root / ".stats" / "buckets" / f"bucket-{bucket_id:05d}.json"


def boundary_prefix(path: Path, requested_tokens: int) -> tuple[int, int]:
    if requested_tokens <= 0:
        raise ValueError("Boundary prefix must request a positive token count")
    parquet_file = pq.ParquetFile(path)
    documents = 0
    tokens = 0
    for row_group_index in range(parquet_file.num_row_groups):
        column = (
            parquet_file.read_row_group(row_group_index, columns=["token_count"])
            .column("token_count")
            .combine_chunks()
        )
        if column.null_count:
            raise ValueError(f"Null token_count values in {path}")
        values = column.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        group_tokens = int(np.sum(values, dtype=np.int64))
        if tokens + group_tokens < requested_tokens:
            documents += len(values)
            tokens += group_tokens
            continue
        cumulative = np.cumsum(values, dtype=np.int64)
        offset = int(
            np.searchsorted(cumulative, requested_tokens - tokens, side="left")
        )
        included = offset + 1
        documents += included
        tokens += int(cumulative[offset])
        return documents, tokens
    raise ValueError(
        f"Requested {requested_tokens:,} tokens exceeds boundary file total in {path}"
    )


def build_plan(args: argparse.Namespace) -> dict:
    shuffle_manifest = load_shuffle_manifest(args.source_root)
    num_buckets = int(shuffle_manifest["num_buckets"])
    selected = []
    source_tokens = 0
    documents = 0
    for bucket_id in range(num_buckets):
        stats_path = bucket_stats_path(args.source_root, bucket_id)
        stats = json.loads(stats_path.read_text())
        source_path = Path(stats["output_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        bucket_tokens = int(stats["token_count"])
        bucket_documents = int(stats["documents"])
        if source_tokens + bucket_tokens <= args.target_source_tokens:
            selected_documents = bucket_documents
            selected_tokens = bucket_tokens
            partial = False
        else:
            remaining = args.target_source_tokens - source_tokens
            selected_documents, selected_tokens = boundary_prefix(
                source_path, remaining
            )
            partial = selected_documents != bucket_documents
        selected.append(
            {
                "file_index": len(selected),
                "bucket_id": bucket_id,
                "source_path": str(source_path.resolve()),
                "source_filename": source_path.name,
                "source_file_size_bytes": source_path.stat().st_size,
                "source_file_documents": bucket_documents,
                "selected_documents": selected_documents,
                "selected_source_tokens": selected_tokens,
                "partial": partial,
            }
        )
        source_tokens += selected_tokens
        documents += selected_documents
        if source_tokens >= args.target_source_tokens:
            break
    if source_tokens < args.target_source_tokens:
        raise ValueError(
            f"Dataset has only {source_tokens:,} source tokens, below requested "
            f"{args.target_source_tokens:,}"
        )

    core = {
        "version": 1,
        "selection": "global-shuffle-prefix-by-source-token_count-document-boundary",
        "source_root": str(args.source_root.resolve()),
        "shuffle_seed": shuffle_manifest["seed"],
        "shuffle_num_buckets": num_buckets,
        "target_source_tokens": args.target_source_tokens,
        "selected_source_tokens": source_tokens,
        "selected_documents": documents,
        "selected_files": len(selected),
        "files": selected,
    }
    plan = {
        **core,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = args.output_root / "selection_plan.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if {k: existing.get(k) for k in core} != core:
            outputs = list((args.output_root / "data").glob("*.parquet"))
            if outputs:
                raise ValueError(
                    f"Existing selection plan differs and {len(outputs)} outputs exist"
                )
    write_json(path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    return plan


def load_plan(output_root: Path) -> dict:
    path = output_root / "selection_plan.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    plan = json.loads(path.read_text())
    return plan


def output_path(output_root: Path, entry: dict) -> Path:
    return output_root / "data" / entry["source_filename"]


def stats_path(output_root: Path, entry: dict) -> Path:
    return output_root / ".stats" / f"{entry['source_filename']}.json"


def validate_output(
    path: Path,
    entry: dict,
    *,
    eos_id: int,
) -> None:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    if (
        schema.names != ["input_ids"]
        or schema.field("input_ids").type != TOKEN_SCHEMA.field("input_ids").type
    ):
        raise ValueError(f"Unexpected tokenized schema: {path}: {schema}")
    if parquet_file.metadata.num_rows != entry["selected_documents"]:
        raise ValueError(f"Document count mismatch: {path}")
    metadata = schema.metadata or {}
    expected = {
        b"special_tokens_policy": b"eos-only",
        b"add_bos": b"false",
        b"add_eos": b"true",
        b"eos_token_id": str(eos_id).encode(),
        b"pad_token_id": b"none",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Metadata mismatch for {key!r}: {path}")
    sample_row_groups = sorted(
        {0, parquet_file.num_row_groups // 2, parquet_file.num_row_groups - 1}
    )
    for row_group_index in sample_row_groups:
        column = parquet_file.read_row_group(
            row_group_index, columns=["input_ids"]
        ).column("input_ids")
        for row_index in sorted({0, len(column) // 2, len(column) - 1}):
            tokens = column[row_index].as_py()
            if not tokens or int(tokens[-1]) != eos_id:
                raise ValueError(f"Sample document does not end in EOS: {path}")


def convert_file(args: argparse.Namespace) -> dict:
    plan = load_plan(args.output_root)
    if args.file_index < 0 or args.file_index >= len(plan["files"]):
        raise IndexError(
            f"file index {args.file_index} outside [0, {len(plan['files'])})"
        )
    entry = plan["files"][args.file_index]
    source_path = Path(entry["source_path"])
    target = output_path(args.output_root, entry)
    target_stats = stats_path(args.output_root, entry)
    (
        tokenizer,
        eos_id,
    ) = tokenizer_properties(args.tokenizer_path)
    if target.is_file() and target_stats.is_file() and not args.force:
        validate_output(
            target,
            entry,
            eos_id=eos_id,
        )
        print(f"SKIP valid output file={args.file_index}: {target}", flush=True)
        return json.loads(target_stats.read_text())
    if target.exists() or target_stats.exists():
        target.unlink(missing_ok=True)
        target_stats.unlink(missing_ok=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    target_stats.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getenv('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)
    schema = TOKEN_SCHEMA.with_metadata(
        {
            b"source_path": str(source_path).encode(),
            b"source_bucket_id": str(entry["bucket_id"]).encode(),
            b"special_tokens_policy": b"eos-only",
            b"add_bos": b"false",
            b"add_eos": b"true",
            b"eos_token_id": str(eos_id).encode(),
            b"pad_token_id": b"none",
        }
    )
    source_file = pq.ParquetFile(source_path)
    remaining_documents = int(entry["selected_documents"])
    documents = 0
    source_tokens = 0
    tokens_without_eos = 0
    output_tokens = 0
    output_row_groups = 0
    started = time.time()
    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression=args.compression,
        compression_level=args.compression_level,
        use_dictionary=False,
        write_statistics=True,
    )
    try:
        for row_group_index in range(source_file.num_row_groups):
            if remaining_documents <= 0:
                break
            take_rows = min(
                remaining_documents,
                source_file.metadata.row_group(row_group_index).num_rows,
            )
            table = source_file.read_row_group(
                row_group_index, columns=["text", "token_count"]
            ).slice(0, take_rows)
            texts = table.column("text").to_pylist()
            if any(not isinstance(text, str) for text in texts):
                raise TypeError(f"Non-string text in {source_path}")
            token_rows: list[list[int]] = []
            for start in range(0, len(texts), args.text_batch_size):
                batch_texts = texts[start : start + args.text_batch_size]
                encoded = tokenizer.processor(
                    batch_texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                for text, tokens in zip(batch_texts, encoded):
                    row = [int(token) for token in tokens]
                    tokens_without_eos += len(row)
                    row.append(eos_id)
                    token_rows.append(row)
                    if documents == 0 and len(token_rows) <= 8:
                        reference = tokenizer.encode(
                            text, eos=True, return_tensors=False
                        )
                        if row != reference:
                            raise ValueError(
                                "Batched EOS-only encoding differs from reference"
                            )
            token_array = pa.array(
                token_rows, type=TOKEN_SCHEMA.field("input_ids").type
            )
            writer.write_table(
                pa.Table.from_arrays([token_array], schema=schema),
                row_group_size=len(token_rows),
            )
            row_source_tokens = int(pc.sum(table.column("token_count")).as_py() or 0)
            documents += len(token_rows)
            source_tokens += row_source_tokens
            output_tokens += sum(len(row) for row in token_rows)
            output_row_groups += 1
            remaining_documents -= len(token_rows)
            elapsed = max(time.time() - started, 1e-9)
            if output_row_groups % 100 == 0 or remaining_documents == 0:
                print(
                    f"file={args.file_index + 1}/{len(plan['files'])} "
                    f"rg={output_row_groups}/{source_file.num_row_groups} "
                    f"docs={documents:,} tokens={output_tokens:,} "
                    f"tok/s={output_tokens / elapsed:,.0f}",
                    flush=True,
                )
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    if remaining_documents != 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Source ended with {remaining_documents} selected documents missing"
        )
    if source_tokens != int(entry["selected_source_tokens"]):
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Source token mismatch {source_tokens} != {entry['selected_source_tokens']}"
        )
    if output_tokens != tokens_without_eos + documents:
        temporary.unlink(missing_ok=True)
        raise ValueError("EOS accounting mismatch")
    os.replace(temporary, target)
    validate_output(
        target,
        entry,
        eos_id=eos_id,
    )
    record = {
        "complete": True,
        "file_index": args.file_index,
        "bucket_id": entry["bucket_id"],
        "source_path": str(source_path),
        "output_path": str(target.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "output_size_bytes": target.stat().st_size,
        "documents": documents,
        "source_tokens": source_tokens,
        "tokens_without_eos": tokens_without_eos,
        "eos_tokens_appended": documents,
        "tokens": output_tokens,
        "row_groups": output_row_groups,
        "eos_token_id": eos_id,
        "special_tokens_policy": "eos-only",
        "compression": args.compression,
        "compression_level": args.compression_level,
        "elapsed_seconds": time.time() - started,
    }
    write_json(target_stats, record)
    print(
        f"DONE tokenize file={args.file_index + 1}/{len(plan['files'])} "
        f"docs={documents:,} source_tokens={source_tokens:,} "
        f"tokens={output_tokens:,} elapsed={record['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return record


def finalize(args: argparse.Namespace) -> dict:
    plan = load_plan(args.output_root)
    (
        _,
        eos_id,
    ) = tokenizer_properties(args.tokenizer_path)
    records = []
    for entry in plan["files"]:
        target = output_path(args.output_root, entry)
        target_stats = stats_path(args.output_root, entry)
        if not target_stats.is_file():
            raise FileNotFoundError(target_stats)
        validate_output(
            target,
            entry,
            eos_id=eos_id,
        )
        record = json.loads(target_stats.read_text())
        if not record.get("complete"):
            raise ValueError(f"Invalid output stats: {target_stats}")
        records.append(record)
    documents = sum(int(record["documents"]) for record in records)
    source_tokens = sum(int(record["source_tokens"]) for record in records)
    eos_tokens = sum(int(record["eos_tokens_appended"]) for record in records)
    if (
        documents != plan["selected_documents"]
        or source_tokens != plan["selected_source_tokens"]
    ):
        raise ValueError("Final totals differ from selection plan")
    if eos_tokens != documents:
        raise ValueError("Exactly one appended EOS per document was not recorded")
    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "selection_plan": str((args.output_root / "selection_plan.json").resolve()),
        "target_source_tokens": plan["target_source_tokens"],
        "selected_source_tokens": source_tokens,
        "documents": documents,
        "files": len(records),
        "partial_final_file": bool(plan["files"][-1]["partial"]),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "schema": "input_ids: list<int32>",
        "special_tokens_policy": "eos-only",
        "add_bos": False,
        "add_eos": True,
        "eos_token_id": eos_id,
        "pad_token_id": None,
        "tokens_without_eos": sum(
            int(record["tokens_without_eos"]) for record in records
        ),
        "eos_tokens_appended": eos_tokens,
        "tokens": sum(int(record["tokens"]) for record in records),
        "source_size_bytes": sum(
            int(record["source_size_bytes"]) for record in records
        ),
        "output_size_bytes": sum(
            int(record["output_size_bytes"]) for record in records
        ),
        "compression": "zstd",
        "compression_level": 3,
    }
    write_json(args.output_root / "manifest.json", manifest)
    write_json(
        args.output_root / "_SUCCESS",
        {
            "manifest": "manifest.json",
            "documents": documents,
            "tokens": manifest["tokens"],
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def plan_count(args: argparse.Namespace) -> None:
    print(len(load_plan(args.output_root)["files"]))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--target-source-tokens", type=int, default=700_000_000_000)
    plan.set_defaults(func=build_plan)

    count = commands.add_parser("plan-count")
    count.set_defaults(func=plan_count)

    convert = commands.add_parser("convert-file")
    convert.add_argument("--file-index", type=int, required=True)
    convert.add_argument("--text-batch-size", type=int, default=512)
    convert.add_argument("--compression", default="zstd")
    convert.add_argument("--compression-level", type=int, default=3)
    convert.add_argument("--force", action="store_true")
    convert.set_defaults(func=convert_file)

    finish = commands.add_parser("finalize")
    finish.set_defaults(func=finalize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
