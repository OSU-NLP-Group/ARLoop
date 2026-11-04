#!/usr/bin/env python3
"""Tokenize shuffled curated datasets with the official SmolLM2 tokenizer.

Each source row remains one document.  Dataset-specific text fields are
serialized first, encoded with ``add_special_tokens=False``, and exactly one
EOS token is appended.  No BOS or padding token is added.  Output Parquet
files contain only ``input_ids: list<int32>``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


TOKEN_TYPE = pa.list_(pa.int32())
TOKEN_SCHEMA = pa.schema([pa.field("input_ids", TOKEN_TYPE)])
SERIALIZER_COLUMNS = {
    "text": ["text"],
    "thread-text": ["ThreadText"],
    "problem-generated-solution": ["problem", "generated_solution"],
    "input-output": ["input", "output"],
    "output-only": ["output"],
}


from .common import write_json


def validate_source_manifest(source_root: Path) -> None:
    success = source_root / "_SUCCESS"
    manifest = source_root / "manifest.json"
    if not success.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Incomplete shuffled source: {source_root}")
    payload = json.loads(manifest.read_text())
    if not payload.get("complete"):
        raise ValueError(f"Source manifest is not complete: {manifest}")


def discover_files(source_root: Path) -> list[Path]:
    files = sorted((source_root / "parquet").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {source_root / 'parquet'}")
    return files


def join_pair(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n\n{right}"


def require_strings(values: list, *, column: str, source_path: Path) -> list[str]:
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{source_path}: non-string or null value in {column}")
    return values


def serialize_table(table: pa.Table, serializer: str, source_path: Path) -> list[str]:
    if serializer == "text":
        return require_strings(
            table.column("text").to_pylist(), column="text", source_path=source_path
        )
    if serializer == "thread-text":
        return require_strings(
            table.column("ThreadText").to_pylist(),
            column="ThreadText",
            source_path=source_path,
        )
    if serializer == "problem-generated-solution":
        problems = require_strings(
            table.column("problem").to_pylist(),
            column="problem",
            source_path=source_path,
        )
        solutions = require_strings(
            table.column("generated_solution").to_pylist(),
            column="generated_solution",
            source_path=source_path,
        )
        return [
            join_pair(problem, solution)
            for problem, solution in zip(problems, solutions)
        ]
    if serializer == "input-output":
        inputs = require_strings(
            table.column("input").to_pylist(), column="input", source_path=source_path
        )
        outputs = require_strings(
            table.column("output").to_pylist(), column="output", source_path=source_path
        )
        missing = sum(not value.strip() or value.strip() == "-" for value in inputs)
        if missing:
            raise ValueError(
                f"{source_path}: input-output serializer found {missing} missing inputs"
            )
        return [
            join_pair(input_text, output) for input_text, output in zip(inputs, outputs)
        ]
    if serializer == "output-only":
        return require_strings(
            table.column("output").to_pylist(), column="output", source_path=source_path
        )
    raise ValueError(f"Unknown serializer: {serializer}")


from .tokenization import tokenizer_properties, sample_indices, stats_path, output_path


def expected_metadata(
    *,
    source_path: Path,
    serializer: str,
    eos_id: int,
) -> dict[bytes, bytes]:
    return {
        b"source_path": str(source_path.resolve()).encode(),
        b"serializer": serializer.encode(),
        b"source_columns": json.dumps(SERIALIZER_COLUMNS[serializer]).encode(),
        b"special_tokens_policy": b"eos-only",
        b"add_bos": b"false",
        b"add_eos": b"true",
        b"eos_token_id": str(eos_id).encode(),
        b"pad_token_id": b"none",
    }


def validate_output(
    source_path: Path,
    target: Path,
    *,
    serializer: str,
    eos_id: int,
) -> None:
    source_file = pq.ParquetFile(source_path)
    output_file = pq.ParquetFile(target)
    schema = output_file.schema_arrow
    if schema.names != ["input_ids"] or schema.field("input_ids").type != TOKEN_TYPE:
        raise ValueError(f"Unexpected tokenized schema in {target}: {schema}")
    if output_file.metadata.num_rows != source_file.metadata.num_rows:
        raise ValueError(f"Document count mismatch: {target}")
    if output_file.num_row_groups != source_file.num_row_groups:
        raise ValueError(f"Row-group count mismatch: {target}")
    metadata = schema.metadata or {}
    expected = expected_metadata(
        source_path=source_path,
        serializer=serializer,
        eos_id=eos_id,
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Metadata mismatch for {key!r}: {target}")
    for row_group_index in sample_indices(output_file.num_row_groups):
        rows = output_file.read_row_group(
            row_group_index, columns=["input_ids"]
        ).column("input_ids")
        for row_index in sample_indices(len(rows)):
            tokens = rows[row_index].as_py()
            if not tokens or int(tokens[-1]) != eos_id:
                raise ValueError(f"Sample document does not end in EOS: {target}")


def convert_file(args: argparse.Namespace) -> dict:
    files = discover_files(args.source_root)
    if args.file_index < 0 or args.file_index >= len(files):
        raise IndexError(f"file index {args.file_index} outside [0, {len(files)})")
    source_path = files[args.file_index]
    target = output_path(args.output_root, source_path)
    target_stats = stats_path(args.output_root, source_path)
    validate_source_manifest(args.source_root)
    (
        tokenizer,
        eos_id,
    ) = tokenizer_properties(args.tokenizer_path)

    if target.is_file() and target_stats.is_file() and not args.force:
        validate_output(
            source_path,
            target,
            serializer=args.serializer,
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
    source_file = pq.ParquetFile(source_path)
    source_names = set(source_file.schema_arrow.names)
    missing_columns = set(SERIALIZER_COLUMNS[args.serializer]) - source_names
    if missing_columns:
        raise ValueError(f"{source_path}: missing columns {sorted(missing_columns)}")

    schema = TOKEN_SCHEMA.with_metadata(
        expected_metadata(
            source_path=source_path,
            serializer=args.serializer,
            eos_id=eos_id,
        )
    )
    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression=args.compression,
        compression_level=args.compression_level,
        use_dictionary=False,
        write_statistics=True,
    )
    started = time.time()
    documents = 0
    tokens_without_eos = 0
    output_tokens = 0
    serialized_characters = 0
    try:
        for row_group_index in range(source_file.num_row_groups):
            table = source_file.read_row_group(
                row_group_index, columns=SERIALIZER_COLUMNS[args.serializer]
            )
            texts = serialize_table(table, args.serializer, source_path)
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
                    serialized_characters += len(text)
                    if documents == 0 and len(token_rows) <= 8:
                        reference = tokenizer.encode(
                            text, eos=True, return_tensors=False
                        )
                        if row != reference:
                            raise ValueError(
                                "Batched EOS encoding differs from reference"
                            )
            writer.write_table(
                pa.Table.from_arrays(
                    [pa.array(token_rows, type=TOKEN_TYPE)], schema=schema
                ),
                row_group_size=len(token_rows),
            )
            documents += len(token_rows)
            output_tokens += sum(len(row) for row in token_rows)
            elapsed = max(time.time() - started, 1e-9)
            if (
                row_group_index + 1
            ) % 20 == 0 or row_group_index + 1 == source_file.num_row_groups:
                print(
                    f"file={args.file_index + 1}/{len(files)} "
                    f"rg={row_group_index + 1}/{source_file.num_row_groups} "
                    f"docs={documents:,} tokens={output_tokens:,} "
                    f"tok/s={output_tokens / elapsed:,.0f}",
                    flush=True,
                )
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()

    if documents != source_file.metadata.num_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Document count mismatch while writing {source_path}")
    if output_tokens != tokens_without_eos + documents:
        temporary.unlink(missing_ok=True)
        raise ValueError("EOS accounting mismatch")
    os.replace(temporary, target)
    validate_output(
        source_path,
        target,
        serializer=args.serializer,
        eos_id=eos_id,
    )
    record = {
        "complete": True,
        "file_index": args.file_index,
        "source_path": str(source_path.resolve()),
        "output_path": str(target.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "output_size_bytes": target.stat().st_size,
        "documents": documents,
        "serialized_characters": serialized_characters,
        "tokens_without_eos": tokens_without_eos,
        "eos_tokens_appended": documents,
        "tokens": output_tokens,
        "row_groups": source_file.num_row_groups,
        "serializer": args.serializer,
        "source_columns": SERIALIZER_COLUMNS[args.serializer],
        "eos_token_id": eos_id,
        "special_tokens_policy": "eos-only",
        "compression": args.compression,
        "compression_level": args.compression_level,
        "elapsed_seconds": time.time() - started,
    }
    write_json(target_stats, record)
    print(
        f"DONE file={args.file_index + 1}/{len(files)} docs={documents:,} "
        f"tokens={output_tokens:,} elapsed={record['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return record


def finalize(args: argparse.Namespace) -> dict:
    files = discover_files(args.source_root)
    validate_source_manifest(args.source_root)
    (
        tokenizer,
        eos_id,
    ) = tokenizer_properties(args.tokenizer_path)
    records = []
    for index, source_path in enumerate(files):
        target = output_path(args.output_root, source_path)
        sidecar = stats_path(args.output_root, source_path)
        if not target.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Missing output or stats for {source_path}")
        validate_output(
            source_path,
            target,
            serializer=args.serializer,
            eos_id=eos_id,
        )
        record = json.loads(sidecar.read_text())
        expected = {
            "file_index": index,
            "source_path": str(source_path.resolve()),
            "serializer": args.serializer,
            "eos_token_id": eos_id,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"Stats mismatch {key!r}: {sidecar}")
        records.append(record)
        print(f"verified {index + 1}/{len(files)} {target.name}", flush=True)

    def total(key: str) -> int:
        return sum(int(record[key]) for record in records)

    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "files": len(files),
        "documents": total("documents"),
        "serialized_characters": total("serialized_characters"),
        "tokens_without_eos": total("tokens_without_eos"),
        "eos_tokens_appended": total("eos_tokens_appended"),
        "tokens": total("tokens"),
        "source_size_bytes": total("source_size_bytes"),
        "output_size_bytes": total("output_size_bytes"),
        "row_groups": total("row_groups"),
        "serializer": args.serializer,
        "source_columns": SERIALIZER_COLUMNS[args.serializer],
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "schema": "input_ids: list<int32>",
        "special_tokens_policy": "eos-only",
        "add_bos": False,
        "add_eos": True,
        "eos_token_id": eos_id,
        "pad_token_id": None,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "file_stats": records,
    }
    write_json(args.output_root / "manifest.json", manifest)
    success = args.output_root / "_SUCCESS"
    success.write_text(
        json.dumps(
            {
                "manifest": "manifest.json",
                "files": len(files),
                "documents": manifest["documents"],
                "tokens": manifest["tokens"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"SUCCESS files={len(files)} documents={manifest['documents']:,} "
        f"tokens={manifest['tokens']:,} bytes={manifest['output_size_bytes']:,}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument(
        "--serializer", choices=sorted(SERIALIZER_COLUMNS), required=True
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("count")
    convert = subparsers.add_parser("convert-file")
    convert.add_argument("--file-index", type=int, required=True)
    convert.add_argument("--text-batch-size", type=int, default=256)
    convert.add_argument("--force", action="store_true")
    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "count":
        validate_source_manifest(args.source_root)
        print(len(discover_files(args.source_root)))
    elif args.command == "convert-file":
        convert_file(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
