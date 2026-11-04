#!/usr/bin/env python3
"""Tokenize one materialized Stack-Edu language in its existing document order.

The input files come from ``download_stackedu_full.py``. Failed downloads and
empty/null texts are skipped. Every retained document is encoded without
special tokens and receives exactly one trailing EOS token. No BOS or padding
token is used.
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


EXP_DIR = Path(__file__).resolve().parents[3]
PARCAE_DIR = EXP_DIR
sys.path.insert(0, str(PARCAE_DIR))


TOKEN_TYPE = pa.list_(pa.int32())
TOKEN_SCHEMA = pa.schema([pa.field("input_ids", TOKEN_TYPE)])


from .common import write_json


from .tokenization import tokenizer_properties, sample_indices, stats_path, output_path


def validate_source_manifest(source_manifest: Path) -> None:
    success = source_manifest.parent / "_SUCCESS"
    if not source_manifest.is_file() or not success.is_file():
        raise FileNotFoundError(
            f"Stack-Edu source is missing {source_manifest} or {success}"
        )
    payload = json.loads(source_manifest.read_text())
    if not payload.get("complete"):
        raise ValueError(f"Source manifest is not complete: {source_manifest}")


def discover_files(source_dir: Path) -> list[Path]:
    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {source_dir}")
    return files


def expected_metadata(
    *,
    source_path: Path,
    eos_id: int,
    language: str,
) -> dict[bytes, bytes]:
    return {
        b"source_path": str(source_path.resolve()).encode(),
        b"language": language.encode(),
        b"source_order": b"stackedu-materialized-source-order",
        b"row_filter": b"download_success=true-and-nonempty-text",
        b"special_tokens_policy": b"eos-only",
        b"add_bos": b"false",
        b"add_eos": b"true",
        b"eos_token_id": str(eos_id).encode(),
        b"pad_token_id": b"none",
    }


def validate_output(
    target: Path,
    *,
    expected_documents: int,
    metadata: dict[bytes, bytes],
    eos_id: int,
) -> None:
    output_file = pq.ParquetFile(target)
    schema = output_file.schema_arrow
    if schema.names != ["input_ids"] or schema.field("input_ids").type != TOKEN_TYPE:
        raise ValueError(f"Unexpected tokenized schema in {target}: {schema}")
    if output_file.metadata.num_rows != expected_documents:
        raise ValueError(
            f"Document count mismatch in {target}: "
            f"{output_file.metadata.num_rows} != {expected_documents}"
        )
    actual_metadata = schema.metadata or {}
    for key, value in metadata.items():
        if actual_metadata.get(key) != value:
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
    files = discover_files(args.source_dir)
    if args.file_index < 0 or args.file_index >= len(files):
        raise IndexError(f"file index {args.file_index} outside [0, {len(files)})")
    source_path = files[args.file_index]
    target = output_path(args.output_root, source_path)
    target_stats = stats_path(args.output_root, source_path)
    validate_source_manifest(args.source_manifest)
    (
        tokenizer,
        eos_id,
    ) = tokenizer_properties(args.tokenizer_path)
    metadata = expected_metadata(
        source_path=source_path,
        eos_id=eos_id,
        language=args.language,
    )

    if target.is_file() and target_stats.is_file() and not args.force:
        record = json.loads(target_stats.read_text())
        validate_output(
            target,
            expected_documents=int(record["documents"]),
            metadata=metadata,
            eos_id=eos_id,
        )
        print(f"SKIP valid output file={args.file_index}: {target}", flush=True)
        return record
    if target.exists() or target_stats.exists():
        target.unlink(missing_ok=True)
        target_stats.unlink(missing_ok=True)

    source_file = pq.ParquetFile(source_path)
    required = {"text", "download_success"}
    missing = required - set(source_file.schema_arrow.names)
    if missing:
        raise ValueError(f"{source_path}: missing columns {sorted(missing)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target_stats.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getenv('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)
    schema = TOKEN_SCHEMA.with_metadata(metadata)
    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression=args.compression,
        compression_level=args.compression_level,
        use_dictionary=False,
        write_statistics=True,
    )
    started = time.time()
    source_rows = 0
    documents = 0
    failed_downloads = 0
    empty_or_null_texts = 0
    tokens_without_eos = 0
    output_tokens = 0
    serialized_characters = 0
    checked_reference = False
    try:
        for row_group_index in range(source_file.num_row_groups):
            table = source_file.read_row_group(
                row_group_index, columns=["text", "download_success"]
            )
            texts_raw = table.column("text").to_pylist()
            success_raw = table.column("download_success").to_pylist()
            source_rows += len(texts_raw)
            texts: list[str] = []
            for text, success in zip(texts_raw, success_raw):
                if success is not True:
                    failed_downloads += 1
                elif not isinstance(text, str) or not text:
                    empty_or_null_texts += 1
                else:
                    texts.append(text)

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
                    if not checked_reference:
                        reference = tokenizer.encode(
                            text, eos=True, return_tensors=False
                        )
                        if row != reference:
                            raise ValueError(
                                "Batched EOS encoding differs from reference"
                            )
                        checked_reference = True
            if token_rows:
                writer.write_table(
                    pa.Table.from_arrays(
                        [pa.array(token_rows, type=TOKEN_TYPE)], schema=schema
                    ),
                    row_group_size=len(token_rows),
                )
            documents += len(token_rows)
            output_tokens += sum(len(row) for row in token_rows)
            if (
                row_group_index + 1
            ) % 20 == 0 or row_group_index + 1 == source_file.num_row_groups:
                elapsed = max(time.time() - started, 1e-9)
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

    if source_rows != source_file.metadata.num_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Source row count mismatch while reading {source_path}")
    if source_rows != documents + failed_downloads + empty_or_null_texts:
        temporary.unlink(missing_ok=True)
        raise ValueError("Stack-Edu row filtering accounting mismatch")
    if output_tokens != tokens_without_eos + documents:
        temporary.unlink(missing_ok=True)
        raise ValueError("EOS accounting mismatch")
    os.replace(temporary, target)
    validate_output(
        target,
        expected_documents=documents,
        metadata=metadata,
        eos_id=eos_id,
    )
    record = {
        "complete": True,
        "file_index": args.file_index,
        "source_path": str(source_path.resolve()),
        "output_path": str(target.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "output_size_bytes": target.stat().st_size,
        "source_rows": source_rows,
        "documents": documents,
        "failed_downloads_filtered": failed_downloads,
        "empty_or_null_texts_filtered": empty_or_null_texts,
        "serialized_characters": serialized_characters,
        "tokens_without_eos": tokens_without_eos,
        "eos_tokens_appended": documents,
        "tokens": output_tokens,
        "source_row_groups": source_file.num_row_groups,
        "language": args.language,
        "eos_token_id": eos_id,
        "special_tokens_policy": "eos-only",
        "compression": args.compression,
        "compression_level": args.compression_level,
        "elapsed_seconds": time.time() - started,
    }
    write_json(target_stats, record)
    print(
        f"DONE file={args.file_index + 1}/{len(files)} source_rows={source_rows:,} "
        f"docs={documents:,} filtered={failed_downloads + empty_or_null_texts:,} "
        f"tokens={output_tokens:,} elapsed={record['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return record


def finalize(args: argparse.Namespace) -> dict:
    files = discover_files(args.source_dir)
    validate_source_manifest(args.source_manifest)
    (
        _,
        eos_id,
    ) = tokenizer_properties(args.tokenizer_path)
    records = []
    for index, source_path in enumerate(files):
        target = output_path(args.output_root, source_path)
        sidecar = stats_path(args.output_root, source_path)
        if not target.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Missing output or stats for {source_path}")
        record = json.loads(sidecar.read_text())
        expected = {
            "file_index": index,
            "source_path": str(source_path.resolve()),
            "language": args.language,
            "eos_token_id": eos_id,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"Stats mismatch {key!r}: {sidecar}")
        validate_output(
            target,
            expected_documents=int(record["documents"]),
            metadata=expected_metadata(
                source_path=source_path,
                eos_id=eos_id,
                language=args.language,
            ),
            eos_id=eos_id,
        )
        records.append(record)
        print(f"verified {index + 1}/{len(files)} {target.name}", flush=True)

    def total(key: str) -> int:
        return sum(int(record[key]) for record in records)

    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(args.source_dir.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_order": "stackedu-materialized-source-order",
        "row_filter": "download_success=true-and-nonempty-text",
        "output_root": str(args.output_root.resolve()),
        "language": args.language,
        "files": len(files),
        "source_rows": total("source_rows"),
        "documents": total("documents"),
        "failed_downloads_filtered": total("failed_downloads_filtered"),
        "empty_or_null_texts_filtered": total("empty_or_null_texts_filtered"),
        "serialized_characters": total("serialized_characters"),
        "tokens_without_eos": total("tokens_without_eos"),
        "eos_tokens_appended": total("eos_tokens_appended"),
        "tokens": total("tokens"),
        "source_size_bytes": total("source_size_bytes"),
        "output_size_bytes": total("output_size_bytes"),
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
        f"SUCCESS files={len(files)} source_rows={manifest['source_rows']:,} "
        f"documents={manifest['documents']:,} tokens={manifest['tokens']:,}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("count")
    convert = subparsers.add_parser("convert-file")
    convert.add_argument("--file-index", type=int, required=True)
    convert.add_argument("--text-batch-size", type=int, default=128)
    convert.add_argument("--force", action="store_true")
    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "count":
        validate_source_manifest(args.source_manifest)
        print(len(discover_files(args.source_dir)))
    elif args.command == "convert-file":
        convert_file(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
