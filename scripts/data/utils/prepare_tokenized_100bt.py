#!/usr/bin/env python3
"""Tokenize global-shuffled FineWeb-Edu parquet files for pqds-fix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

EXP_DIR = Path(__file__).resolve().parents[3]
PARCAE_DIR = EXP_DIR
sys.path.insert(0, str(PARCAE_DIR))

from loopedlm.tokenizer import Tokenizer  # noqa: E402
from scripts.data.utils.packing_reference import BestFitPackingCollator  # noqa: E402
from scripts.data.utils.parquet_reference import ParquetStreamPure  # noqa: E402

DEFAULT_SOURCE_ROOT = EXP_DIR / "data" / "fineweb_edu_100bt_shuffled_r8"
DEFAULT_OUTPUT_ROOT = EXP_DIR / "data" / "100bt_tokenized"
DEFAULT_TOKENIZER_PATH = EXP_DIR / "tokenizer" / "parcae_tokenizer"
TOKEN_SCHEMA = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])


@dataclass(frozen=True)
class SourceFile:
    index: int
    split: str
    path: Path


from .common import write_json


def discover_files(source_root: Path) -> list[SourceFile]:
    discovered: list[SourceFile] = []
    for split in ("train", "val"):
        split_dir = source_root / split
        paths = sorted(split_dir.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found under {split_dir}")
        for path in paths:
            discovered.append(SourceFile(len(discovered), split, path))
    return discovered


def target_path(output_root: Path, source: SourceFile) -> Path:
    return output_root / source.split / source.path.name


def stats_path(output_root: Path, source: SourceFile) -> Path:
    return output_root / ".stats" / source.split / f"{source.path.name}.json"


def row_group_rows(parquet_file: pq.ParquetFile) -> list[int]:
    return [
        parquet_file.metadata.row_group(index).num_rows
        for index in range(parquet_file.num_row_groups)
    ]


def validate_output_structure(
    source: SourceFile,
    output_path: Path,
) -> None:
    if not output_path.is_file():
        raise FileNotFoundError(output_path)

    source_file = pq.ParquetFile(source.path)
    output_file = pq.ParquetFile(output_path)
    output_schema = output_file.schema_arrow
    if output_schema.names != ["input_ids"]:
        raise ValueError(f"{output_path}: unexpected columns {output_schema.names}")
    if output_schema.field("input_ids").type != TOKEN_SCHEMA.field("input_ids").type:
        raise ValueError(
            f"{output_path}: expected {TOKEN_SCHEMA.field('input_ids').type}, "
            f"got {output_schema.field('input_ids').type}"
        )
    if output_file.metadata.num_rows != source_file.metadata.num_rows:
        raise ValueError(
            f"{output_path}: row count mismatch "
            f"{output_file.metadata.num_rows} != {source_file.metadata.num_rows}"
        )
    if row_group_rows(output_file) != row_group_rows(source_file):
        raise ValueError(f"{output_path}: row-group boundaries differ from source")
    codecs = {
        output_file.metadata.row_group(index).column(0).compression
        for index in range(output_file.num_row_groups)
    }
    if codecs != {"ZSTD"}:
        raise ValueError(f"{output_path}: expected ZSTD compression, got {codecs}")


def recover_file_stats(
    source: SourceFile,
    output_path: Path,
) -> dict:
    output_file = pq.ParquetFile(output_path)
    tokens = 0
    for row_group_index in range(output_file.num_row_groups):
        token_rows = output_file.read_row_group(
            row_group_index, columns=["input_ids"]
        ).column("input_ids")
        tokens += int(pc.sum(pc.list_value_length(token_rows)).as_py() or 0)
    return {
        "complete": True,
        "file_index": source.index,
        "split": source.split,
        "source_path": str(source.path),
        "output_path": str(output_path),
        "source_size_bytes": source.path.stat().st_size,
        "output_size_bytes": output_path.stat().st_size,
        "documents": output_file.metadata.num_rows,
        "tokens": tokens,
        "row_groups": output_file.num_row_groups,
        "dtype": "int32",
        "compression": "zstd",
        "compression_level": 3,
        "elapsed_seconds": None,
        "stats_recovered": True,
    }


def encode_text_batch(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    if not texts:
        return []
    encoded = tokenizer.processor(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return [[int(token) for token in tokens] for tokens in encoded]


def convert_file(
    source: SourceFile,
    output_root: Path,
    tokenizer_path: Path,
    text_batch_size: int,
    compression: str,
    compression_level: int,
    force: bool,
    max_row_groups: int | None = None,
) -> dict:
    output_path = target_path(output_root, source)
    output_stats_path = stats_path(output_root, source)

    if output_path.exists() and not force and max_row_groups is None:
        validate_output_structure(
            source,
            output_path,
        )
        if not output_stats_path.is_file():
            recovered = recover_file_stats(
                source,
                output_path,
            )
            write_json(output_stats_path, recovered)
            print(
                f"RECOVERED missing stats for valid output: {output_path}",
                flush=True,
            )
        print(f"SKIP valid output: {output_path}", flush=True)
        return json.loads(output_stats_path.read_text())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getenv('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)

    tokenizer = Tokenizer(tokenizer_path)
    source_file = pq.ParquetFile(source.path)
    num_row_groups = source_file.num_row_groups
    if max_row_groups is not None:
        num_row_groups = min(num_row_groups, max_row_groups)

    schema = TOKEN_SCHEMA.with_metadata(
        {
            b"source_filename": source.path.name.encode(),
            b"source_split": source.split.encode(),
            b"special_tokens_added": b"false",
        }
    )
    started = time.time()
    documents = 0
    tokens = 0

    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=False,
        write_statistics=True,
    )
    try:
        for row_group_index in range(num_row_groups):
            source_table = source_file.read_row_group(row_group_index, columns=["text"])
            texts = source_table.column("text").to_pylist()
            if any(not isinstance(text, str) for text in texts):
                raise TypeError(
                    f"{source.path} row group {row_group_index} contains "
                    "non-string text"
                )

            token_rows: list[list[int]] = []
            for start in range(0, len(texts), text_batch_size):
                batch_texts = texts[start : start + text_batch_size]
                encoded = encode_text_batch(tokenizer, batch_texts)
                if row_group_index == 0 and start == 0:
                    for offset in range(min(8, len(batch_texts))):
                        reference = tokenizer.encode(
                            batch_texts[offset], return_tensors=False
                        )
                        if encoded[offset] != reference:
                            raise ValueError(
                                "Batched tokenizer output differs from Tokenizer.encode"
                            )
                token_rows.extend(encoded)

            token_array = pa.array(
                token_rows, type=TOKEN_SCHEMA.field("input_ids").type
            )
            output_table = pa.Table.from_arrays([token_array], schema=schema)
            writer.write_table(output_table, row_group_size=len(token_rows))
            documents += len(token_rows)
            row_tokens = sum(len(row) for row in token_rows)
            tokens += row_tokens
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"file={source.path.name} rg={row_group_index + 1}/"
                f"{num_row_groups} docs={documents:,} tokens={tokens:,} "
                f"tok/s={tokens / elapsed:,.0f}",
                flush=True,
            )
    except Exception:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    if max_row_groups is None:
        os.replace(temporary, output_path)
        validate_output_structure(
            source,
            output_path,
        )
    else:
        output_path = temporary

    payload = {
        "complete": max_row_groups is None,
        "file_index": source.index,
        "split": source.split,
        "source_path": str(source.path),
        "output_path": str(output_path),
        "source_size_bytes": source.path.stat().st_size,
        "output_size_bytes": output_path.stat().st_size,
        "documents": documents,
        "tokens": tokens,
        "row_groups": num_row_groups,
        "dtype": "int32",
        "compression": compression,
        "compression_level": compression_level,
        "elapsed_seconds": time.time() - started,
    }
    if max_row_groups is None:
        write_json(output_stats_path, payload)
    return payload


def selected_files(
    files: list[SourceFile], indices: Iterable[int] | None
) -> list[SourceFile]:
    if indices is None:
        return files
    selected_indices = list(indices)
    by_index = {source.index: source for source in files}
    missing = [index for index in selected_indices if index not in by_index]
    if missing:
        raise IndexError(f"Unknown file indices: {missing}")
    return [by_index[index] for index in selected_indices]


def sample_indices(length: int) -> list[int]:
    if length <= 0:
        return []
    return sorted({0, length // 2, length - 1})


def verify_token_samples(
    source: SourceFile,
    output_path: Path,
    tokenizer: Tokenizer,
) -> int:
    source_file = pq.ParquetFile(source.path)
    output_file = pq.ParquetFile(output_path)
    compared = 0
    for row_group_index in sample_indices(source_file.num_row_groups):
        text_rows = (
            source_file.read_row_group(row_group_index, columns=["text"])
            .column("text")
            .to_pylist()
        )
        token_rows = (
            output_file.read_row_group(row_group_index, columns=["input_ids"])
            .column("input_ids")
            .to_pylist()
        )
        for row_index in sample_indices(len(text_rows)):
            expected = tokenizer.encode(text_rows[row_index], return_tensors=False)
            actual = token_rows[row_index]
            if actual != expected:
                raise ValueError(
                    f"Token mismatch in {source.path.name}, row group "
                    f"{row_group_index}, row {row_index}"
                )
            compared += 1
    return compared


def compare_pqds_loaders(
    source: SourceFile,
    output_path: Path,
    tokenizer_path: Path,
    batches: int,
    batch_size: int,
    block_size: int,
) -> dict:
    if batches <= 0:
        return {"batches": 0, "examples": 0}

    tokenizer = Tokenizer(tokenizer_path)
    with tempfile.TemporaryDirectory(prefix="pqds-tokenized-compare-") as temp:
        temp_root = Path(temp)
        raw_dir = temp_root / "raw"
        tokenized_dir = temp_root / "tokenized"
        raw_dir.mkdir()
        tokenized_dir.mkdir()
        os.symlink(source.path, raw_dir / source.path.name)
        os.symlink(output_path, tokenized_dir / output_path.name)

        common = {
            "seed": 42,
            "shuffle": True,
            "num_processes": 1,
            "process_rank": 0,
            "shuffle_filenames": True,
            "filename_seed": 42,
            "stateful": False,
            "plan_for_later_rank_expansion_to": 0,
            "verbose": False,
        }
        raw_dataset = ParquetStreamPure(
            dataset_folder_path=str(raw_dir),
            data_signature={"keys": ["text"], "format_fn": "pass_text"},
            tokenizer=tokenizer,
            **common,
        )
        tokenized_dataset = ParquetStreamPure(
            dataset_folder_path=str(tokenized_dir),
            data_signature={"keys": ["input_ids"]},
            tokenizer=tokenizer,
            **common,
        )
        raw_collator = BestFitPackingCollator(
            tokenizer=tokenizer,
            block_size=block_size,
            add_bos=True,
            add_eos=True,
            buffer_size=1000,
        )
        tokenized_collator = BestFitPackingCollator(
            tokenizer=tokenizer,
            block_size=block_size,
            add_bos=True,
            add_eos=True,
            buffer_size=1000,
        )
        raw_loader = DataLoader(
            raw_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=raw_collator,
            num_workers=0,
        )
        tokenized_loader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=tokenized_collator,
            num_workers=0,
        )
        raw_iterator = iter(raw_loader)
        tokenized_iterator = iter(tokenized_loader)
        for batch_index in range(batches):
            raw_batch = next(raw_iterator)
            tokenized_batch = next(tokenized_iterator)
            if not torch.equal(raw_batch[0], tokenized_batch[0]):
                raise ValueError(f"input_ids differ at packed batch {batch_index}")
            if not torch.equal(raw_batch[1], tokenized_batch[1]):
                raise ValueError(f"labels differ at packed batch {batch_index}")
            if batch_index % 25 == 0 or batch_index + 1 == batches:
                print(
                    f"loader A/B matched {batch_index + 1}/{batches} batches",
                    flush=True,
                )
    return {"batches": batches, "examples": batches * batch_size}


def verify_dataset(
    source_root: Path,
    output_root: Path,
    tokenizer_path: Path,
    indices: list[int] | None,
    compare_batches: int,
    compare_batch_size: int,
    compare_block_size: int,
    write_success: bool,
) -> dict:
    all_files = discover_files(source_root)
    files = selected_files(all_files, indices)
    if write_success and len(files) != len(all_files):
        raise ValueError("Cannot write _SUCCESS for a partial dataset")
    tokenizer = Tokenizer(tokenizer_path)
    file_stats = []
    sample_comparisons = 0
    total_row_groups = 0
    total_documents = 0
    total_tokens = 0
    total_source_bytes = 0
    total_output_bytes = 0

    for source in files:
        output_path = target_path(output_root, source)
        validate_output_structure(
            source,
            output_path,
        )
        sample_comparisons += verify_token_samples(source, output_path, tokenizer)
        sidecar = stats_path(output_root, source)
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        stats = json.loads(sidecar.read_text())
        if stats.get("documents") != pq.ParquetFile(source.path).metadata.num_rows:
            raise ValueError(f"{sidecar}: document count mismatch")
        file_stats.append(stats)
        total_row_groups += int(stats["row_groups"])
        total_documents += int(stats["documents"])
        total_tokens += int(stats["tokens"])
        total_source_bytes += int(stats["source_size_bytes"])
        total_output_bytes += int(stats["output_size_bytes"])
        print(
            f"verified {source.index + 1}/{len(all_files)} "
            f"{source.split}/{source.path.name}",
            flush=True,
        )

    if compare_batches > 0:
        train_file = next((source for source in files if source.split == "train"), None)
        if train_file is None:
            raise ValueError(
                "Loader comparison requires at least one selected train file"
            )
        loader_comparison = compare_pqds_loaders(
            train_file,
            target_path(output_root, train_file),
            tokenizer_path,
            batches=compare_batches,
            batch_size=compare_batch_size,
            block_size=compare_block_size,
        )
    else:
        loader_comparison = {"batches": 0, "examples": 0}
    payload = {
        "complete": write_success,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "tokenizer_path": str(tokenizer_path),
        "schema": "input_ids: list<int32>",
        "special_tokens_added": False,
        "compression": "zstd",
        "files": len(files),
        "train_files": sum(source.split == "train" for source in files),
        "val_files": sum(source.split == "val" for source in files),
        "row_groups": total_row_groups,
        "documents": total_documents,
        "tokens": total_tokens,
        "source_size_bytes": total_source_bytes,
        "output_size_bytes": total_output_bytes,
        "sample_token_comparisons": sample_comparisons,
        "loader_comparison": loader_comparison,
        "file_stats": file_stats,
    }
    if write_success:
        write_json(output_root / "manifest.json", payload)
        success_path = output_root / "_SUCCESS"
        success_path.write_text(
            json.dumps(
                {
                    "manifest": "manifest.json",
                    "files": len(files),
                    "tokens": total_tokens,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return payload


def parse_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part) for part in value.split(",")]


def create_test_source(root: Path) -> None:
    schema = pa.schema([pa.field("text", pa.string())])
    examples = {
        "train": [
            ["hello world", "", "numbers 12 34"],
            ["second row group", "unicode text", "last train document"],
        ],
        "val": [
            ["validation one", "validation two"],
            ["validation final"],
        ],
    }
    for split, row_groups in examples.items():
        split_dir = root / split
        split_dir.mkdir(parents=True)
        path = split_dir / (
            "train-00000-of-00002.parquet"
            if split == "train"
            else "train-00001-of-00002.parquet"
        )
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        for rows in row_groups:
            writer.write_table(
                pa.Table.from_arrays([pa.array(rows, type=pa.string())], schema=schema),
                row_group_size=len(rows),
            )
        writer.close()


def run_self_test(tokenizer_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="tokenize-100bt-self-test-") as temp:
        root = Path(temp)
        source_root = root / "source"
        output_root = root / "output"
        create_test_source(source_root)
        files = discover_files(source_root)
        for source in files:
            convert_file(
                source,
                output_root,
                tokenizer_path,
                text_batch_size=2,
                compression="zstd",
                compression_level=3,
                force=False,
            )
        first_stats = stats_path(output_root, files[0])
        first_stats.unlink()
        recovered = convert_file(
            files[0],
            output_root,
            tokenizer_path,
            text_batch_size=2,
            compression="zstd",
            compression_level=3,
            force=False,
        )
        assert recovered["stats_recovered"] is True
        result = verify_dataset(
            source_root,
            output_root,
            tokenizer_path,
            indices=None,
            compare_batches=1,
            compare_batch_size=2,
            compare_block_size=16,
            write_success=True,
        )
        assert result["files"] == 2
        assert (output_root / "_SUCCESS").is_file()
        print(json.dumps(result, indent=2, sort_keys=True))
        print("SELF TEST PASSED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--json", action="store_true")

    convert_parser = subparsers.add_parser("convert-file")
    convert_parser.add_argument("--file-index", type=int, required=True)
    convert_parser.add_argument("--text-batch-size", type=int, default=512)
    convert_parser.add_argument("--compression", default="zstd")
    convert_parser.add_argument("--compression-level", type=int, default=3)
    convert_parser.add_argument("--force", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--indices")
    verify_parser.add_argument("--compare-batches", type=int, default=305)
    verify_parser.add_argument("--compare-batch-size", type=int, default=64)
    verify_parser.add_argument("--compare-block-size", type=int, default=2049)
    verify_parser.add_argument("--write-success", action="store_true")
    subparsers.add_parser("self-test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        files = discover_files(args.source_root)
        payload = [
            {
                "index": source.index,
                "split": source.split,
                "path": str(source.path),
            }
            for source in files
        ]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for item in payload:
                print(f"{item['index']:03d} {item['split']:5s} {item['path']}")
    elif args.command == "convert-file":
        files = discover_files(args.source_root)
        source = selected_files(files, [args.file_index])[0]
        result = convert_file(
            source,
            args.output_root,
            args.tokenizer_path,
            text_batch_size=args.text_batch_size,
            compression=args.compression,
            compression_level=args.compression_level,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "verify":
        result = verify_dataset(
            args.source_root,
            args.output_root,
            args.tokenizer_path,
            indices=parse_indices(args.indices),
            compare_batches=args.compare_batches,
            compare_batch_size=args.compare_batch_size,
            compare_block_size=args.compare_block_size,
            write_success=args.write_success,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "self-test":
        run_self_test(args.tokenizer_path)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
