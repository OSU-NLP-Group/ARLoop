#!/usr/bin/env python3
"""Restore OpenCodeReasoning split_1 questions from pinned APPS/TACO rows.

OpenCodeReasoning split_1 stores ``input == "-"`` and identifies the original
question by ``(dataset, split, index)``.  This program reproduces the official
lookup, preserves the already-shuffled row order and Parquet row groups, and
writes a new raw dataset whose ``input`` column contains the restored question.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download, snapshot_download


APPS_REPO = "codeparrot/apps"
TACO_REPO = "BAAI/TACO"
EXPECTED_LENGTHS = {
    ("apps", "train"): 5_000,
    ("apps", "test"): 5_000,
    ("taco", "train"): 25_443,
    ("taco", "test"): 1_000,
}
REQUIRED_COLUMNS = {"id", "input", "output", "dataset", "split", "index"}


from .common import write_json


def discover_files(source_root: Path) -> list[Path]:
    manifest = source_root / "manifest.json"
    success = source_root / "_SUCCESS"
    if not manifest.is_file() or not success.is_file():
        raise FileNotFoundError(f"Incomplete shuffled source: {source_root}")
    if not json.loads(manifest.read_text()).get("complete"):
        raise ValueError(f"Source manifest is not complete: {manifest}")
    files = sorted((source_root / "parquet").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {source_root / 'parquet'}")
    return files


def parse_key(
    dataset: str, split: str, index: str, *, location: str
) -> tuple[str, str, int]:
    if dataset not in {"apps", "taco"}:
        raise ValueError(f"Unsupported dataset {dataset!r} at {location}")
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split {split!r} at {location}")
    try:
        numeric_index = int(index)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid index {index!r} at {location}") from error
    expected_length = EXPECTED_LENGTHS[(dataset, split)]
    if numeric_index < 0 or numeric_index >= expected_length:
        raise IndexError(
            f"{dataset}/{split} index {numeric_index} outside [0, {expected_length}) at {location}"
        )
    return dataset, split, numeric_index


def scan_requirements(files: list[Path]):
    needed: dict[tuple[str, str], set[int]] = defaultdict(set)
    counts: Counter = Counter()
    documents = 0
    for file_index, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        missing = REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row_group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(
                row_group_index, columns=["id", "input", "dataset", "split", "index"]
            )
            values = {name: table[name].to_pylist() for name in table.column_names}
            for row_index, (row_id, old_input, dataset, split, index) in enumerate(
                zip(
                    values["id"],
                    values["input"],
                    values["dataset"],
                    values["split"],
                    values["index"],
                )
            ):
                location = (
                    f"file={file_index} row_group={row_group_index} row={row_index}"
                )
                if not isinstance(row_id, str):
                    raise TypeError(f"Non-string id at {location}")
                if old_input != "-":
                    raise ValueError(
                        f"Expected input == '-' at {location}, found {old_input!r}"
                    )
                key = parse_key(dataset, split, index, location=location)
                needed[key[:2]].add(key[2])
                counts[key[:2]] += 1
                documents += 1
    return needed, counts, documents


def load_apps_questions(
    needed: dict[tuple[str, str], set[int]], *, revision: str, cache_dir: Path
) -> dict[tuple[str, str, int], str]:
    questions: dict[tuple[str, str, int], str] = {}
    for split in ("train", "test"):
        path = Path(
            hf_hub_download(
                repo_id=APPS_REPO,
                filename=f"{split}.jsonl",
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
            )
        )
        required = needed[("apps", split)]
        line_count = 0
        with path.open("rb") as handle:
            for index, line in enumerate(handle):
                line_count += 1
                if index not in required:
                    continue
                payload = json.loads(line)
                question = payload.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"Empty APPS question at {split}[{index}]")
                questions[("apps", split, index)] = question
        expected = EXPECTED_LENGTHS[("apps", split)]
        if line_count != expected:
            raise ValueError(f"APPS {split} has {line_count} rows; expected {expected}")
        if len([key for key in questions if key[:2] == ("apps", split)]) != len(
            required
        ):
            raise ValueError(f"Failed to retrieve every required APPS {split} question")
        print(
            f"loaded APPS/{split}: rows={line_count:,} required={len(required):,}",
            flush=True,
        )
    return questions


def load_taco_questions(
    needed: dict[tuple[str, str], set[int]], *, revision: str, cache_dir: Path
) -> dict[tuple[str, str, int], str]:
    snapshot = Path(
        snapshot_download(
            repo_id=TACO_REPO,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["train/*.arrow", "test/*.arrow"],
            max_workers=16,
        )
    )
    questions: dict[tuple[str, str, int], str] = {}
    for split in ("train", "test"):
        paths = sorted((snapshot / split).glob("*.arrow"))
        if not paths:
            raise FileNotFoundError(
                f"No pinned TACO {split} Arrow files below {snapshot}"
            )
        required = needed[("taco", split)]
        offset = 0
        for path in paths:
            shard = Dataset.from_file(str(path))
            stop = offset + len(shard)
            local_indices = sorted(
                index - offset for index in required if offset <= index < stop
            )
            for local_index in local_indices:
                question = shard[local_index]["question"]
                global_index = offset + local_index
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"Empty TACO question at {split}[{global_index}]")
                questions[("taco", split, global_index)] = question
            offset = stop
        expected = EXPECTED_LENGTHS[("taco", split)]
        if offset != expected:
            raise ValueError(f"TACO {split} has {offset} rows; expected {expected}")
        if len([key for key in questions if key[:2] == ("taco", split)]) != len(
            required
        ):
            raise ValueError(f"Failed to retrieve every required TACO {split} question")
        print(
            f"loaded TACO/{split}: rows={offset:,} required={len(required):,}",
            flush=True,
        )
    return questions


def target_schema(
    source_schema: pa.Schema,
    args: argparse.Namespace,
) -> pa.Schema:
    metadata = dict(source_schema.metadata or {})
    metadata.update(
        {
            b"opencodereasoning_input_rehydrated": b"true",
            b"lookup_algorithm": b"dataset-split-index-v1",
            b"apps_repo": APPS_REPO.encode(),
            b"apps_revision": args.apps_revision.encode(),
            b"taco_repo": TACO_REPO.encode(),
            b"taco_revision": args.taco_revision.encode(),
            b"shuffle_order_preserved": b"true",
        }
    )
    return source_schema.with_metadata(metadata)


def write_rehydrated_files(
    files: list[Path],
    questions: dict[tuple[str, str, int], str],
    args: argparse.Namespace,
):
    records: list[dict] = []
    for file_index, source_path in enumerate(files):
        started = time.time()
        source_file = pq.ParquetFile(source_path)
        schema = target_schema(
            source_file.schema_arrow,
            args,
        )
        target = args.output_root / "parquet" / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        writer = pq.ParquetWriter(
            temporary,
            schema,
            compression=args.compression,
            compression_level=args.compression_level,
            use_dictionary=True,
            write_statistics=True,
        )
        rows = 0
        input_characters = 0
        try:
            for row_group_index in range(source_file.num_row_groups):
                table = source_file.read_row_group(row_group_index)
                datasets = table["dataset"].to_pylist()
                splits = table["split"].to_pylist()
                indices = table["index"].to_pylist()
                old_inputs = table["input"].to_pylist()
                ids = table["id"].to_pylist()
                restored: list[str] = []
                for row_index, (dataset, split, index, old_input, row_id) in enumerate(
                    zip(datasets, splits, indices, old_inputs, ids)
                ):
                    location = (
                        f"file={file_index} row_group={row_group_index} row={row_index}"
                    )
                    if old_input != "-":
                        raise ValueError(f"Expected input == '-' at {location}")
                    key = parse_key(dataset, split, index, location=location)
                    question = questions.get(key)
                    if question is None:
                        raise KeyError(
                            f"Missing restored question for {key} at {location}"
                        )
                    restored.append(question)
                    input_characters += len(question)
                input_index = table.schema.get_field_index("input")
                table = table.set_column(
                    input_index, "input", pa.array(restored, type=pa.string())
                )
                writer.write_table(
                    table.replace_schema_metadata(schema.metadata),
                    row_group_size=len(table),
                )
                rows += len(table)
        except Exception:
            writer.close()
            temporary.unlink(missing_ok=True)
            raise
        writer.close()
        os.replace(temporary, target)

        output_file = pq.ParquetFile(target)
        if output_file.metadata.num_rows != source_file.metadata.num_rows:
            raise ValueError(f"Row count changed while rehydrating {source_path}")
        if output_file.num_row_groups != source_file.num_row_groups:
            raise ValueError(f"Row-group count changed while rehydrating {source_path}")
        metadata = output_file.schema_arrow.metadata or {}
        if metadata.get(b"opencodereasoning_input_rehydrated") != b"true":
            raise ValueError(f"Missing rehydration metadata in {target}")
        record = {
            "file_index": file_index,
            "source_path": str(source_path.resolve()),
            "output_path": str(target.resolve()),
            "rows": rows,
            "row_groups": source_file.num_row_groups,
            "source_size_bytes": source_path.stat().st_size,
            "output_size_bytes": target.stat().st_size,
            "input_characters": input_characters,
            "elapsed_seconds": time.time() - started,
        }
        records.append(record)
        print(
            f"wrote {file_index + 1}/{len(files)} {target.name}: "
            f"rows={rows:,} input_chars={input_characters:,} "
            f"elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return records


def print_samples(output_root: Path, count: int = 3) -> None:
    remaining = count
    for path in sorted((output_root / "parquet").glob("*.parquet")):
        table = pq.read_table(
            path, columns=["id", "dataset", "split", "index", "input", "output"]
        )
        for row_index in range(min(remaining, len(table))):
            print(
                f"sample {count - remaining + 1}: "
                + json.dumps(
                    {
                        "id": table["id"][row_index].as_py(),
                        "lookup": [
                            table["dataset"][row_index].as_py(),
                            table["split"][row_index].as_py(),
                            table["index"][row_index].as_py(),
                        ],
                        "input_preview": table["input"][row_index].as_py()[:300],
                        "output_preview": table["output"][row_index].as_py()[:300],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            remaining -= 1
        if remaining == 0:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--apps-revision", required=True)
    parser.add_argument("--taco-revision", required=True)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    args = parser.parse_args()

    source_files = discover_files(args.source_root)
    needed, counts, documents = scan_requirements(source_files)
    print(
        f"scanned source: files={len(source_files)} documents={documents:,} "
        f"unique_questions={sum(map(len, needed.values())):,}",
        flush=True,
    )
    for key in sorted(counts):
        print(
            f"  {key[0]}/{key[1]}: rows={counts[key]:,} unique={len(needed[key]):,}",
            flush=True,
        )

    apps_questions = load_apps_questions(
        needed, revision=args.apps_revision, cache_dir=args.cache_dir
    )
    taco_questions = load_taco_questions(
        needed, revision=args.taco_revision, cache_dir=args.cache_dir
    )
    questions = apps_questions | taco_questions
    expected_unique = sum(map(len, needed.values()))
    if len(questions) != expected_unique:
        raise ValueError(
            f"Retrieved {len(questions)} questions; expected {expected_unique}"
        )

    records = write_rehydrated_files(
        source_files,
        questions,
        args,
    )
    if sum(record["rows"] for record in records) != documents:
        raise ValueError("Final document count mismatch")

    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "rehydrate-opencodereasoning-split1-input-v1",
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "files": records,
        "file_count": len(records),
        "documents": documents,
        "unique_questions": expected_unique,
        "counts": {
            f"{dataset}/{split}": counts[(dataset, split)]
            for dataset, split in sorted(counts)
        },
        "unique_counts": {
            f"{dataset}/{split}": len(needed[(dataset, split)])
            for dataset, split in sorted(counts)
        },
        "apps_repo": APPS_REPO,
        "apps_revision": args.apps_revision,
        "taco_repo": TACO_REPO,
        "taco_revision": args.taco_revision,
        "lookup_field": "question",
        "lookup_key": ["dataset", "split", "index"],
        "input_placeholder": "-",
        "input_rehydrated": True,
        "shuffle_order_preserved": True,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "output_size_bytes": sum(record["output_size_bytes"] for record in records),
        "input_characters": sum(record["input_characters"] for record in records),
    }
    write_json(args.output_root / "manifest.json", manifest)
    success = args.output_root / "_SUCCESS"
    success.write_text(
        json.dumps(
            {
                "manifest": "manifest.json",
                "documents": documents,
                "unique_questions": expected_unique,
            },
            sort_keys=True,
        )
        + "\n"
    )
    print_samples(args.output_root)
    print(
        f"SUCCESS documents={documents:,} unique_questions={expected_unique:,} "
        f"bytes={manifest['output_size_bytes']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
