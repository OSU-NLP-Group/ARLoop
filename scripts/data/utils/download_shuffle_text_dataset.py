#!/usr/bin/env python3
"""Download a pinned HF JSON corpus, globally shuffle it, and write text Parquet.

The physical output contains one column, ``text: string``.  Dolmino sources
already provide that field.  Natural Reasoning rows are serialized as
``question + "\n\n" + responses[0].response`` before shuffling.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as paj
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError


OUTPUT_SCHEMA = pa.schema([pa.field("text", pa.string())])
_SHUFFLED_DATASET: Dataset | None = None


from .common import write_json


def inspect_parquet(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "size_bytes": path.stat().st_size,
        "schema": str(parquet.schema_arrow),
    }


def expected_shard_rows(total_rows: int, num_shards: int, shard_id: int) -> int:
    quotient, remainder = divmod(total_rows, num_shards)
    return quotient + int(shard_id < remainder)


def write_one_shard(task: tuple[int, int, str, int, str, int]) -> dict[str, Any]:
    shard_id, num_shards, target_text, batch_rows, compression, compression_level = task
    if _SHUFFLED_DATASET is None:
        raise RuntimeError("Worker did not inherit the shuffled dataset")
    pa.set_cpu_count(1)
    target = Path(target_text)
    shard = _SHUFFLED_DATASET.shard(
        num_shards=num_shards, index=shard_id, contiguous=True
    )
    expected_rows = len(shard)
    if target.is_file():
        existing = inspect_parquet(target)
        if (
            existing["rows"] == expected_rows
            and pq.ParquetFile(target).schema_arrow == OUTPUT_SCHEMA
        ):
            existing.update({"shard_id": shard_id, "reused": True})
            return existing
        raise ValueError(f"Refusing to reuse invalid shard: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    started = time.time()
    arrow_shard = shard.with_format("arrow")
    with pq.ParquetWriter(
        temporary,
        schema=OUTPUT_SCHEMA,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=False,
        write_statistics=True,
    ) as writer:
        for batch in arrow_shard.iter(batch_size=batch_rows):
            table = pa.Table.from_arrays([batch["text"]], schema=OUTPUT_SCHEMA)
            writer.write_table(table, row_group_size=len(table))
    os.replace(temporary, target)
    result = inspect_parquet(target)
    if result["rows"] != expected_rows:
        raise ValueError(f"{target}: rows={result['rows']} expected={expected_rows}")
    if pq.ParquetFile(target).schema_arrow != OUTPUT_SCHEMA:
        raise ValueError(f"Unexpected schema in {target}")
    result.update(
        {
            "shard_id": shard_id,
            "reused": False,
            "elapsed_seconds": time.time() - started,
        }
    )
    return result


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def get_dataset_info_with_rate_limit_retry(
    api: HfApi, repo_id: str, revision: str, max_attempts: int = 12
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return api.dataset_info(repo_id, revision=revision, files_metadata=True)
        except HfHubHTTPError as error:
            response = error.response
            if (
                response is None
                or response.status_code != 429
                or attempt == max_attempts
            ):
                raise
            retry_after = response.headers.get("retry-after", "60")
            try:
                wait_seconds = max(1.0, float(retry_after)) + 2.0
            except ValueError:
                wait_seconds = 62.0
            print(
                f"HF API rate limited; retrying dataset metadata in {wait_seconds:.0f}s "
                f"(attempt {attempt}/{max_attempts})",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def download_source_files(
    args: argparse.Namespace,
) -> tuple[list[Path], list[dict[str, Any]], str]:
    api = HfApi()
    info = get_dataset_info_with_rate_limit_retry(api, args.repo_id, args.revision)
    if info.sha != args.revision:
        raise ValueError(f"Pinned revision changed: {args.revision} -> {info.sha}")
    selected = []
    for sibling in info.siblings:
        remote_path = sibling.rfilename
        if not matches_any(remote_path, args.include_pattern):
            continue
        if args.exclude_pattern and matches_any(remote_path, args.exclude_pattern):
            continue
        selected.append(
            {
                "remote_path": remote_path,
                "size_bytes": int(sibling.size or 0),
            }
        )
    selected.sort(key=lambda item: item["remote_path"])
    if not selected:
        raise FileNotFoundError(
            f"No files selected from {args.repo_id}@{args.revision}: {args.include_pattern}"
        )

    args.download_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.download_root,
        allow_patterns=args.include_pattern,
        ignore_patterns=args.exclude_pattern or None,
        max_workers=args.download_workers,
    )
    local_files = [args.download_root / item["remote_path"] for item in selected]
    missing = [str(path) for path in local_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} downloaded files; first={missing[0]}"
        )
    for item, path in zip(selected, local_files):
        if item["size_bytes"] and path.stat().st_size != item["size_bytes"]:
            raise ValueError(
                f"Downloaded size mismatch for {path}: {path.stat().st_size} != {item['size_bytes']}"
            )
    return local_files, selected, info.sha


def prepare_json_files(files: list[Path], staging_root: Path) -> list[Path]:
    """Decompress rare .zst JSONL files for environments without pyzstd."""
    prepared: list[Path] = []
    zstd_root = staging_root / "zstd_jsonl"
    for index, path in enumerate(files):
        if path.suffix != ".zst":
            prepared.append(path)
            continue
        zstd_root.mkdir(parents=True, exist_ok=True)
        target = zstd_root / f"{index:05d}-{path.name.removesuffix('.zst')}"
        if not target.is_file():
            temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
            temporary.unlink(missing_ok=True)
            with temporary.open("wb") as output:
                subprocess.run(
                    ["zstd", "-dc", str(path.resolve())], stdout=output, check=True
                )
            os.replace(temporary, target)
        prepared.append(target)
    return prepared


def serialize_natural_reasoning(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
    texts: list[str] = []
    for question, responses in zip(batch["question"], batch["responses"]):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Natural Reasoning row has an empty question")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ValueError(
                f"Natural Reasoning expected one response, found {responses!r}"
            )
        response = responses[0].get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Natural Reasoning row has an empty generated response")
        texts.append(f"{question.rstrip()}\n\n{response.lstrip()}")
    return {"text": texts}


def validate_text_dataset(dataset: Dataset) -> tuple[int, int]:
    if dataset.column_names != ["text"]:
        raise ValueError(f"Expected only text column, got {dataset.column_names}")
    documents = 0
    characters = 0
    for batch in dataset.with_format("arrow").iter(batch_size=8192):
        values = batch["text"]
        if values.null_count:
            raise ValueError("Dataset contains null text")
        empty = pc.sum(
            pc.equal(pc.utf8_length(pc.utf8_trim_whitespace(values)), 0)
        ).as_py()
        if empty:
            raise ValueError(f"Dataset contains {empty} empty text rows")
        documents += len(values)
        characters += int(pc.sum(pc.utf8_length(values)).as_py())
    return documents, characters


def load_projected_text_dataset(files: list[Path], staging_root: Path) -> Dataset:
    """Read only ``text`` so heterogeneous auxiliary columns cannot conflict."""
    arrow_path = staging_root / "projected_text.arrow"
    if not arrow_path.is_file():
        temporary = arrow_path.with_name(f".{arrow_path.name}.tmp.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        parse_options = paj.ParseOptions(
            explicit_schema=OUTPUT_SCHEMA,
            unexpected_field_behavior="ignore",
        )
        read_options = paj.ReadOptions(block_size=64 << 20, use_threads=True)
        try:
            with pa.OSFile(str(temporary), "wb") as sink:
                with pa.ipc.new_stream(sink, OUTPUT_SCHEMA) as writer:
                    for file_index, path in enumerate(files, start=1):
                        try:
                            with pa.input_stream(
                                str(path), compression="detect"
                            ) as source:
                                reader = paj.open_json(
                                    source,
                                    read_options=read_options,
                                    parse_options=parse_options,
                                )
                                for batch in reader:
                                    writer.write_batch(batch)
                        except Exception as error:
                            raise RuntimeError(
                                f"Failed to project text from {path}"
                            ) from error
                        if file_index % 100 == 0 or file_index == len(files):
                            print(
                                f"project_text files={file_index:,}/{len(files):,}",
                                flush=True,
                            )
            os.replace(temporary, arrow_path)
        finally:
            temporary.unlink(missing_ok=True)
    return Dataset.from_file(str(arrow_path))


def load_text_dataset(
    files: list[Path], args: argparse.Namespace, staging_root: Path
) -> tuple[Dataset, int, int]:
    if args.serializer == "text":
        dataset = load_projected_text_dataset(files, staging_root)
    elif args.serializer == "natural-reasoning":
        prepared = prepare_json_files(files, staging_root)
        cache_root = staging_root / "datasets_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        raw = load_dataset(
            "json",
            data_files=[str(path) for path in prepared],
            split="train",
            cache_dir=str(cache_root),
            num_proc=min(args.load_num_proc, len(prepared)),
        )
        if not isinstance(raw, Dataset):
            raise TypeError(f"Expected Dataset, got {type(raw)!r}")
        required = {"question", "responses"}
        if not required.issubset(raw.column_names):
            raise ValueError(
                f"Missing Natural Reasoning columns: {required - set(raw.column_names)}"
            )
        dataset = raw.map(
            serialize_natural_reasoning,
            batched=True,
            batch_size=1000,
            num_proc=min(args.load_num_proc, max(1, len(raw))),
            remove_columns=raw.column_names,
            desc="Serialize Natural Reasoning question/response",
        )
    else:
        raise ValueError(args.serializer)
    documents, characters = validate_text_dataset(dataset)
    return dataset, documents, characters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--include-pattern", action="append", required=True)
    parser.add_argument("--exclude-pattern", action="append", default=[])
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--serializer", choices=["text", "natural-reasoning"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--load-num-proc", type=int, default=16)
    parser.add_argument("--write-workers", type=int, default=16)
    parser.add_argument("--batch-rows", type=int, default=1000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    args = parser.parse_args()

    started = time.time()
    args.download_root = args.download_root.resolve()
    args.output_root = args.output_root.resolve()
    success = args.output_root / "_SUCCESS"
    if success.is_file():
        print(f"SKIP complete output: {args.output_root}", flush=True)
        return
    parquet_root = args.output_root / "parquet"
    staging_root = args.output_root / ".staging"
    parquet_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    download_started = time.time()
    files, source_records, resolved_revision = download_source_files(args)
    download_seconds = time.time() - download_started
    print(
        f"download_complete repo={args.repo_id}@{resolved_revision} files={len(files):,} "
        f"bytes={sum(path.stat().st_size for path in files):,} elapsed={download_seconds:.1f}s",
        flush=True,
    )

    load_started = time.time()
    dataset, documents, characters = load_text_dataset(files, args, staging_root)
    source_first_row = dataset[0]["text"]
    load_seconds = time.time() - load_started
    print(
        f"load_complete documents={documents:,} characters={characters:,} "
        f"elapsed={load_seconds:.1f}s",
        flush=True,
    )

    indices_path = staging_root / f"shuffle_indices_seed{args.seed}.arrow"
    shuffle_started = time.time()
    shuffled = dataset.shuffle(
        seed=args.seed,
        keep_in_memory=False,
        indices_cache_file_name=str(indices_path),
    )
    shuffled_first_row = shuffled[0]["text"]
    if documents > 1 and shuffled_first_row == source_first_row:
        raise RuntimeError("Shuffle unexpectedly retained the first source document")
    shuffle_seconds = time.time() - shuffle_started
    print(
        f"shuffle_complete indices_bytes={indices_path.stat().st_size:,} "
        f"elapsed={shuffle_seconds:.1f}s",
        flush=True,
    )

    global _SHUFFLED_DATASET
    _SHUFFLED_DATASET = shuffled
    tasks = []
    for shard_id in range(args.num_shards):
        target = parquet_root / f"train-{shard_id:05d}-of-{args.num_shards:05d}.parquet"
        tasks.append(
            (
                shard_id,
                args.num_shards,
                str(target),
                args.batch_rows,
                args.compression,
                args.compression_level,
            )
        )
    write_started = time.time()
    records = []
    with ProcessPoolExecutor(
        max_workers=min(args.write_workers, args.num_shards),
        mp_context=mp.get_context("fork"),
    ) as executor:
        futures = {executor.submit(write_one_shard, task): task[0] for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"shard_complete id={record['shard_id']} rows={record['rows']:,} "
                f"bytes={record['size_bytes']:,} done={len(records)}/{len(tasks)}",
                flush=True,
            )
    records.sort(key=lambda item: item["shard_id"])
    write_seconds = time.time() - write_started
    if sum(record["rows"] for record in records) != documents:
        raise ValueError("Physical output document count mismatch")
    for record in records:
        expected = expected_shard_rows(documents, args.num_shards, record["shard_id"])
        if record["rows"] != expected:
            raise ValueError(f"Shard {record['shard_id']} row count mismatch")

    first_output = (
        pq.read_table(records[0]["path"], columns=["text"]).column("text")[0].as_py()
    )
    output_first_row = first_output
    if output_first_row != shuffled_first_row:
        raise ValueError(
            "First physical document does not match shuffled document zero"
        )

    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "include_patterns": args.include_pattern,
        "exclude_patterns": args.exclude_pattern,
        "download_root": str(args.download_root),
        "source_file_count": len(files),
        "source_size_bytes": sum(path.stat().st_size for path in files),
        "source_files": source_records,
        "serializer": args.serializer,
        "serialization": (
            "question.rstrip() + '\\n\\n' + responses[0].response.lstrip()"
            if args.serializer == "natural-reasoning"
            else "text"
        ),
        "output_root": str(args.output_root),
        "schema": "text: string",
        "documents": documents,
        "characters": characters,
        "seed": args.seed,
        "shuffle": "datasets.Dataset.shuffle",
        "shuffle_indices_path": str(indices_path),
        "shuffle_indices_size_bytes": indices_path.stat().st_size,
        "num_shards": args.num_shards,
        "output_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "download_seconds": download_seconds,
        "load_seconds": load_seconds,
        "shuffle_seconds": shuffle_seconds,
        "write_seconds": write_seconds,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_root / "manifest.json", manifest)
    success.write_text(
        json.dumps(
            {
                "manifest": "manifest.json",
                "documents": documents,
                "files": args.num_shards,
                "seed": args.seed,
                "resolved_revision": resolved_revision,
            },
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"SUCCESS documents={documents:,} files={args.num_shards} "
        f"bytes={manifest['output_size_bytes']:,} elapsed={manifest['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
