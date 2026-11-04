#!/usr/bin/env python3
"""Rehydrate, script-convert, and physically shuffle The Stack v2 notebooks.

The Hugging Face rows contain Software Heritage blob identifiers and
provenance, not notebook content.  A global Dataset.shuffle permutation is
applied to those rows first.  Contiguous chunks of that permutation are then
rehydrated from S3 and converted with nbformat+jupytext, so the resulting
Parquet file order is already a deterministic global document shuffle.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


REPO_ID = "bigcode/the-stack-v2"
DATA_DIR = "data/Jupyter_Notebook"
BUCKET = "softwareheritage"
PROVENANCE_COLUMNS = (
    "blob_id",
    "content_id",
    "repo_name",
    "path",
    "detected_licenses",
    "license_type",
    "snapshot_id",
    "revision_id",
)
RAW_SCHEMA = pa.schema(
    [
        pa.field("text", pa.large_string()),
        pa.field("blob_id", pa.string()),
        pa.field("content_id", pa.string()),
        pa.field("repo_name", pa.string()),
        pa.field("path", pa.string()),
        pa.field("detected_licenses", pa.list_(pa.string())),
        pa.field("license_type", pa.string()),
        pa.field("snapshot_id", pa.string()),
        pa.field("revision_id", pa.string()),
        pa.field("script_extension", pa.string()),
    ]
)
PROVENANCE_SCHEMA = pa.schema(
    [RAW_SCHEMA.field(name) for name in PROVENANCE_COLUMNS]
    + [RAW_SCHEMA.field("script_extension")]
)


from .common import write_json


def hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    path = Path.home() / ".cache" / "huggingface" / "token"
    if path.is_file():
        token = path.read_text().strip()
        return token or None
    return None


def add_jupyter_dependencies(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing isolated jupyter dependencies: {path}")
    sys.path.insert(0, str(path))


def load_source(args: argparse.Namespace) -> Dataset:
    dataset = load_dataset(
        REPO_ID,
        data_dir=DATA_DIR,
        split="train",
        revision=args.revision,
        token=hf_token(),
        cache_dir=str(args.cache_root / "datasets"),
        num_proc=min(args.metadata_num_proc, 16),
        keep_in_memory=False,
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)!r}")
    missing = set(PROVENANCE_COLUMNS) - set(dataset.column_names)
    if missing:
        raise ValueError(f"Stack metadata is missing columns: {sorted(missing)}")
    return dataset


def shuffled_source(args: argparse.Namespace) -> Dataset:
    source = load_source(args)
    indices = args.output_root / ".staging" / f"shuffle_indices_seed{args.seed}.arrow"
    if not indices.is_file():
        raise FileNotFoundError(f"Run plan first; missing {indices}")
    return source.shuffle(
        seed=args.seed,
        keep_in_memory=False,
        indices_cache_file_name=str(indices),
    ).select_columns(list(PROVENANCE_COLUMNS))


def raw_path(args: argparse.Namespace, chunk_id: int, num_chunks: int) -> Path:
    return (
        args.output_root
        / "parquet"
        / f"train-{chunk_id:05d}-of-{num_chunks:05d}.parquet"
    )


def provenance_path(args: argparse.Namespace, chunk_id: int, num_chunks: int) -> Path:
    return (
        args.provenance_root
        / "parquet"
        / f"part-{chunk_id:05d}-of-{num_chunks:05d}.parquet"
    )


def stats_path(args: argparse.Namespace, chunk_id: int, num_chunks: int) -> Path:
    return args.output_root / ".stats" / f"part-{chunk_id:05d}-of-{num_chunks:05d}.json"


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if (args.output_root / "_SUCCESS").exists():
        raise FileExistsError(f"Refusing to overwrite completed {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.provenance_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / ".staging").mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=hf_token()).dataset_info(REPO_ID, revision=args.revision)
    if info.sha != args.revision:
        raise ValueError(f"Revision moved: {args.revision} -> {info.sha}")
    source = load_source(args)
    source_rows = len(source)
    total_rows = min(source_rows, args.max_rows) if args.max_rows else source_rows
    length_bytes = (
        source["length_bytes"] if "length_bytes" in source.column_names else []
    )
    total_declared_bytes = sum(int(value or 0) for value in length_bytes)
    indices = args.output_root / ".staging" / f"shuffle_indices_seed{args.seed}.arrow"
    started = time.time()
    source.shuffle(
        seed=args.seed,
        keep_in_memory=False,
        indices_cache_file_name=str(indices),
    )
    num_chunks = (total_rows + args.chunk_rows - 1) // args.chunk_rows
    chunks = []
    for chunk_id in range(num_chunks):
        start = chunk_id * args.chunk_rows
        stop = min(start + args.chunk_rows, total_rows)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "lane_id": chunk_id % args.num_lanes,
                "start": start,
                "stop": stop,
                "rows": stop - start,
                "raw_path": str(raw_path(args, chunk_id, num_chunks).resolve()),
                "provenance_path": str(
                    provenance_path(args, chunk_id, num_chunks).resolve()
                ),
            }
        )
    payload = {
        "complete": True,
        "version": 1,
        "repo_id": REPO_ID,
        "data_dir": DATA_DIR,
        "revision": info.sha,
        "seed": args.seed,
        "shuffle": "datasets.Dataset.shuffle before blob retrieval",
        "source_rows": source_rows,
        "selected_rows": total_rows,
        "source_declared_length_bytes": total_declared_bytes,
        "chunk_rows": args.chunk_rows,
        "num_chunks": num_chunks,
        "num_lanes": args.num_lanes,
        "shuffle_indices_path": str(indices.resolve()),
        "shuffle_indices_size_bytes": indices.stat().st_size,
        "output_root": str(args.output_root.resolve()),
        "provenance_root": str(args.provenance_root.resolve()),
        "elapsed_seconds": time.time() - started,
        "chunks": chunks,
    }
    plan = args.output_root / "download_plan.json"
    if plan.is_file():
        previous = json.loads(plan.read_text())
        stable = ("revision", "seed", "selected_rows", "chunk_rows", "num_lanes")
        for key in stable:
            if previous.get(key) != payload.get(key):
                raise ValueError(f"Existing plan differs at {key}")
        print(f"SKIP existing plan: {plan}", flush=True)
        return previous
    write_json(plan, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "chunks"}, indent=2
        ),
        flush=True,
    )
    return payload


def load_plan(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_root / "download_plan.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    plan = json.loads(path.read_text())
    expected = {
        "complete": True,
        "revision": args.revision,
        "seed": args.seed,
        "chunk_rows": args.chunk_rows,
        "num_lanes": args.num_lanes,
        "output_root": str(args.output_root.resolve()),
        "provenance_root": str(args.provenance_root.resolve()),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Plan {key}={plan.get(key)!r}, expected {value!r}")
    return plan


def s3_client(args: argparse.Namespace):
    kwargs: dict[str, Any] = {}
    if args.allow_unsigned:
        kwargs["signature_version"] = UNSIGNED
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(
            **kwargs,
            connect_timeout=10,
            read_timeout=180,
            retries={"max_attempts": 10, "mode": "adaptive"},
            max_pool_connections=max(args.download_workers, 4),
        ),
    )


def notebook_extension(notebook: Any) -> str:
    metadata = notebook.get("metadata", {})
    language_info = metadata.get("language_info", {}) or {}
    extension = language_info.get("file_extension")
    if (
        isinstance(extension, str)
        and extension.startswith(".")
        and len(extension) <= 16
    ):
        return extension.lower()
    names = [
        language_info.get("name", ""),
        (metadata.get("kernelspec", {}) or {}).get("language", ""),
        (metadata.get("kernelspec", {}) or {}).get("name", ""),
    ]
    normalized = " ".join(str(value).lower() for value in names)
    mapping = {
        "python": ".py",
        "julia": ".jl",
        "ir": ".r",
        "r ": ".r",
        "scala": ".scala",
        "javascript": ".js",
        "typescript": ".ts",
        "bash": ".sh",
        "shell": ".sh",
        "matlab": ".m",
        "wolfram": ".wl",
        "csharp": ".cs",
        "fsharp": ".fs",
        "java": ".java",
        "ruby": ".rb",
        "rust": ".rs",
        "cpp": ".cpp",
        "c++": ".cpp",
    }
    for needle, result in mapping.items():
        if needle in normalized:
            return result
    if "py" in normalized:
        return ".py"
    raise ValueError("unknown_notebook_language")


def convert_blob(client: Any, row: dict[str, Any]) -> dict[str, Any]:
    import jupytext
    import nbformat

    blob_id = str(row["blob_id"])
    try:
        obj = client.get_object(Bucket=BUCKET, Key=f"content/{blob_id}")
        with gzip.GzipFile(fileobj=obj["Body"]) as handle:
            raw = handle.read()
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", "ClientError"))
        return {"ok": False, "stage": "download", "error": code}
    except Exception as error:
        return {"ok": False, "stage": "download", "error": type(error).__name__}
    try:
        notebook = nbformat.from_dict(json.loads(raw.decode("utf-8", errors="strict")))
        for cell in notebook.cells:
            if isinstance(cell.source, list):
                cell.source = "".join(cell.source)
        extension = notebook_extension(notebook)
        text = jupytext.writes(notebook, fmt=extension.lstrip("."))
        # Some notebooks contain unpaired UTF-16 surrogate code points in
        # cell strings.  They are not valid UTF-8 and PyArrow cannot write
        # them to a string column, so replace only those invalid code points.
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        if not text.strip():
            raise ValueError("empty_script")
    except Exception as error:
        return {"ok": False, "stage": "convert", "error": type(error).__name__}
    output = {name: row.get(name) for name in PROVENANCE_COLUMNS}
    output.update({"text": text, "script_extension": extension})
    return {
        "ok": True,
        "row": output,
        "decoded_bytes": len(raw),
        "text_bytes": len(text.encode("utf-8")),
    }


def validate_existing(
    raw: Path, provenance: Path, stats: Path, expected_rows: int
) -> bool:
    if not raw.is_file() or not provenance.is_file() or not stats.is_file():
        return False
    record = json.loads(stats.read_text())
    if not record.get("complete") or int(record.get("attempted", -1)) != expected_rows:
        return False
    raw_file = pq.ParquetFile(raw)
    provenance_file = pq.ParquetFile(provenance)
    return (
        raw_file.metadata.num_rows == int(record["documents"])
        and provenance_file.metadata.num_rows == int(record["documents"])
        and raw_file.schema_arrow.equals(RAW_SCHEMA)
        and provenance_file.schema_arrow.equals(PROVENANCE_SCHEMA)
    )


def process_chunk(
    args: argparse.Namespace,
    source: Dataset,
    client: Any,
    chunk: dict[str, Any],
    num_chunks: int,
) -> dict[str, Any]:
    chunk_id = int(chunk["chunk_id"])
    raw = raw_path(args, chunk_id, num_chunks)
    provenance = provenance_path(args, chunk_id, num_chunks)
    stats = stats_path(args, chunk_id, num_chunks)
    if validate_existing(raw, provenance, stats, int(chunk["rows"])):
        print(f"SKIP valid chunk={chunk_id}/{num_chunks}", flush=True)
        return json.loads(stats.read_text())
    raw.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    stats.parent.mkdir(parents=True, exist_ok=True)
    raw.unlink(missing_ok=True)
    provenance.unlink(missing_ok=True)
    stats.unlink(missing_ok=True)
    raw_tmp = raw.with_name(f".{raw.name}.tmp.{os.getpid()}")
    provenance_tmp = provenance.with_name(f".{provenance.name}.tmp.{os.getpid()}")
    rows = source[int(chunk["start"]) : int(chunk["stop"])]
    input_rows = [
        dict(zip(rows, values)) for values in zip(*(rows[name] for name in rows))
    ]
    started = time.time()
    failures: dict[str, int] = {}
    documents = decoded_bytes = text_bytes = 0
    with (
        pq.ParquetWriter(
            raw_tmp, RAW_SCHEMA, compression="zstd", compression_level=3
        ) as raw_writer,
        pq.ParquetWriter(
            provenance_tmp, PROVENANCE_SCHEMA, compression="zstd", compression_level=3
        ) as provenance_writer,
        ThreadPoolExecutor(max_workers=args.download_workers) as executor,
    ):
        for start in range(0, len(input_rows), args.write_batch_rows):
            batch = input_rows[start : start + args.write_batch_rows]
            results = list(executor.map(lambda row: convert_blob(client, row), batch))
            successful = []
            for result in results:
                if result["ok"]:
                    successful.append(result["row"])
                    documents += 1
                    decoded_bytes += int(result["decoded_bytes"])
                    text_bytes += int(result["text_bytes"])
                else:
                    key = f"{result['stage']}:{result['error']}"
                    failures[key] = failures.get(key, 0) + 1
            if successful:
                table = pa.Table.from_pylist(successful, schema=RAW_SCHEMA)
                raw_writer.write_table(table, row_group_size=len(successful))
                provenance_writer.write_table(
                    table.select(list(PROVENANCE_SCHEMA.names)),
                    row_group_size=len(successful),
                )
            print(
                f"chunk={chunk_id + 1}/{num_chunks} attempted={min(start + len(batch), len(input_rows)):,}/{len(input_rows):,} "
                f"documents={documents:,} failures={sum(failures.values()):,}",
                flush=True,
            )
    os.replace(raw_tmp, raw)
    os.replace(provenance_tmp, provenance)
    record = {
        "complete": True,
        "chunk_id": chunk_id,
        "num_chunks": num_chunks,
        "lane_id": int(chunk["lane_id"]),
        "start": int(chunk["start"]),
        "stop": int(chunk["stop"]),
        "attempted": len(input_rows),
        "documents": documents,
        "failures": failures,
        "decoded_bytes": decoded_bytes,
        "text_bytes": text_bytes,
        "raw_path": str(raw.resolve()),
        "raw_size_bytes": raw.stat().st_size,
        "provenance_path": str(provenance.resolve()),
        "provenance_size_bytes": provenance.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    write_json(stats, record)
    return record


def download_lane(args: argparse.Namespace) -> None:
    if args.jupyter_deps is not None:
        add_jupyter_dependencies(args.jupyter_deps)
    plan = load_plan(args)
    if args.lane_id < 0 or args.lane_id >= args.num_lanes:
        raise IndexError(f"lane {args.lane_id} outside [0, {args.num_lanes})")
    source = shuffled_source(args)
    client = s3_client(args)
    chunks = [
        chunk for chunk in plan["chunks"] if int(chunk["lane_id"]) == args.lane_id
    ]
    for done, chunk in enumerate(chunks, 1):
        process_chunk(args, source, client, chunk, int(plan["num_chunks"]))
        print(f"lane={args.lane_id} chunks={done}/{len(chunks)}", flush=True)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_plan(args)
    records = []
    for chunk in plan["chunks"]:
        raw = Path(chunk["raw_path"])
        provenance = Path(chunk["provenance_path"])
        stats = stats_path(args, int(chunk["chunk_id"]), int(plan["num_chunks"]))
        if not validate_existing(raw, provenance, stats, int(chunk["rows"])):
            raise FileNotFoundError(f"Missing or invalid chunk {chunk['chunk_id']}")
        records.append(json.loads(stats.read_text()))
    attempted = sum(int(record["attempted"]) for record in records)
    documents = sum(int(record["documents"]) for record in records)
    failures: dict[str, int] = {}
    for record in records:
        for key, value in record["failures"].items():
            failures[key] = failures.get(key, 0) + int(value)
    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_id": REPO_ID,
        "data_dir": DATA_DIR,
        "revision": args.revision,
        "seed": args.seed,
        "shuffle": "global Dataset.shuffle on metadata before SWH retrieval",
        "conversion": "nbformat.from_dict + jupytext script conversion",
        "source_rows": int(plan["source_rows"]),
        "attempted_rows": attempted,
        "documents": documents,
        "failed_rows": attempted - documents,
        "failures": failures,
        "num_shards": int(plan["num_chunks"]),
        "columns": RAW_SCHEMA.names,
        "features": str(RAW_SCHEMA),
        "output_size_bytes": sum(int(record["raw_size_bytes"]) for record in records),
        "decoded_bytes": sum(int(record["decoded_bytes"]) for record in records),
        "text_bytes": sum(int(record["text_bytes"]) for record in records),
        "provenance_root": str(args.provenance_root.resolve()),
        "provenance_size_bytes": sum(
            int(record["provenance_size_bytes"]) for record in records
        ),
        "files": records,
    }
    if documents == 0:
        raise ValueError("No notebooks were successfully converted")
    write_json(args.output_root / "manifest.json", manifest)
    write_json(
        args.provenance_root / "manifest.json",
        {
            "complete": True,
            "repo_id": REPO_ID,
            "revision": args.revision,
            "documents": documents,
            "seed": args.seed,
            "columns": PROVENANCE_SCHEMA.names,
            "source_manifest": str((args.output_root / "manifest.json").resolve()),
        },
    )
    (args.output_root / "_SUCCESS").write_text(
        f"documents={documents} seed={args.seed}\n"
    )
    (args.provenance_root / "_SUCCESS").write_text(
        f"documents={documents} seed={args.seed}\n"
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "files"}, indent=2
        ),
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--jupyter-deps", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=10000)
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--metadata-num-proc", type=int, default=16)
    parser.add_argument("--max-rows", type=int)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    download = subparsers.add_parser("download-lane")
    download.add_argument("--lane-id", type=int, required=True)
    download.add_argument("--download-workers", type=int, default=48)
    download.add_argument("--write-batch-rows", type=int, default=128)
    download.add_argument("--allow-unsigned", action="store_true")
    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_root = args.output_root.resolve()
    args.provenance_root = args.provenance_root.resolve()
    args.cache_root = args.cache_root.resolve()
    if args.jupyter_deps is not None:
        args.jupyter_deps = args.jupyter_deps.resolve()
    if args.command == "plan":
        build_plan(args)
    elif args.command == "download-lane":
        download_lane(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
