#!/usr/bin/env python3
"""Materialize all Stack-Edu Software Heritage blobs into resumable Parquet chunks."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import (
    Dataset,
    disable_progress_bars,
    get_dataset_config_names,
    load_dataset,
)


REPO_ID = "HuggingFaceTB/stack-edu"
BUCKET_NAME = "softwareheritage"
S3_CLIENT = None


from .common import write_json


def get_s3_client():
    global S3_CLIENT
    if S3_CLIENT is None:
        S3_CLIENT = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(
                signature_version=UNSIGNED,
                connect_timeout=10,
                read_timeout=120,
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=4,
            ),
        )
    return S3_CLIENT


def failed_download(error_code: str) -> dict[str, object]:
    return {
        "text": "",
        "download_success": False,
        "download_error": error_code,
        "compressed_bytes": 0,
        "decoded_bytes": 0,
    }


def download_contents(blob_id: str) -> dict[str, object]:
    """Follow the Stack-Edu dataset card and retrieve one gzipped SWH blob."""
    key = f"content/{blob_id}"
    for attempt in range(5):
        try:
            obj = get_s3_client().get_object(Bucket=BUCKET_NAME, Key=key)
            compressed_bytes = int(obj.get("ContentLength", 0))
            with gzip.GzipFile(fileobj=obj["Body"]) as fin:
                raw = fin.read()
            return {
                "text": raw.decode("utf-8", errors="ignore"),
                "download_success": True,
                "download_error": "",
                "compressed_bytes": compressed_bytes,
                "decoded_bytes": len(raw),
            }
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", "ClientError"))
            if code in {"NoSuchKey", "404"}:
                return failed_download(code)
            if attempt == 4:
                raise
        except Exception:
            if attempt == 4:
                raise
        time.sleep(min(2**attempt, 16))
    raise RuntimeError("unreachable")


def plan_path(output_root: Path) -> Path:
    return output_root / "download_plan.json"


def slug(config_name: str) -> str:
    return config_name.lower().replace("#", "sharp").replace("+", "p")


def output_path(output_root: Path, work: dict) -> Path:
    return (
        output_root
        / "parquet"
        / slug(work["config_name"])
        / f"part-{work['chunk_id']:05d}-of-{work['num_chunks']:05d}.parquet"
    )


def stats_path(output_root: Path, work: dict) -> Path:
    return (
        output_root
        / ".stats"
        / slug(work["config_name"])
        / f"part-{work['chunk_id']:05d}-of-{work['num_chunks']:05d}.json"
    )


def load_source(config_name: str, args: argparse.Namespace) -> Dataset:
    source = load_dataset(
        REPO_ID,
        config_name,
        split="train",
        revision=args.revision,
        cache_dir=str(args.cache_root / "datasets"),
        num_proc=min(args.metadata_num_proc, 16),
    )
    if not isinstance(source, Dataset):
        raise TypeError(f"Expected Dataset for {config_name}, got {type(source)!r}")
    if "blob_id" not in source.column_names:
        raise ValueError(f"{config_name}: missing blob_id column")
    return source


def build_plan(args: argparse.Namespace) -> dict:
    success = args.output_root / "_SUCCESS"
    if success.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {success}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    existing = plan_path(args.output_root)
    if existing.is_file():
        previous = json.loads(existing.read_text())
        expected = {
            "complete": True,
            "repo_id": REPO_ID,
            "revision": args.revision,
            "chunk_rows": args.chunk_rows,
            "num_lanes": args.num_lanes,
            "output_root": str(args.output_root.resolve()),
        }
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"Existing plan has {key}={previous.get(key)!r}, expected {value!r}"
                )
        print(f"SKIP valid existing plan: {existing}", flush=True)
        return previous
    configs = get_dataset_config_names(REPO_ID, revision=args.revision)
    if not configs:
        raise ValueError("Stack-Edu has no configs")

    work_items = []
    config_records = []
    for config_name in configs:
        source = load_source(config_name, args)
        rows = len(source)
        num_chunks = (rows + args.chunk_rows - 1) // args.chunk_rows
        config_records.append(
            {
                "config_name": config_name,
                "rows": rows,
                "num_chunks": num_chunks,
                "columns": source.column_names,
                "features": str(source.features),
            }
        )
        for chunk_id in range(num_chunks):
            start = chunk_id * args.chunk_rows
            stop = min(start + args.chunk_rows, rows)
            work_items.append(
                {
                    "work_id": len(work_items),
                    "config_name": config_name,
                    "chunk_id": chunk_id,
                    "num_chunks": num_chunks,
                    "start": start,
                    "stop": stop,
                    "rows": stop - start,
                }
            )
        del source
        print(
            f"planned config={config_name} rows={rows:,} chunks={num_chunks}",
            flush=True,
        )

    payload = {
        "version": 1,
        "complete": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "repo_id": REPO_ID,
        "revision": args.revision,
        "chunk_rows": args.chunk_rows,
        "num_lanes": args.num_lanes,
        "configs": config_records,
        "num_configs": len(config_records),
        "total_rows": sum(record["rows"] for record in config_records),
        "num_work_items": len(work_items),
        "work_items": work_items,
        "output_root": str(args.output_root.resolve()),
    }
    write_json(existing, payload)
    print(
        f"PLAN COMPLETE configs={payload['num_configs']} "
        f"rows={payload['total_rows']:,} work_items={payload['num_work_items']}",
        flush=True,
    )
    return payload


def load_plan(args: argparse.Namespace) -> dict:
    path = plan_path(args.output_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    plan = json.loads(path.read_text())
    expected = {
        "repo_id": REPO_ID,
        "revision": args.revision,
        "chunk_rows": args.chunk_rows,
        "num_lanes": args.num_lanes,
        "output_root": str(args.output_root.resolve()),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Plan has {key}={plan.get(key)!r}, expected {value!r}")
    return plan


def validate_completed(output: Path, sidecar: Path, work: dict) -> dict:
    if not output.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Incomplete work item: {output}")
    record = json.loads(sidecar.read_text())
    expected = {
        "complete": True,
        "work_id": work["work_id"],
        "config_name": work["config_name"],
        "chunk_id": work["chunk_id"],
        "start": work["start"],
        "stop": work["stop"],
        "rows": work["rows"],
        "output_path": str(output.resolve()),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"{sidecar}: {key}={record.get(key)!r}, expected {value!r}"
            )
    parquet_rows = pq.ParquetFile(output).metadata.num_rows
    if parquet_rows != work["rows"]:
        raise ValueError(f"{output}: {parquet_rows} rows, expected {work['rows']}")
    if output.stat().st_size != int(record["output_size_bytes"]):
        raise ValueError(f"Output size mismatch: {output}")
    return record


def summarize_downloaded(downloaded: Dataset) -> dict:
    successes = 0
    compressed_bytes = 0
    decoded_bytes = 0
    errors: dict[str, int] = defaultdict(int)
    columns = [
        "download_success",
        "download_error",
        "compressed_bytes",
        "decoded_bytes",
    ]
    for batch in downloaded.select_columns(columns).iter(batch_size=20_000):
        successes += sum(bool(value) for value in batch["download_success"])
        compressed_bytes += sum(int(value) for value in batch["compressed_bytes"])
        decoded_bytes += sum(int(value) for value in batch["decoded_bytes"])
        for success, error in zip(batch["download_success"], batch["download_error"]):
            if not success:
                errors[str(error)] += 1
    return {
        "successes": successes,
        "failures": len(downloaded) - successes,
        "compressed_bytes": compressed_bytes,
        "decoded_bytes": decoded_bytes,
        "failure_codes": dict(sorted(errors.items())),
    }


def convert_work(
    source: Dataset,
    work: dict,
    args: argparse.Namespace,
) -> dict:
    output = output_path(args.output_root, work)
    sidecar = stats_path(args.output_root, work)
    if output.is_file() and sidecar.is_file() and not args.force:
        record = validate_completed(output, sidecar, work)
        print(f"SKIP work={work['work_id']} output={output}", flush=True)
        return record
    if output.exists() or sidecar.exists():
        output.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    map_cache = args.cache_root / f"download-work-{work['work_id']:05d}.arrow"
    map_cache.unlink(missing_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getenv('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)

    started = time.perf_counter()
    sample = source.select(range(work["start"], work["stop"]))
    downloaded = sample.map(
        download_contents,
        input_columns="blob_id",
        num_proc=args.num_proc,
        writer_batch_size=1_000,
        load_from_cache_file=False,
        cache_file_name=str(map_cache),
        desc=f"Stack-Edu work {work['work_id']}",
    )
    if len(downloaded) != work["rows"]:
        raise ValueError(
            f"work {work['work_id']}: downloaded {len(downloaded)} rows, "
            f"expected {work['rows']}"
        )
    summary = summarize_downloaded(downloaded)
    pq.write_table(
        downloaded.data.table,
        temporary,
        row_group_size=1_000,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
    )
    if pq.ParquetFile(temporary).metadata.num_rows != work["rows"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Temporary output row mismatch: {temporary}")
    os.replace(temporary, output)
    elapsed = time.perf_counter() - started
    record = {
        "complete": True,
        **work,
        **summary,
        "repo_id": REPO_ID,
        "revision": args.revision,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "num_proc": args.num_proc,
        "output_path": str(output.resolve()),
        "output_size_bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
        "documents_per_second": work["rows"] / max(elapsed, 1e-9),
    }
    write_json(sidecar, record)
    validate_completed(output, sidecar, work)
    map_cache.unlink(missing_ok=True)
    print(
        f"DONE work={work['work_id']} config={work['config_name']} "
        f"chunk={work['chunk_id'] + 1}/{work['num_chunks']} rows={work['rows']:,} "
        f"successes={summary['successes']:,} failures={summary['failures']:,} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return record


def download_lane(args: argparse.Namespace) -> None:
    plan = load_plan(args)
    if args.lane_id < 0 or args.lane_id >= args.num_lanes:
        raise IndexError(f"lane {args.lane_id} outside [0, {args.num_lanes})")
    lane_work = [
        work
        for work in plan["work_items"]
        if int(work["work_id"]) % args.num_lanes == args.lane_id
    ]
    print(
        f"LANE START lane={args.lane_id}/{args.num_lanes} "
        f"work_items={len(lane_work)} rows={sum(w['rows'] for w in lane_work):,}",
        flush=True,
    )
    current_config = None
    source = None
    for position, work in enumerate(lane_work):
        if work["config_name"] != current_config:
            if source is not None:
                del source
            current_config = work["config_name"]
            source = load_source(current_config, args)
            config_plan = next(
                item
                for item in plan["configs"]
                if item["config_name"] == current_config
            )
            if len(source) != int(config_plan["rows"]):
                raise ValueError(f"{current_config}: source row count changed")
        assert source is not None
        print(
            f"lane={args.lane_id} item={position + 1}/{len(lane_work)} "
            f"work={work['work_id']} config={current_config}",
            flush=True,
        )
        convert_work(source, work, args)
    print(f"LANE COMPLETE lane={args.lane_id}", flush=True)


def finalize(args: argparse.Namespace) -> dict:
    plan = load_plan(args)
    records = []
    by_config: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "rows": 0,
            "successes": 0,
            "failures": 0,
            "compressed_bytes": 0,
            "decoded_bytes": 0,
            "output_size_bytes": 0,
            "chunks": 0,
        }
    )
    for position, work in enumerate(plan["work_items"]):
        record = validate_completed(
            output_path(args.output_root, work),
            stats_path(args.output_root, work),
            work,
        )
        records.append(record)
        aggregate = by_config[work["config_name"]]
        for key in (
            "rows",
            "successes",
            "failures",
            "compressed_bytes",
            "decoded_bytes",
            "output_size_bytes",
        ):
            aggregate[key] += int(record[key])
        aggregate["chunks"] += 1
        if (position + 1) % 50 == 0 or position + 1 == len(plan["work_items"]):
            print(
                f"verified {position + 1}/{len(plan['work_items'])} chunks", flush=True
            )

    totals = {
        key: sum(int(record[key]) for record in records)
        for key in (
            "rows",
            "successes",
            "failures",
            "compressed_bytes",
            "decoded_bytes",
            "output_size_bytes",
        )
    }
    if totals["rows"] != int(plan["total_rows"]):
        raise ValueError(
            f"Output rows {totals['rows']:,} != plan {plan['total_rows']:,}"
        )
    if totals["successes"] + totals["failures"] != totals["rows"]:
        raise ValueError("Success/failure accounting mismatch")
    manifest = {
        "complete": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "repo_id": REPO_ID,
        "revision": args.revision,
        "output_root": str(args.output_root.resolve()),
        "chunk_rows": args.chunk_rows,
        "num_lanes": args.num_lanes,
        "num_configs": plan["num_configs"],
        "num_chunks": len(records),
        **totals,
        "success_rate": totals["successes"] / max(totals["rows"], 1),
        "configs": dict(sorted(by_config.items())),
        "download_plan": str(plan_path(args.output_root).resolve()),
        "schema_note": "Original Stack-Edu metadata plus text/download status fields",
    }
    write_json(args.output_root / "manifest.json", manifest)
    write_json(
        args.output_root / "_SUCCESS",
        {
            "manifest": "manifest.json",
            "rows": totals["rows"],
            "successes": totals["successes"],
            "failures": totals["failures"],
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metadata-num-proc", type=int, default=16)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    lane = commands.add_parser("download-lane")
    lane.add_argument("--lane-id", type=int, required=True)
    lane.add_argument("--num-proc", type=int, default=48)
    lane.add_argument("--force", action="store_true")
    commands.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    disable_progress_bars()
    args = parse_args()
    if args.chunk_rows < 1 or args.num_lanes < 1:
        raise ValueError("chunk-rows and num-lanes must be positive")
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
