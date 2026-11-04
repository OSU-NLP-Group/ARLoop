#!/usr/bin/env python3
"""Download the non-sample FineWeb-Edu 1.3T Parquet files reproducibly."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download


EXP_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = EXP_DIR / "data" / "fineweb_edu_1p3t_raw"
DEFAULT_REPO_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


from .common import write_json


def fetch_remote_manifest(repo_id: str, revision: str) -> dict:
    info = HfApi().dataset_info(
        repo_id,
        revision=revision,
        files_metadata=True,
    )
    files = []
    for sibling in info.siblings or []:
        relative_path = sibling.rfilename
        if not relative_path.startswith("data/") or not relative_path.endswith(
            ".parquet"
        ):
            continue
        parts = relative_path.split("/")
        if len(parts) != 3 or not parts[1].startswith("CC-MAIN-"):
            raise ValueError(f"Unexpected default-data path: {relative_path}")
        if sibling.size is None:
            raise ValueError(f"Missing file size for {relative_path}")
        files.append(
            {
                "path": relative_path,
                "size_bytes": int(sibling.size),
            }
        )
    files.sort(key=lambda record: record["path"])
    if not files:
        raise RuntimeError("No default FineWeb-Edu Parquet files discovered")
    dumps = sorted({record["path"].split("/")[1] for record in files})
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "selection": "data/CC-MAIN-*/*.parquet; sample/ excluded",
        "files": files,
        "file_count": len(files),
        "dump_count": len(dumps),
        "dumps": dumps,
        "total_size_bytes": sum(record["size_bytes"] for record in files),
    }


def verify_one(output_root: Path, record: dict) -> dict:
    path = output_root / record["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != record["size_bytes"]:
        raise ValueError(
            f"{path}: size {actual_size} != expected {record['size_bytes']}"
        )
    parquet_file = pq.ParquetFile(path)
    schema_names = parquet_file.schema_arrow.names
    if "text" not in schema_names or "id" not in schema_names:
        raise ValueError(f"{path}: unexpected schema columns {schema_names}")
    return {
        **record,
        "documents": parquet_file.metadata.num_rows,
        "row_groups": parquet_file.num_row_groups,
    }


def verify_download(
    output_root: Path,
    remote_manifest: dict,
    verify_workers: int,
) -> dict:
    started = time.time()
    with ThreadPoolExecutor(max_workers=verify_workers) as executor:
        verified = list(
            executor.map(
                lambda record: verify_one(output_root, record),
                remote_manifest["files"],
            )
        )
    local_paths = sorted((output_root / "data").glob("*/*.parquet"))
    expected_paths = {
        output_root / record["path"] for record in remote_manifest["files"]
    }
    unexpected = [str(path) for path in local_paths if path not in expected_paths]
    if unexpected:
        raise ValueError(
            f"Unexpected Parquet files under output root: {unexpected[:8]}"
        )
    return {
        "complete": True,
        "verified_at": datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_id": remote_manifest["repo_id"],
        "requested_revision": remote_manifest["requested_revision"],
        "resolved_revision": remote_manifest["resolved_revision"],
        "selection": remote_manifest["selection"],
        "output_root": str(output_root.resolve()),
        "file_count": len(verified),
        "dump_count": remote_manifest["dump_count"],
        "total_size_bytes": sum(record["size_bytes"] for record in verified),
        "documents": sum(record["documents"] for record in verified),
        "row_groups": sum(record["row_groups"] for record in verified),
        "verification": (
            "All expected files present with exact remote LFS sizes and readable "
            "Parquet footers; text and id columns present."
        ),
        "elapsed_seconds": time.time() - started,
        "files": verified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--verify-workers", type=int, default=16)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remote_manifest = fetch_remote_manifest(args.repo_id, args.revision)
    if remote_manifest["resolved_revision"] != args.revision:
        raise ValueError(
            "Pinned revision did not resolve to itself: "
            f"{args.revision} -> {remote_manifest['resolved_revision']}"
        )
    print(
        f"repo={args.repo_id}@{remote_manifest['resolved_revision']} "
        f"files={remote_manifest['file_count']:,} "
        f"dumps={remote_manifest['dump_count']:,} "
        f"bytes={remote_manifest['total_size_bytes']:,} "
        f"TiB={remote_manifest['total_size_bytes'] / 2**40:.3f}",
        flush=True,
    )
    if args.inspect_only:
        print(json.dumps(remote_manifest, indent=2, sort_keys=True))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    success_path = args.output_root / "_SUCCESS"
    if success_path.is_file() and not args.verify_only:
        print(f"SKIP completed download: {args.output_root}", flush=True)
        return

    if not args.verify_only:
        started = time.time()
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=remote_manifest["resolved_revision"],
            allow_patterns=["data/*/*.parquet", "README.md"],
            local_dir=args.output_root,
            max_workers=args.max_workers,
        )
        print(
            f"download_elapsed_seconds={time.time() - started:.1f}",
            flush=True,
        )

    local_manifest = verify_download(
        args.output_root,
        remote_manifest,
        args.verify_workers,
    )
    write_json(args.output_root / "manifest.json", local_manifest)
    success_path.write_text(
        f"FineWeb-Edu download complete revision={remote_manifest['resolved_revision']}\n"
    )
    print(
        f"VERIFIED files={local_manifest['file_count']:,} "
        f"documents={local_manifest['documents']:,} "
        f"bytes={local_manifest['total_size_bytes']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
