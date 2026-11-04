#!/usr/bin/env python3
"""Download globally shuffled FineWeb-Edu 100BT and create a held-out split."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


EXP_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = EXP_DIR / "data" / "fineweb_edu_100bt_shuffled_raw"
DEFAULT_SPLIT_DIR = EXP_DIR / "data" / "fineweb_edu_100bt_shuffled_r8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download HuggingFaceFW/fineweb_edu_100BT-shuffled and create "
            "file-disjoint train/validation symlink directories."
        )
    )
    parser.add_argument("--repo-id", default="HuggingFaceFW/fineweb_edu_100BT-shuffled")
    parser.add_argument("--revision", default="be6b2a50d3a9c60d330c45384e80c7863cd3a25d")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--expected-shards", type=int, default=100)
    parser.add_argument("--val-shards", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parquet_entries(api: HfApi, repo_id: str, revision: str) -> list[object]:
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="data",
        recursive=True,
        revision=revision,
    )
    return sorted(
        (entry for entry in entries if getattr(entry, "path", "").endswith(".parquet")),
        key=lambda entry: getattr(entry, "path", ""),
    )


def replace_split_links(directory: Path, files: list[Path]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected_names = {path.name for path in files}

    for existing in directory.glob("*.parquet"):
        if existing.name in expected_names and existing.is_symlink():
            continue
        if not existing.is_symlink():
            raise RuntimeError(f"Refusing to replace non-symlink split file: {existing}")
        existing.unlink()

    for source in files:
        link = directory / source.name
        target = source.resolve()
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            if not link.is_symlink():
                raise RuntimeError(f"Refusing to replace non-symlink split file: {link}")
            link.unlink()
        link.symlink_to(target)


def main() -> None:
    args = parse_args()
    if args.val_shards <= 0:
        raise ValueError("--val-shards must be positive")
    if args.val_shards >= args.expected_shards:
        raise ValueError("--val-shards must be smaller than --expected-shards")

    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision)
    resolved_revision = info.sha
    entries = parquet_entries(api, args.repo_id, resolved_revision)
    if len(entries) != args.expected_shards:
        raise RuntimeError(
            f"Expected {args.expected_shards} parquet shards, found {len(entries)} "
            f"at revision {resolved_revision}"
        )

    total_size = sum((getattr(entry, "size", None) or 0) for entry in entries)
    print(f"Repository: {args.repo_id}@{resolved_revision}")
    print(f"Remote parquet shards: {len(entries)}")
    if total_size:
        print(f"Remote reported size: {total_size / (1024**3):.2f} GiB")
    print(f"Raw directory: {args.raw_dir}")
    print(f"Split directory: {args.split_dir}")
    print(f"Planned split: {len(entries) - args.val_shards} train / {args.val_shards} val")
    print("Validation shards:")
    for entry in entries[-args.val_shards :]:
        print(f"  {getattr(entry, 'path', entry)}")
    if args.dry_run:
        return

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        allow_patterns=["data/*.parquet", "README.md"],
        local_dir=args.raw_dir,
        max_workers=args.max_workers,
    )

    expected_paths = [getattr(entry, "path") for entry in entries]
    source_files = [args.raw_dir / relative_path for relative_path in expected_paths]
    missing = [path for path in source_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Download incomplete; {len(missing)} parquet shards are missing")
    if len({path.name for path in source_files}) != len(source_files):
        raise RuntimeError("Parquet basenames are not unique; flat symlink split would be ambiguous")

    train_files = source_files[: -args.val_shards]
    val_files = source_files[-args.val_shards :]
    train_names = {path.name for path in train_files}
    val_names = {path.name for path in val_files}
    if train_names & val_names:
        raise RuntimeError("Train/validation shard overlap detected")

    replace_split_links(args.split_dir / "train", train_files)
    replace_split_links(args.split_dir / "val", val_files)
    train_links = list((args.split_dir / "train").glob("*.parquet"))
    val_links = list((args.split_dir / "val").glob("*.parquet"))
    print(f"Prepared split: {len(train_links)} train / {len(val_links)} val")


if __name__ == "__main__":
    main()
