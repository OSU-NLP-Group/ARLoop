"""Run preparation stages with the current Python environment."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def run(module: str, args: list[str], *, capture: bool = False):
    result = subprocess.run(
        [sys.executable, "-m", f"scripts.data.utils.{module}", *map(str, args)],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def convert_files(module: str, args: list[str], count_command: str = "count"):
    count = int(run(module, [*args, count_command], capture=True))
    for index in range(count):
        run(module, [*args, "convert-file", "--file-index", str(index)])
    run(module, [*args, "finalize"])


def pack_source(args: list[str]):
    module = "prepare_concat_packed_prefix"
    run(module, [*args, "plan"])
    count = int(run(module, [*args, "plan-count"], capture=True))
    for lane in range(count):
        run(module, [*args, "pack-lane", "--lane-id", str(lane)])
    run(module, [*args, "finalize", "--write-success"])


def create_control(args):
    raw = args.data_root / "fineweb_edu_100bt_shuffled_raw"
    split = args.data_root / "fineweb_edu_100bt_shuffled_r8"
    tokens = args.data_root / "100bt_tokenized"
    packed = args.data_root / "100bt_packed_2049"
    run(
        "prepare_fineweb_edu_100bt_shuffled",
        [
            "--raw-dir",
            raw,
            "--split-dir",
            split,
            "--revision",
            args.revision,
        ],
    )
    tokenize_args = [
        "--source-root",
        split,
        "--output-root",
        tokens,
        "--tokenizer-path",
        args.tokenizer_path,
    ]
    files = json.loads(
        run("prepare_tokenized_100bt", [*tokenize_args, "list", "--json"], capture=True)
    )
    for index in range(len(files)):
        run(
            "prepare_tokenized_100bt",
            [*tokenize_args, "convert-file", "--file-index", str(index)],
        )
    run("prepare_tokenized_100bt", [*tokenize_args, "verify", "--write-success"])
    pack_args = [
        "--source-root",
        tokens,
        "--output-root",
        packed,
        "--tokenizer-path",
        args.tokenizer_path,
        "--num-lanes",
        str(args.num_lanes),
    ]
    for lane in range(args.num_lanes):
        run(
            "prepare_packed_100bt", [*pack_args, "convert-lane", "--lane-id", str(lane)]
        )
    run("prepare_packed_100bt", [*pack_args, "verify", "--write-success"])
