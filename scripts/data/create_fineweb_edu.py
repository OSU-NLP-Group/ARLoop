#!/usr/bin/env python3
"""Create the controlled FineWeb-Edu training data, or run one preparation stage."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.utils.pipeline import create_control, run

STAGES = {
    "download": "prepare_fineweb_edu_100bt_shuffled",
    "tokenize": "prepare_tokenized_100bt",
    "pack": "prepare_packed_100bt",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in STAGES:
        return run(STAGES[argv[0]], argv[1:])
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Stages: download, tokenize, pack. Use STAGE --help for stage options.",
    )
    parser.add_argument("command", choices=["all"], nargs="?", default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--tokenizer-path", type=Path, default=ROOT / "tokenizer/parcae_tokenizer"
    )
    parser.add_argument(
        "--revision", default="be6b2a50d3a9c60d330c45384e80c7863cd3a25d"
    )
    parser.add_argument("--num-lanes", type=int, default=64)
    args = parser.parse_args(argv)
    if args.num_lanes < 1:
        parser.error("--num-lanes must be positive")
    # Children run from ROOT; resolve caller-supplied relative paths first.
    args.data_root = args.data_root.resolve()
    args.tokenizer_path = args.tokenizer_path.resolve()
    create_control(args)


if __name__ == "__main__":
    main()
