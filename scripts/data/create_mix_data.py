#!/usr/bin/env python3
"""Prepare each mixture source independently, from download through packing."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.sources import SOURCES


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SOURCES:
        return importlib.import_module(f"scripts.data.sources.{argv[0]}").main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=SOURCES, help='Use SOURCE --help for workflow options')
    parser.parse_args(argv)


if __name__ == '__main__':
    main()
