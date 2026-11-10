#!/usr/bin/env python3
"""Prepare each mixture source independently, from download through packing."""
import argparse
import importlib
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.sources import SOURCES


def expand_tasks(recipe, overrides):
    if not isinstance(recipe, dict) or not isinstance(recipe.get('sources'), list) or not recipe['sources']:
        raise ValueError('Recipe must contain a nonempty sources list')
    defaults = recipe.get('defaults', {})
    if not isinstance(defaults, dict):
        raise ValueError('Recipe defaults must be a mapping')
    tasks = []
    seen = set()
    for entry in recipe['sources']:
        if not isinstance(entry, dict) or entry.get('source') not in SOURCES:
            raise ValueError(f'Unknown source entry: {entry}')
        source = entry['source']
        subsets = entry.get('subsets', [None])
        if not isinstance(subsets, list) or not subsets:
            raise ValueError(f'{source}: subsets must be a nonempty list')
        options = {**defaults, **{k: v for k, v in entry.items() if k not in ('source', 'subsets')}, **overrides}
        for subset in subsets:
            key = (source, subset)
            if key in seen:
                raise ValueError(f'Duplicate preparation task: {key}')
            seen.add(key)
            args = []
            for name, value in options.items():
                if value is None:
                    continue
                flag = '--' + name.replace('_', '-')
                for item in value if isinstance(value, list) else [value]:
                    args.extend([flag, str(item)])
            if subset is not None:
                args.extend(['--subset', str(subset)])
            tasks.append((source, args))
    return tasks


def run_all(argv):
    p = argparse.ArgumentParser(description='Prepare the complete mixture and FineWeb validation data')
    p.add_argument('--recipe', type=Path, default=ROOT / 'configs/data/mix.yaml')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--tokenizer-path', type=Path)
    p.add_argument('--workers', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--num-shards', type=int)
    p.add_argument('--stage', choices=('all', 'fetch', 'tokenize', 'pack'), default='all')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    overrides = {k: v for k, v in vars(a).items() if k not in ('recipe', 'dry_run') and v is not None}
    recipe = yaml.safe_load(a.recipe.read_text())
    tasks = expand_tasks(recipe, overrides)
    modules = {source: importlib.import_module(f'scripts.data.sources.{source}') for source, _ in tasks}
    # Parse every source's real CLI before starting any downloads or conversion.
    print(f'Checking {len(tasks)} preparation tasks...', flush=True)
    for source, args in tasks:
        modules[source].main([*args, '--dry-run'])
    if a.dry_run:
        return
    for index, (source, args) in enumerate(tasks, 1):
        print(f'[{index}/{len(tasks)}] {source} {" ".join(args)}', flush=True)
        modules[source].main(args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'all':
        return run_all(argv[1:])
    if argv and argv[0] in SOURCES:
        return importlib.import_module(f"scripts.data.sources.{argv[0]}").main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('all', *SOURCES), help='Use SOURCE --help for workflow options')
    parser.parse_args(argv)


if __name__ == '__main__':
    main()
