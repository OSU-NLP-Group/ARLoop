"""Shared source CLI, download adapters, tokenization and packing orchestration."""

import argparse
from pathlib import Path

from .pipeline import ROOT, convert_files, pack_source, run


def parser(description, *, subsets=(), revision=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dry-run', action='store_true', help='Validate options without preparing data')
    p.add_argument('--stage', choices=['all', 'fetch', 'tokenize', 'pack'], default='all',
                   help='fetch downloads and establishes the source document order')
    if subsets:
        p.add_argument('--subset', choices=subsets, default=subsets[0])
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--tokenizer-path', type=Path, default=ROOT / 'tokenizer/smollm2_tokenizer')
    p.add_argument('--revision', default=revision,
                   help='Pinned upstream commit; required for fetch when no recorded default exists')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-shards', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--target-output-tokens', type=int,
                   help='Default: pack all complete records in the tokenized source')
    return p


def parse(p, argv):
    a = p.parse_args(argv)
    for key in ('data_root', 'tokenizer_path'):
        setattr(a, key, getattr(a, key).resolve())
    if a.stage in ('all', 'tokenize') and not a.tokenizer_path.is_dir():
        p.error(f'Tokenizer directory does not exist: {a.tokenizer_path}')
    if a.workers < 1 or a.num_shards < 1:
        p.error('--workers and --num-shards must be positive')
    if a.target_output_tokens is not None and a.target_output_tokens < 4097:
        p.error('--target-output-tokens must be at least 4097')
    if a.stage in ('all', 'fetch') and not a.revision:
        p.error('--revision must specify the upstream dataset commit')
    return a


def complete(path):
    return (path / '_SUCCESS').is_file() and (path / 'manifest.json').is_file()


def hf_fetch(a, raw, repo, config='default', split='train'):
    run('shuffle_hf_config', [
        '--repo-id', repo, '--config-name', config, '--split', split,
        '--revision', a.revision, '--output-root', raw, '--seed', a.seed,
        '--num-shards', a.num_shards, '--download-num-proc', a.workers,
        '--write-workers', a.workers,
    ])


def text_fetch(a, raw, repo, patterns, serializer='text'):
    options = [
        '--repo-id', repo, '--revision', a.revision,
        '--download-root', raw.parent / 'raw_download', '--output-root', raw,
        '--serializer', serializer, '--seed', a.seed, '--num-shards', a.num_shards,
        '--download-workers', a.workers, '--load-num-proc', a.workers,
        '--write-workers', a.workers,
    ]
    for pattern in patterns:
        options += ['--include-pattern', pattern]
    run('download_shuffle_text_dataset', options)


def packed_path(a, source, subset=None):
    """Return the common packed-output location for a mixture source."""
    path = a.data_root / 'packed' / source
    if subset is not None:
        path /= subset.lower().replace('-', '_')
    return path / f'smollm2_seq4096_s{a.seed}'


def prepared_source(a, base, fetch, *, source, subset=None, serializer='text', raw=None,
                    token_kind='curated', source_manifest=None, language=None,
                    selection_options=()):
    """Execute one source workflow, reusing completed outputs on continuation."""
    raw = raw or base / f'raw_shuffled_s{a.seed}'
    tokens = base / f'smollm2_tokenized_eos_s{a.seed}'
    packed = packed_path(a, source, subset)
    if a.dry_run:
        print(f'{base}: {a.stage} -> {packed}', flush=True)
        return packed
    if a.stage in ('all', 'fetch') and not complete(raw):
        fetch(raw)
    if a.stage in ('all', 'tokenize') and not complete(tokens):
        options = ['--source-root', raw, '--output-root', tokens,
                   '--tokenizer-path', a.tokenizer_path]
        if token_kind == 'stackedu':
            options[0] = '--source-dir'
            convert_files('tokenize_stackedu_smollm2', [*options,
                          '--source-manifest', source_manifest, '--language', language])
        elif token_kind == 'fineweb':
            run('prepare_smol_tokenized_shuffled_prefix', [*options, 'plan',
                *selection_options])
            convert_files('prepare_smol_tokenized_shuffled_prefix', options, 'plan-count')
        else:
            convert_files('tokenize_curated_datasets_smollm2', [*options, '--serializer', serializer])
    if a.stage in ('all', 'pack') and not complete(packed):
        target = a.target_output_tokens
        if target is None:
            from .prepare_concat_packed_prefix import discover_sources, load_source_manifest
            load_source_manifest(tokens)
            target = sum((s['source_tokens'] // 4097) * 4097 for s in discover_sources(tokens))
            if not target:
                raise ValueError('Source has no complete 4097-token records')
        pack_source(['--source-root', tokens, '--output-root', packed,
                     '--target-output-tokens', target])
    return packed


def lane_fetch(module, options, lanes, workers, worker_flag):
    run(module, [*options, 'plan'])
    for lane in range(lanes):
        run(module, [*options, 'download-lane', '--lane-id', lane, worker_flag, workers])
    run(module, [*options, 'finalize'])
