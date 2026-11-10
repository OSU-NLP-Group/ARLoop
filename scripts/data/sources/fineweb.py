"""Download FineWeb-Edu, globally shuffle, reserve validation, tokenize all remaining documents and concatenate-pack."""
from ..utils.workflow import parser, parse, prepared_source, run, complete, packed_path
from ..utils.pipeline import convert_files, pack_source


def main(argv=None):
    p = parser(__doc__, revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9')
    p.add_argument('--num-lanes', type=int, default=40)
    p.add_argument('--num-buckets', type=int, default=1024)
    p.add_argument('--validation-tokens', type=int, default=10_000_000, help='Packed validation token budget')
    p.add_argument('--validation-source-tokens', type=int, default=20_000_000)
    a = parse(p, argv)
    if a.validation_tokens < 4097:
        p.error('--validation-tokens must be at least 4097')
    if a.validation_source_tokens < 1:
        p.error('--validation-source-tokens must be positive')
    if a.num_lanes < 1 or a.num_buckets < 2 or a.num_buckets & (a.num_buckets - 1):
        p.error('positive lanes and a power-of-two bucket count >= 2 are required')
    def fetch(raw):
        downloaded = a.data_root / 'fineweb_edu_1p3t_raw'
        if not complete(downloaded):
            run('download_fineweb_edu_1p3t', ['--output-root', downloaded, '--revision', a.revision])
        options = ['--source-root', downloaded, '--output-root', raw,
                   '--seed', a.seed, '--num-lanes', a.num_lanes, '--num-buckets', a.num_buckets]
        for lane in range(a.num_lanes):
            run('prepare_hash_shuffled_raw_fineweb_lanes', [*options, 'partition-lane', '--lane-id', lane])
        for bucket in range(a.num_buckets):
            run('prepare_hash_shuffled_raw_fineweb_lanes', [*options, 'sort-bucket', '--bucket-id', bucket])
        run('prepare_hash_shuffled_raw_fineweb_lanes', [*options, 'finalize'])
    label = '1p3t'
    raw = a.data_root / f'fineweb_edu_1p3t_shuffled_s{a.seed}'
    val_label = f'{a.validation_tokens // 1_000_000}m' if a.validation_tokens % 1_000_000 == 0 else str(a.validation_tokens)
    val_tokens = a.data_root / f'fineweb_edu_1p3t_heldout_{val_label}' / f'smollm2_tokenized_eos_s{a.seed}'
    if not a.dry_run:
        if a.stage in ('all', 'fetch') and not complete(raw):
            fetch(raw)
        if a.stage != 'fetch':
            prepare_validation(a, label)
    packed = prepared_source(a, a.data_root / 'fineweb_edu_1p3t', fetch,
        raw=raw, token_kind='fineweb',
        selection_options=['--exclude-plan', val_tokens / 'selection_plan.json'], source='fineweb')
    if a.dry_run:
        print(f'  validation: {a.validation_tokens} packed tokens reserved before training selection')
    return packed


def prepare_validation(a, label):
    val_label = f'{a.validation_tokens // 1_000_000}m' if a.validation_tokens % 1_000_000 == 0 else str(a.validation_tokens)
    base = a.data_root / f'fineweb_edu_{label}_heldout_{val_label}'
    tokens = base / f'smollm2_tokenized_eos_s{a.seed}'
    output = packed_path(a, 'fineweb', f'heldout_{val_label}')
    if a.stage in ('all', 'tokenize') and not complete(tokens):
        options = ['--source-root', a.data_root / f'fineweb_edu_1p3t_shuffled_s{a.seed}',
                   '--output-root', tokens, '--tokenizer-path', a.tokenizer_path]
        run('prepare_smol_tokenized_shuffled_prefix', [*options, 'plan',
            '--target-source-tokens', a.validation_source_tokens])
        convert_files('prepare_smol_tokenized_shuffled_prefix', options, 'plan-count')
    if a.stage in ('all', 'pack') and not complete(output):
        pack_source(['--source-root', tokens, '--output-root', output, '--split', 'val',
                     '--target-output-tokens', a.validation_tokens])


if __name__ == "__main__":
    main()
