"""Download FineWeb-Edu, globally shuffle, tokenize a prefix and concatenate-pack."""
from ..utils.workflow import parser, parse, prepared_source, run, complete


def main(argv=None):
    p = parser(__doc__, revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9')
    p.add_argument('--target-source-tokens', type=int, default=700_000_000_000)
    p.add_argument('--num-lanes', type=int, default=40)
    p.add_argument('--num-buckets', type=int, default=1024)
    a = parse(p, argv)
    if a.num_lanes < 1 or a.num_buckets < 2 or a.num_buckets & (a.num_buckets - 1):
        p.error('positive lanes and a power-of-two bucket count >= 2 are required')
    if a.target_source_tokens < 1:
        p.error('--target-source-tokens must be positive')
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
    label = str(a.target_source_tokens // 1_000_000_000) + 'bt' if a.target_source_tokens % 1_000_000_000 == 0 else str(a.target_source_tokens) + 't'
    return prepared_source(a, a.data_root / ('fineweb_edu_' + label), fetch,
        raw=a.data_root / f'fineweb_edu_1p3t_shuffled_s{a.seed}', token_kind='fineweb',
        target_source_tokens=a.target_source_tokens,
        packed=a.data_root / f'fineweb_edu_{label}_smollm2_packed_seq4096_s{a.seed}')


if __name__ == "__main__":
    main()
