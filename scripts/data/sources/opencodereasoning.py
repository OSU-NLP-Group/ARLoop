"""Download, shuffle and restore OpenCodeReasoning questions before packing."""
from ..utils.workflow import parser, parse, hf_fetch, prepared_source, run, complete


def main(argv=None):
    p = parser(__doc__, subsets=('split_0', 'split_1'))
    p.add_argument('--apps-revision')
    p.add_argument('--taco-revision')
    a = parse(p, argv)
    if a.subset == 'split_1' and a.stage in ('all', 'fetch'):
        if not a.apps_revision or not a.taco_revision:
            p.error('split_1 fetch requires --apps-revision and --taco-revision')
    base = a.data_root / 'opencodereasoning' / a.subset
    def fetch(raw):
        shuffled = base / f'raw_shuffled_s{a.seed}'
        if not complete(shuffled):
            hf_fetch(a, shuffled, 'nvidia/OpenCodeReasoning', config=a.subset, split=a.subset)
        if a.subset == 'split_1':
            run('rehydrate_opencodereasoning_split1', ['--source-root', shuffled,
                '--output-root', raw, '--cache-dir', a.data_root / '.cache/rehydrate',
                '--apps-revision', a.apps_revision, '--taco-revision', a.taco_revision])
    raw = base / (f'raw_rehydrated_s{a.seed}' if a.subset == 'split_1' else f'raw_shuffled_s{a.seed}')
    return prepared_source(a, base, fetch, raw=raw, serializer='input-output')


if __name__ == "__main__":
    main()
