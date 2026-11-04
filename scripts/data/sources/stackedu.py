"""Materialize Stack-Edu code and tokenize/pack one language in source order."""
from ..utils.workflow import parser, parse, prepared_source, lane_fetch, complete

LANGUAGES = ('python', 'java', 'javascript', 'c', 'cpp', 'csharp', 'php',
             'typescript', 'swift', 'sql', 'ruby', 'markdown', 'rust', 'go', 'shell')


def main(argv=None):
    p = parser(__doc__, subsets=LANGUAGES)
    p.add_argument('--num-lanes', type=int, default=8)
    a = parse(p, argv)
    if a.num_lanes < 1:
        p.error('--num-lanes must be positive')
    downloaded = a.data_root / 'stack_edu/raw_download'
    def fetch(raw):
        if not complete(downloaded):
            lane_fetch('download_stackedu_full', ['--output-root', downloaded,
                '--revision', a.revision, '--cache-root', a.data_root / '.cache/stackedu',
                '--num-lanes', a.num_lanes, '--metadata-num-proc', a.workers],
                a.num_lanes, a.workers, '--num-proc')
    return prepared_source(a, a.data_root / 'stack_edu' / a.subset, fetch,
        raw=downloaded / 'parquet' / a.subset, token_kind='stackedu',
        source_manifest=downloaded / 'manifest.json', language=a.subset,
        packed=a.data_root / 'stack_edu' / a.subset / 'smollm2_packed_seq4096_source_order')


if __name__ == "__main__":
    main()
