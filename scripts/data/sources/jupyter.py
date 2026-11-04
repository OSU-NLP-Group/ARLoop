"""Rehydrate, script-convert and shuffle Stack v2 notebooks, then tokenize/pack."""
from ..utils.workflow import parser, parse, prepared_source, lane_fetch


def main(argv=None):
    p = parser(__doc__, revision='e565caa3a78c2423bd374333a472b049eb090e47')
    p.add_argument('--num-lanes', type=int, default=8)
    a = parse(p, argv)
    if a.num_lanes < 1:
        p.error('--num-lanes must be positive')
    base = a.data_root / 'stack_v2_jupyter_20bt'
    def fetch(raw):
        lane_fetch('prepare_stackv2_jupyter', ['--output-root', raw,
            '--provenance-root', base / 'provenance', '--revision', a.revision,
            '--cache-root', a.data_root / '.cache/jupyter', '--seed', a.seed,
            '--num-lanes', a.num_lanes, '--metadata-num-proc', a.workers],
            a.num_lanes, a.workers, '--download-workers')
    return prepared_source(a, base, fetch)


if __name__ == "__main__":
    main()
