"""Download StackExchange sites, shuffle threads, tokenize and pack."""
from ..utils.workflow import parser, parse, prepared_source, run


def main(argv=None):
    a = parse(parser(__doc__, revision='fe382c48856f4eae5d8402cd4aa9be55ee1e1ee0'), argv)
    def fetch(raw):
        run('shuffle_stackexchange_2025', ['--revision', a.revision,
            '--output-root', raw, '--seed', a.seed, '--num-shards', a.num_shards,
            '--download-workers', a.workers, '--load-num-proc', a.workers,
            '--write-workers', a.workers])
    return prepared_source(a, a.data_root / 'stackexchange_2025_md', fetch,
                          serializer='thread-text')


if __name__ == "__main__":
    main()
