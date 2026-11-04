"""Download all OpenMathReasoning splits, shuffle, tokenize and pack."""
from ..utils.workflow import parser, parse, prepared_source, run


def main(argv=None):
    a = parse(parser(__doc__, revision='d3d08664755704f422af97d43a7ff0ded4bd95df'), argv)
    def fetch(raw):
        run('shuffle_openmathreasoning', ['--revision', a.revision,
            '--output-root', raw, '--seed', a.seed, '--num-shards', a.num_shards,
            '--download-num-proc', a.workers, '--write-workers', a.workers])
    return prepared_source(a, a.data_root / 'openmathreasoning', fetch,
        serializer='problem-generated-solution',
        packed=a.data_root / f'openmathreasoning_smollm2_packed_seq4096_s{a.seed}')


if __name__ == "__main__":
    main()
