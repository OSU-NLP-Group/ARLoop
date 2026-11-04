"""Download, shuffle, tokenize and pack a FineMath/InfiWebMath subset."""
from ..utils.workflow import parser, parse, hf_fetch, prepared_source

SUBSETS = ('finemath-3plus', 'finemath-4plus', 'infiwebmath-3plus', 'infiwebmath-4plus')


def main(argv=None):
    a = parse(parser(__doc__, subsets=SUBSETS), argv)
    return prepared_source(a, a.data_root / 'finemath' / a.subset,
        lambda raw: hf_fetch(a, raw, 'HuggingFaceTB/finemath', config=a.subset))


if __name__ == "__main__":
    main()
