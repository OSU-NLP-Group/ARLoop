"""Download, shuffle, tokenize and pack GitHub issues or Kaggle notebooks."""
from ..utils.workflow import parser, parse, hf_fetch, prepared_source


def main(argv=None):
    a = parse(parser(__doc__, subsets=('issues', 'kaggle')), argv)
    return prepared_source(a, a.data_root / 'issues_kaggle_notebooks' / a.subset,
        lambda raw: hf_fetch(a, raw, 'HuggingFaceTB/issues-kaggle-notebooks', config=a.subset),
        source='issues_kaggle', subset=a.subset)


if __name__ == "__main__":
    main()
