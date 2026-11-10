"""Download Natural Reasoning, serialize question/response, shuffle and pack."""
from ..utils.workflow import parser, parse, text_fetch, prepared_source


def main(argv=None):
    p = parser(__doc__)
    p.add_argument('--include-pattern', action='append', default=None,
                   help='Upstream JSON/JSONL file glob; repeat to select several paths')
    a = parse(p, argv)
    return prepared_source(a, a.data_root / 'natural_reasoning',
        lambda raw: text_fetch(a, raw, 'facebook/natural_reasoning',
            a.include_pattern or ['*.jsonl*'], serializer='natural-reasoning'), source='natural_reasoning')


if __name__ == "__main__":
    main()
