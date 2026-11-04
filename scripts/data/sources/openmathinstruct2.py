"""Download, shuffle, tokenize and pack OpenMathInstruct-2."""
from ..utils.workflow import parser, parse, hf_fetch, prepared_source


def main(argv=None):
    a = parse(parser(__doc__, revision='469216e3f46f4dacf476b382e192485ea51a143e'), argv)
    return prepared_source(a, a.data_root / 'openmathinstruct2',
        lambda raw: hf_fetch(a, raw, 'nvidia/OpenMathInstruct-2'),
        serializer='problem-generated-solution')


if __name__ == "__main__":
    main()
