"""Download, shuffle, tokenize and pack a Dolmino mathematics subset."""
from ..utils.workflow import parser, parse, text_fetch, prepared_source

SUBSETS = {
    'basic_math': 'math/dolmino_math_synth/basic_math',
    'gsm8k-synth': 'math/dolmino_math_synth/gsm8k-synth',
    'gsm_mind': 'math/dolmino_math_synth/gsm_mind',
    'gsm8k': 'math/gsm8k',
    'tinyGSM-MIND': 'math/tinyGSM-MIND',
}


def main(argv=None):
    a = parse(parser(__doc__, subsets=tuple(SUBSETS)), argv)
    path = SUBSETS[a.subset]
    return prepared_source(a, a.data_root / 'dolmino_mix_1124' / path,
        lambda raw: text_fetch(a, raw, 'allenai/dolmino-mix-1124', ['data/' + path + '/*']),
        source='dolmino', subset=a.subset)


if __name__ == "__main__":
    main()
