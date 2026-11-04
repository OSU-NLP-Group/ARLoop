#!/usr/bin/env python3
"""Train the control byte-level BPE tokenizer from local FineWeb-Edu Parquet."""

# MIT License
#
# Copyright (c) 2026 Hayden Prairie
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer, pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# Special tokens for the tokenizer
SPECIAL_TOKENS = [
    "<|bos|>",  # beginning of sequence
    "<|eos|>",  # end of sequence
    "<|pad|>",  # padding token
]

# GPT-4 style split pattern
# NOTE: Using \p{N}{1,2} instead of \p{N}{1,3} to avoid wasting tokens on numbers for smaller vocabs
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""



def create_gpt4_style_tokenizer():
    """Create a GPT-4 style BPE tokenizer configuration."""
    tokenizer = Tokenizer(BPE(
        byte_fallback=True,
        unk_token=None,
        fuse_unk=False,
    ))

    # No normalizer (like GPT-4)
    tokenizer.normalizer = None

    # GPT-4 style pre-tokenizer
    gpt4_split_regex = Regex(SPLIT_PATTERN)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])

    # ByteLevel decoder (pairs with ByteLevel pre-tokenizer)
    tokenizer.decoder = decoders.ByteLevel()

    # No post-processor
    tokenizer.post_processor = None

    return tokenizer


def train_tokenizer(text_iterator, vocab_size):
    """Train a BPE tokenizer from a text iterator."""
    tokenizer = create_gpt4_style_tokenizer()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        min_frequency=0,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.train_from_iterator(text_iterator, trainer)
    return tokenizer


def iter_text(data_dir, *, max_chars, doc_cap, holdout_files):
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {data_dir}")
    # The original recipe reserves the last shard when more than one is present.
    if holdout_files and len(files) > 1:
        if holdout_files >= len(files):
            raise ValueError("holdout-files must leave at least one training shard")
        files = files[:-holdout_files]
    nchars = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for index in range(parquet.num_row_groups):
            for text in parquet.read_row_group(index, columns=["text"])["text"].to_pylist():
                if not isinstance(text, str):
                    continue
                text = text[:doc_cap]
                nchars += len(text)
                yield text
                # Preserve the original full-document character-budget boundary.
                if nchars > max_chars:
                    return


def save_tokenizer(tokenizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=SPECIAL_TOKENS[0], eos_token=SPECIAL_TOKENS[1],
        pad_token=SPECIAL_TOKENS[2], clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.save_pretrained(output_dir)
    special_ids = {tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    token_bytes = torch.tensor([
        0 if i in special_ids else len(tokenizer.decode([i]).encode("utf-8"))
        for i in range(tokenizer.get_vocab_size())
    ], dtype=torch.int32)
    torch.save(token_bytes, output_dir / "token_bytes.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing raw Parquet shards with a text column")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "outputs/parcae_tokenizer")
    parser.add_argument("--max-chars", type=int, default=2_000_000_000)
    parser.add_argument("--doc-cap", type=int, default=10_000)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--holdout-files", type=int, default=1,
                        help="Reserve the last N sorted shards; use 0 for an existing train-only split")
    args = parser.parse_args()
    if args.max_chars <= 0 or args.doc_cap <= 0:
        parser.error("max-chars and doc-cap must be positive")
    if args.vocab_size < 256 + len(SPECIAL_TOKENS):
        parser.error("vocab-size must fit the 256 byte tokens and three special tokens")
    if args.holdout_files < 0:
        parser.error("holdout-files must be nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty; use a new directory for a new tokenizer")
    tokenizer = train_tokenizer(iter_text(
        args.data_dir, max_chars=args.max_chars, doc_cap=args.doc_cap,
        holdout_files=args.holdout_files,
    ), args.vocab_size)
    text = "Hello world! 123 4567 I'm here. 你好世界 🌍\n"
    if tokenizer.decode(tokenizer.encode(text).ids) != text:
        raise RuntimeError("Tokenizer roundtrip check failed")
    save_tokenizer(tokenizer, args.output_dir)
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) to {args.output_dir}")


if __name__ == "__main__":
    main()
