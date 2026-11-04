"""Shared SmolLM2 EOS-only tokenization policy."""

from pathlib import Path
from loopedlm.tokenizer import Tokenizer


def tokenizer_properties(tokenizer_path: Path) -> tuple[Tokenizer, int]:
    tokenizer = Tokenizer(tokenizer_path)
    eos_id = int(tokenizer.get_eos_token_id())
    if tokenizer.pad_id is not None or tokenizer.processor.pad_token_id is not None:
        raise ValueError("Tokenizer must match official SmolLM2 and have no PAD token")
    return (
        tokenizer,
        eos_id,
    )


def output_path(output_root: Path, source_path: Path) -> Path:
    return output_root / "parquet" / source_path.name


def stats_path(output_root: Path, source_path: Path) -> Path:
    return output_root / ".stats" / f"{source_path.name}.json"


def sample_indices(length: int) -> list[int]:
    if length <= 0:
        return []
    return sorted({0, length // 2, length - 1})
