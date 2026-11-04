"""Deterministic integer mixing and partition accounting shared by shufflers."""

import math
import numpy as np

UINT64_MASK = (1 << 64) - 1


def require_power_of_two(value: int) -> None:
    if value < 2 or value > (1 << 16) or value & (value - 1):
        raise ValueError("--num-buckets must be a power of two in [2, 65536]")


def splitmix64(
    values: np.ndarray, seed: int, shuffle_salt: int = 0x243F6A8885A308D3
) -> np.ndarray:
    if values.dtype != np.uint64:
        raise TypeError(f"Expected uint64 document IDs, got {values.dtype}")
    salt = (int(seed) ^ shuffle_salt) & UINT64_MASK
    with np.errstate(over="ignore"):
        mixed = values + np.uint64((salt + 0x9E3779B97F4A7C15) & UINT64_MASK)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
    return mixed


def bucket_ids(keys: np.ndarray, num_buckets: int) -> np.ndarray:
    bits = int(math.log2(num_buckets))
    return (keys >> np.uint64(64 - bits)).astype(np.uint32, copy=False)


def aggregate(records: list[dict]) -> dict:
    result = {
        "documents": 0,
        "token_count": 0,
    }
    for record in records:
        result["documents"] += int(record["documents"])
        result["token_count"] += int(record["token_count"])
    return result
