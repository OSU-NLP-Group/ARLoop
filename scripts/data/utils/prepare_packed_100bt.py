#!/usr/bin/env python3
"""Build deterministic offline nanochat-style BestFit shards."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from bisect import bisect_right, insort
from collections import deque
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from loopedlm.tokenizer import Tokenizer  # noqa: E402

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "100bt_tokenized"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "100bt_packed_2049"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "parcae_tokenizer"
PACK_DTYPE = np.dtype("<u2")
TOKEN_COLUMN = "input_ids"


@dataclass(frozen=True)
class Lane:
    split: str
    lane_id: int
    num_lanes: int
    source_files: tuple[Path, ...]

    @property
    def stem(self) -> str:
        return f"lane-{self.lane_id:03d}-of-{self.num_lanes:03d}"


from .common import write_json


def discover_split_files(source_root: Path, split: str, seed: int) -> list[Path]:
    files = sorted((source_root / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {source_root / split}")
    random.Random(seed).shuffle(files)
    return files


def make_lane(
    source_root: Path,
    split: str,
    lane_id: int,
    requested_lanes: int,
    seed: int,
) -> Lane | None:
    files = discover_split_files(source_root, split, seed)
    num_lanes = min(requested_lanes, len(files))
    if lane_id >= num_lanes:
        return None
    return Lane(split, lane_id, num_lanes, tuple(files[lane_id::num_lanes]))


def lane_output_path(output_root: Path, lane: Lane) -> Path:
    return output_root / lane.split / f"{lane.stem}.bin"


def lane_stats_path(output_root: Path, lane: Lane) -> Path:
    return output_root / ".stats" / lane.split / f"{lane.stem}.json"


class ParquetDocumentStream:
    """Yield shuffled tokenized documents from an explicit ordered file lane."""

    def __init__(self, files: tuple[Path, ...], seed: int, vocab_size: int):
        self.files = files
        self.seed = seed
        self.vocab_size = vocab_size
        self.documents = 0
        self.tokens = 0
        self.row_groups = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        rng = random.Random(self.seed)
        for file_index, path in enumerate(self.files):
            parquet_file = pq.ParquetFile(path)
            if parquet_file.schema_arrow.names != [TOKEN_COLUMN]:
                raise ValueError(
                    f"{path}: expected only {TOKEN_COLUMN}, got "
                    f"{parquet_file.schema_arrow.names}"
                )
            for row_group_index in range(parquet_file.num_row_groups):
                token_column = (
                    parquet_file.read_row_group(row_group_index, columns=[TOKEN_COLUMN])
                    .column(TOKEN_COLUMN)
                    .combine_chunks()
                )
                values = token_column.values
                if len(values):
                    minimum = int(pc.min(values).as_py())
                    maximum = int(pc.max(values).as_py())
                    if minimum < 0 or maximum >= self.vocab_size:
                        raise ValueError(
                            f"{path} row group {row_group_index}: token range "
                            f"[{minimum}, {maximum}] outside [0, {self.vocab_size})"
                        )
                rows = token_column.to_pylist()
                rng.shuffle(rows)
                self.row_groups += 1
                while rows:
                    tokens = np.asarray(rows.pop(), dtype=PACK_DTYPE)
                    self.documents += 1
                    self.tokens += int(tokens.size)
                    yield tokens
            print(
                f"source_file={file_index + 1}/{len(self.files)} "
                f"name={path.name} docs={self.documents:,} "
                f"tokens={self.tokens:,}",
                flush=True,
            )


class OfflineBestFitPacker:
    """Equivalent indexed BestFit policy with explicit crop statistics."""

    def __init__(
        self,
        documents: Iterator[np.ndarray],
        *,
        capacity: int,
        buffer_size: int,
        bos_id: int,
        pad_id: int | None,
    ):
        self.documents = documents
        self.capacity = capacity
        self.buffer_size = buffer_size
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.doc_buffer: list[np.ndarray | None] = []
        self.active_documents = 0
        self.indices_by_length: dict[int, deque[int]] = {}
        self.sorted_lengths: list[int] = []
        self.source_exhausted = False
        self.source_documents = 0
        self.source_tokens = 0
        self.source_tokens_with_bos = 0
        self.selected_documents = 0
        self.cropped_documents = 0
        self.discarded_tail_tokens = 0
        self.discarded_incomplete_row_tokens = 0
        self.padding_tokens = 0
        self.output_rows = 0
        self.compactions = 0

    def _buffered_documents(self) -> list[np.ndarray]:
        return [document for document in self.doc_buffer if document is not None]

    def _rebuild_index(self) -> None:
        documents = self._buffered_documents()
        self.doc_buffer = list(documents)
        self.active_documents = len(documents)
        self.indices_by_length = {}
        self.sorted_lengths = []
        for index, document in enumerate(documents):
            length = len(document)
            if length not in self.indices_by_length:
                self.indices_by_length[length] = deque()
                insort(self.sorted_lengths, length)
            self.indices_by_length[length].append(index)
        self.compactions += 1

    def _maybe_compact(self) -> None:
        if len(self.doc_buffer) > max(2 * self.buffer_size, 2 * self.active_documents):
            self._rebuild_index()

    def _append_document(self, tokens: np.ndarray) -> None:
        document = np.empty(tokens.size + 1, dtype=PACK_DTYPE)
        document[0] = self.bos_id
        document[1:] = tokens
        index = len(self.doc_buffer)
        self.doc_buffer.append(document)
        length = len(document)
        if length not in self.indices_by_length:
            self.indices_by_length[length] = deque()
            insort(self.sorted_lengths, length)
        self.indices_by_length[length].append(index)
        self.active_documents += 1
        self.source_documents += 1
        self.source_tokens += int(tokens.size)
        self.source_tokens_with_bos += int(document.size)

    def _refill(self) -> None:
        self._maybe_compact()
        while self.active_documents < self.buffer_size and not self.source_exhausted:
            try:
                tokens = next(self.documents)
            except StopIteration:
                self.source_exhausted = True
                break
            self._append_document(tokens)

    def _pop_best_fit(self, remaining: int) -> np.ndarray:
        length_position = bisect_right(self.sorted_lengths, remaining) - 1
        fits = length_position >= 0
        if not fits:
            length_position = 0
        document_length = self.sorted_lengths[length_position]
        indices = self.indices_by_length[document_length]
        document_index = indices.popleft()
        if not indices:
            del self.indices_by_length[document_length]
            del self.sorted_lengths[length_position]
        document = self.doc_buffer[document_index]
        if document is None:
            raise RuntimeError("BestFit length index referenced a removed document")
        self.doc_buffer[document_index] = None
        self.active_documents -= 1
        self.selected_documents += 1
        if fits:
            return document
        self.cropped_documents += 1
        self.discarded_tail_tokens += len(document) - remaining
        return document[:remaining]

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            row = np.empty(self.capacity, dtype=PACK_DTYPE)
            position = 0
            while position < self.capacity:
                self._refill()
                if self.active_documents == 0:
                    if position == 0:
                        return
                    if self.pad_id is None:
                        self.discarded_incomplete_row_tokens += position
                        return
                    padding = self.capacity - position
                    row[position:] = self.pad_id
                    self.padding_tokens += padding
                    position = self.capacity
                    break
                document = self._pop_best_fit(self.capacity - position)
                row[position : position + len(document)] = document
                position += len(document)
            self.output_rows += 1
            yield row

    def stats(self) -> dict:
        return {
            "source_documents": self.source_documents,
            "source_tokens": self.source_tokens,
            "source_tokens_with_bos": self.source_tokens_with_bos,
            "selected_documents": self.selected_documents,
            "cropped_documents": self.cropped_documents,
            "discarded_tail_tokens": self.discarded_tail_tokens,
            "discarded_incomplete_row_tokens": (self.discarded_incomplete_row_tokens),
            "padding_tokens": self.padding_tokens,
            "output_rows": self.output_rows,
            "output_capacity_tokens": self.output_rows * self.capacity,
            "compactions": self.compactions,
        }


def validate_completed_lane(
    output_path: Path,
    stats_path: Path,
    expected: dict,
) -> dict:
    if not output_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Incomplete lane: {output_path}")
    stats = json.loads(stats_path.read_text())
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(
                f"{stats_path}: expected {key}={value!r}, got {stats.get(key)!r}"
            )
    expected_size = int(stats["output_rows"]) * int(stats["row_bytes"])
    if output_path.stat().st_size != expected_size:
        raise ValueError(
            f"{output_path}: expected {expected_size} bytes, got "
            f"{output_path.stat().st_size}"
        )
    return stats


def convert_lane(
    lane: Lane,
    *,
    output_root: Path,
    tokenizer_path: Path,
    capacity: int,
    buffer_size: int,
    seed: int,
    write_batch_rows: int,
    force: bool,
) -> dict:
    tokenizer = Tokenizer(tokenizer_path)
    bos_id = int(tokenizer.get_bos_token_id())
    pad_id = int(tokenizer.pad_id) if tokenizer.pad_id is not None else None
    vocab_size = int(tokenizer.vocab_size)
    if vocab_size > np.iinfo(PACK_DTYPE).max + 1:
        raise ValueError(f"Vocabulary size {vocab_size} does not fit uint16")

    output_path = lane_output_path(output_root, lane)
    stats_path = lane_stats_path(output_root, lane)
    expected = {
        "complete": True,
        "split": lane.split,
        "lane_id": lane.lane_id,
        "num_lanes": lane.num_lanes,
        "source_files": [str(path) for path in lane.source_files],
        "capacity": capacity,
        "buffer_size": buffer_size,
        "seed": seed,
        "bos_id": bos_id,
        "pad_id": pad_id,
        "dtype": "uint16-le",
    }
    if output_path.exists() and not force:
        stats = validate_completed_lane(output_path, stats_path, expected)
        print(f"SKIP valid lane: {output_path}", flush=True)
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getenv('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)
    if force:
        output_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)

    stream = ParquetDocumentStream(lane.source_files, seed + lane.lane_id, vocab_size)
    packer = OfflineBestFitPacker(
        iter(stream),
        capacity=capacity,
        buffer_size=buffer_size,
        bos_id=bos_id,
        pad_id=pad_id,
    )
    write_buffer = np.empty((write_batch_rows, capacity), dtype=PACK_DTYPE)
    buffered_rows = 0
    started = time.time()
    last_report = started

    try:
        with temporary.open("wb", buffering=32 * 1024 * 1024) as handle:
            for row in packer:
                write_buffer[buffered_rows] = row
                buffered_rows += 1
                if buffered_rows == write_batch_rows:
                    payload = write_buffer.tobytes(order="C")
                    handle.write(payload)
                    buffered_rows = 0
                now = time.time()
                if now - last_report >= 60:
                    elapsed = max(now - started, 1e-9)
                    print(
                        f"split={lane.split} lane={lane.lane_id}/"
                        f"{lane.num_lanes} rows={packer.output_rows:,} "
                        f"tokens/s={packer.output_rows * capacity / elapsed:,.0f}",
                        flush=True,
                    )
                    last_report = now
            if buffered_rows:
                payload = write_buffer[:buffered_rows].tobytes(order="C")
                handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    os.replace(temporary, output_path)
    elapsed = time.time() - started
    payload = {
        **expected,
        **packer.stats(),
        "source_row_groups": stream.row_groups,
        "source_size_bytes": sum(path.stat().st_size for path in lane.source_files),
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "row_bytes": capacity * PACK_DTYPE.itemsize,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (packer.output_rows * capacity / max(elapsed, 1e-9)),
        "algorithm": "nanochat_bestfit_length_index_v1",
        "add_bos": True,
        "add_eos": False,
        "crop_tail_when_no_document_fits": True,
    }
    write_json(stats_path, payload)
    validate_completed_lane(output_path, stats_path, expected)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def expected_lanes(source_root: Path, requested_lanes: int, seed: int) -> list[Lane]:
    lanes = []
    for split in ("train", "val"):
        for lane_id in range(requested_lanes):
            lane = make_lane(source_root, split, lane_id, requested_lanes, seed)
            if lane is not None:
                lanes.append(lane)
    return lanes


def replay_prefix(
    lane: Lane,
    *,
    tokenizer: Tokenizer,
    capacity: int,
    buffer_size: int,
    seed: int,
    rows: int,
) -> Iterator[np.ndarray]:
    stream = ParquetDocumentStream(
        lane.source_files, seed + lane.lane_id, int(tokenizer.vocab_size)
    )
    packer = OfflineBestFitPacker(
        iter(stream),
        capacity=capacity,
        buffer_size=buffer_size,
        bos_id=int(tokenizer.get_bos_token_id()),
        pad_id=(int(tokenizer.pad_id) if tokenizer.pad_id is not None else None),
    )
    yield from islice(iter(packer), rows)


def verify_dataset(
    *,
    source_root: Path,
    output_root: Path,
    tokenizer_path: Path,
    requested_lanes: int,
    capacity: int,
    buffer_size: int,
    seed: int,
    replay_rows: int,
    write_success: bool,
) -> dict:
    tokenizer = Tokenizer(tokenizer_path)
    lanes = expected_lanes(source_root, requested_lanes, seed)
    stats_list = []
    stats_by_lane = {}
    assigned_files = []
    for lane in lanes:
        expected = {
            "complete": True,
            "split": lane.split,
            "lane_id": lane.lane_id,
            "num_lanes": lane.num_lanes,
            "source_files": [str(path) for path in lane.source_files],
            "capacity": capacity,
            "buffer_size": buffer_size,
            "seed": seed,
            "bos_id": int(tokenizer.get_bos_token_id()),
            "pad_id": (int(tokenizer.pad_id) if tokenizer.pad_id is not None else None),
            "dtype": "uint16-le",
        }
        stats = validate_completed_lane(
            lane_output_path(output_root, lane),
            lane_stats_path(output_root, lane),
            expected,
        )
        stats_list.append(stats)
        stats_by_lane[(lane.split, lane.lane_id)] = stats
        assigned_files.extend(stats["source_files"])

    discovered = [
        str(path)
        for split in ("train", "val")
        for path in discover_split_files(source_root, split, seed)
    ]
    if sorted(assigned_files) != sorted(discovered):
        raise ValueError("Lane source assignments are incomplete or duplicated")

    for split in ("train", "val"):
        lane = next(item for item in lanes if item.split == split)
        output = lane_output_path(output_root, lane)
        memmap = np.memmap(
            output,
            mode="r",
            dtype=PACK_DTYPE,
            shape=(
                int(stats_by_lane[(lane.split, lane.lane_id)]["output_rows"]),
                capacity,
            ),
        )
        compare_rows = min(replay_rows, len(memmap))
        for row_index, expected_row in enumerate(
            replay_prefix(
                lane,
                tokenizer=tokenizer,
                capacity=capacity,
                buffer_size=buffer_size,
                seed=seed,
                rows=compare_rows,
            )
        ):
            if not np.array_equal(memmap[row_index], expected_row):
                raise ValueError(
                    f"Replay mismatch in {split} lane {lane.lane_id}, row {row_index}"
                )
        if compare_rows and not np.all(
            memmap[:compare_rows, 0] == int(tokenizer.bos_id)
        ):
            raise ValueError(f"{split} replay rows do not all begin with BOS")
        del memmap
        print(
            f"replayed {compare_rows} rows for {split}/{lane.stem}",
            flush=True,
        )

    aggregate_keys = [
        "source_documents",
        "source_tokens",
        "source_tokens_with_bos",
        "selected_documents",
        "cropped_documents",
        "discarded_tail_tokens",
        "discarded_incomplete_row_tokens",
        "padding_tokens",
        "output_rows",
        "output_capacity_tokens",
        "source_size_bytes",
        "output_size_bytes",
    ]
    totals = {
        key: sum(int(stats[key]) for stats in stats_list) for key in aggregate_keys
    }
    manifest = {
        "complete": write_success,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": int(tokenizer.vocab_size),
        "bos_id": int(tokenizer.get_bos_token_id()),
        "pad_id": (int(tokenizer.pad_id) if tokenizer.pad_id is not None else None),
        "add_bos": True,
        "add_eos": False,
        "algorithm": "nanochat_bestfit_length_index_v1",
        "crop_tail_when_no_document_fits": True,
        "capacity": capacity,
        "buffer_size": buffer_size,
        "seed": seed,
        "dtype": "uint16-le",
        "row_layout": f"input_ids[{capacity}]",
        "requested_lanes": requested_lanes,
        "train_lanes": sum(lane.split == "train" for lane in lanes),
        "val_lanes": sum(lane.split == "val" for lane in lanes),
        "replay_rows_per_split": replay_rows,
        "totals": totals,
        "lane_stats": stats_list,
    }
    if write_success:
        write_json(output_root / "manifest.json", manifest)
        success = {
            "manifest": "manifest.json",
            "output_rows": totals["output_rows"],
            "output_capacity_tokens": totals["output_capacity_tokens"],
        }
        write_json(output_root / "_SUCCESS", success)
    return manifest


def create_test_source(root: Path, vocab_size: int) -> None:
    schema = pa.schema([pa.field(TOKEN_COLUMN, pa.list_(pa.int32()))])
    rng = random.Random(7)
    for split, files in (("train", 4), ("val", 2)):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        for file_index in range(files):
            path = split_dir / f"{split}-{file_index:03d}.parquet"
            writer = pq.ParquetWriter(path, schema, compression="zstd")
            for row_group in range(2):
                rows = []
                for doc_index in range(80):
                    length = (
                        1 + (file_index * 97 + row_group * 43 + doc_index * 31) % 5000
                    )
                    rows.append(
                        [rng.randrange(1, max(2, vocab_size)) for _ in range(length)]
                    )
                writer.write_table(
                    pa.Table.from_arrays(
                        [pa.array(rows, type=schema.field(TOKEN_COLUMN).type)],
                        schema=schema,
                    ),
                    row_group_size=len(rows),
                )
            writer.close()


def verify_reference_packer_equivalence(tokenizer: Tokenizer) -> None:
    import torch
    from torch.utils.data import IterableDataset

    from scripts.data.utils.packing_reference import NanochatBestFitPackingDataset

    rng = np.random.default_rng(20260727)
    lengths = [
        1,
        2,
        17,
        31,
        64,
        128,
        255,
        256,
        257,
        511,
        1024,
        *rng.integers(1, 800, size=300).tolist(),
    ]
    documents = [
        rng.integers(
            1,
            int(tokenizer.vocab_size),
            size=int(length),
            dtype=np.int64,
        ).astype(PACK_DTYPE)
        for length in lengths
    ]

    class Documents(IterableDataset):
        def __iter__(self):
            yield from (
                torch.from_numpy(document.astype(np.int64, copy=False))
                for document in documents
            )

    capacity = 257
    buffer_size = 32
    reference = NanochatBestFitPackingDataset(
        source_dataset=Documents(),
        tokenizer=tokenizer,
        block_size=capacity,
        buffer_size=buffer_size,
        stateful=False,
        use_length_index=True,
        prefetch_rows=0,
    )
    offline = OfflineBestFitPacker(
        iter(documents),
        capacity=capacity,
        buffer_size=buffer_size,
        bos_id=int(tokenizer.get_bos_token_id()),
        pad_id=(int(tokenizer.pad_id) if tokenizer.pad_id is not None else None),
    )
    reference_rows = [row.numpy().astype(PACK_DTYPE) for row in reference]
    offline_rows = list(offline)
    if tokenizer.pad_id is None and offline.discarded_incomplete_row_tokens:
        reference_rows.pop()
    if len(reference_rows) != len(offline_rows):
        raise AssertionError(
            f"Reference emitted {len(reference_rows)} rows; "
            f"offline emitted {len(offline_rows)}"
        )
    for row_index, (reference_row, offline_row) in enumerate(
        zip(reference_rows, offline_rows)
    ):
        if not np.array_equal(reference_row, offline_row):
            mismatch = int(np.flatnonzero(reference_row != offline_row)[0])
            raise AssertionError(
                f"Reference mismatch at row {row_index}, token {mismatch}: "
                f"{reference_row[mismatch]} != {offline_row[mismatch]}"
            )
    print(
        f"reference NanochatBestFitPackingDataset matched exactly for "
        f"{len(reference_rows):,} rows",
        flush=True,
    )


def run_self_test(tokenizer_path: Path) -> None:
    tokenizer = Tokenizer(tokenizer_path)
    verify_reference_packer_equivalence(tokenizer)
    with tempfile.TemporaryDirectory(prefix="packed-100bt-self-test-") as temp:
        root = Path(temp)
        source_root = root / "source"
        output_root = root / "output"
        create_test_source(source_root, int(tokenizer.vocab_size))
        for lane_id in range(2):
            for split in ("train", "val"):
                lane = make_lane(source_root, split, lane_id, 2, 42)
                if lane is not None:
                    convert_lane(
                        lane,
                        output_root=output_root,
                        tokenizer_path=tokenizer_path,
                        capacity=2049,
                        buffer_size=32,
                        seed=42,
                        write_batch_rows=16,
                        force=False,
                    )
        result = verify_dataset(
            source_root=source_root,
            output_root=output_root,
            tokenizer_path=tokenizer_path,
            requested_lanes=2,
            capacity=2049,
            buffer_size=32,
            seed=42,
            replay_rows=32,
            write_success=True,
        )
        assert result["complete"]
        assert (output_root / "_SUCCESS").is_file()
        print("SELF TEST PASSED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--capacity", type=int, default=2049)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-lanes", type=int, default=64)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert-lane")
    convert.add_argument("--lane-id", type=int, required=True)
    convert.add_argument("--write-batch-rows", type=int, default=4096)
    convert.add_argument("--force", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--replay-rows", type=int, default=4096)
    verify.add_argument("--write-success", action="store_true")
    subparsers.add_parser("self-test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.capacity < 2:
        raise ValueError("capacity must be at least 2")
    if args.buffer_size < 1:
        raise ValueError("buffer-size must be positive")
    if args.num_lanes < 1:
        raise ValueError("num-lanes must be positive")

    if args.command == "convert-lane":
        results = []
        for split in ("train", "val"):
            lane = make_lane(
                args.source_root,
                split,
                args.lane_id,
                args.num_lanes,
                args.seed,
            )
            if lane is not None:
                results.append(
                    convert_lane(
                        lane,
                        output_root=args.output_root,
                        tokenizer_path=args.tokenizer_path,
                        capacity=args.capacity,
                        buffer_size=args.buffer_size,
                        seed=args.seed,
                        write_batch_rows=args.write_batch_rows,
                        force=args.force,
                    )
                )
        print(json.dumps(results, indent=2, sort_keys=True))
    elif args.command == "verify":
        result = verify_dataset(
            source_root=args.source_root,
            output_root=args.output_root,
            tokenizer_path=args.tokenizer_path,
            requested_lanes=args.num_lanes,
            capacity=args.capacity,
            buffer_size=args.buffer_size,
            seed=args.seed,
            replay_rows=args.replay_rows,
            write_success=args.write_success,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "self-test":
        run_self_test(args.tokenizer_path)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
