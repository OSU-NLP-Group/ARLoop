#!/usr/bin/env python3
"""BestFit-pack the same token prefix selected by a concat-pack manifest.

Source documents already end in EOS.  Packing preserves those token arrays and
does not inject BOS, EOS, or padding.  The BestFit policy matches the previous
offline packer: keep a bounded document buffer, choose the longest document
that fits the current row, and crop the shortest document when none fits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from bisect import bisect_right, insort
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .prepare_concat_packed_prefix import write_json, load_source_manifest


TOKEN_COLUMN = "input_ids"
PACK_DTYPE = np.dtype("<u2")
ALGORITHM = "nanochat_bestfit_length_index_no_special_tokens_v1"


def plan_path(output_root: Path) -> Path:
    return output_root / "packing_plan.json"


def lane_output_path(output_root: Path, lane_id: int, num_lanes: int) -> Path:
    return output_root / "train" / f"lane-{lane_id:03d}-of-{num_lanes:03d}.bin"


def lane_stats_path(output_root: Path, lane_id: int, num_lanes: int) -> Path:
    return output_root / ".stats" / f"lane-{lane_id:03d}-of-{num_lanes:03d}.json"


def load_reference_manifest(path: Path, source_root: Path, capacity: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    expected = {
        "complete": True,
        "algorithm": "stream_concat_eos_documents_v1",
        "source_root": str(source_root.resolve()),
        "capacity": capacity,
        "add_bos_during_packing": False,
        "add_eos_during_packing": False,
        "padding_tokens": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Reference manifest has {key}={manifest.get(key)!r}, "
                f"expected {value!r}"
            )
    lanes = manifest.get("lane_stats")
    if not lanes:
        raise ValueError("Reference manifest has no lane_stats")
    return manifest


def build_plan(args: argparse.Namespace) -> dict:
    if (args.output_root / "_SUCCESS").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root / '_SUCCESS'}")
    source_manifest = load_source_manifest(args.source_root)
    reference = load_reference_manifest(
        args.reference_manifest, args.source_root, args.capacity
    )
    reference_lanes = sorted(reference["lane_stats"], key=lambda item: item["lane_id"])
    num_lanes = len(reference_lanes)
    lanes = []
    for expected_id, reference_lane in enumerate(reference_lanes):
        if int(reference_lane["lane_id"]) != expected_id:
            raise ValueError("Reference lane ids are not contiguous")
        source_path = Path(reference_lane["source_path"]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        try:
            source_path.relative_to(args.source_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Source path is outside source root: {source_path}"
            ) from exc
        lanes.append(
            {
                "lane_id": expected_id,
                "num_lanes": num_lanes,
                "source_path": str(source_path),
                "source_token_budget": int(reference_lane["source_tokens_consumed"]),
                "reference_output_rows": int(reference_lane["output_rows"]),
                "reference_output_tokens": int(reference_lane["output_tokens"]),
                "output_path": str(
                    lane_output_path(args.output_root, expected_id, num_lanes).resolve()
                ),
            }
        )
    target_source_tokens = sum(lane["source_token_budget"] for lane in lanes)
    if target_source_tokens != int(reference["output_capacity_tokens"]):
        raise ValueError(
            f"Lane source budgets sum to {target_source_tokens:,}, reference has "
            f"{int(reference['output_capacity_tokens']):,}"
        )
    payload = {
        "version": 1,
        "complete": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "algorithm": ALGORITHM,
        "source_root": str(args.source_root.resolve()),
        "source_manifest": str((args.source_root / "manifest.json").resolve()),
        "source_eos_token_id": int(source_manifest["eos_token_id"]),
        "source_special_tokens_policy": source_manifest["special_tokens_policy"],
        "reference_manifest": str(args.reference_manifest.resolve()),
        "reference_algorithm": reference["algorithm"],
        "output_root": str(args.output_root.resolve()),
        "capacity": args.capacity,
        "training_sequence_length": args.capacity - 1,
        "target_source_tokens": target_source_tokens,
        "num_lanes": num_lanes,
        "buffer_size": args.buffer_size,
        "dtype": "uint16-le",
        "add_bos_during_packing": False,
        "add_eos_during_packing": False,
        "padding_tokens": 0,
        "crop_tail_when_no_document_fits": True,
        "drop_final_incomplete_row": True,
        "lanes": lanes,
    }
    existing = plan_path(args.output_root)
    if existing.is_file():
        previous = json.loads(existing.read_text())
        stable_keys = (
            "algorithm",
            "source_root",
            "capacity",
            "target_source_tokens",
            "num_lanes",
            "buffer_size",
            "lanes",
        )
        for key in stable_keys:
            if previous.get(key) != payload.get(key):
                raise ValueError(f"Existing packing plan differs at {key}")
        print(f"SKIP valid existing plan: {existing}", flush=True)
        return previous
    write_json(existing, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def load_plan(args: argparse.Namespace) -> dict:
    path = plan_path(args.output_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    plan = json.loads(path.read_text())
    expected = {
        "algorithm": ALGORITHM,
        "source_root": str(args.source_root.resolve()),
        "reference_manifest": str(args.reference_manifest.resolve()),
        "output_root": str(args.output_root.resolve()),
        "capacity": args.capacity,
        "buffer_size": args.buffer_size,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Plan has {key}={plan.get(key)!r}, expected {value!r}")
    return plan


class PrefixDocumentStream:
    """Yield complete EOS-terminated documents within a per-lane token budget."""

    def __init__(self, path: Path, token_budget: int, eos_id: int):
        self.path = path
        self.token_budget = token_budget
        self.eos_id = eos_id
        self.documents = 0
        self.tokens = 0
        self.eos_tokens = 0
        self.row_groups = 0
        self.max_document_tokens = 0
        self.long_documents = 0
        self.boundary_document_tokens = 0

    @property
    def budget_shortfall_tokens(self) -> int:
        return self.token_budget - self.tokens

    def __iter__(self) -> Iterator[np.ndarray]:
        parquet_file = pq.ParquetFile(self.path)
        schema = parquet_file.schema_arrow.remove_metadata()
        if schema.names != [TOKEN_COLUMN]:
            raise ValueError(f"{self.path}: unexpected columns {schema.names}")
        token_type = schema.field(TOKEN_COLUMN).type
        if not pa.types.is_list(token_type) or token_type.value_type != pa.int32():
            raise ValueError(
                f"{self.path}: unexpected {TOKEN_COLUMN} type {token_type}"
            )

        for row_group_index in range(parquet_file.num_row_groups):
            if self.tokens == self.token_budget:
                return
            column = (
                parquet_file.read_row_group(row_group_index, columns=[TOKEN_COLUMN])
                .column(TOKEN_COLUMN)
                .combine_chunks()
            )
            if column.null_count:
                raise ValueError(f"{self.path} row group {row_group_index}: null docs")
            values = column.values.to_numpy(zero_copy_only=False)
            if len(values):
                minimum = int(values.min())
                maximum = int(values.max())
                if minimum < 0 or maximum > np.iinfo(PACK_DTYPE).max:
                    raise ValueError(
                        f"{self.path} row group {row_group_index}: token range "
                        f"[{minimum}, {maximum}] does not fit uint16"
                    )
            self.row_groups += 1
            offsets = column.offsets.to_numpy(zero_copy_only=False)
            flat_values = values.astype(PACK_DTYPE, copy=False)
            for row_index in range(len(column)):
                start = int(offsets[row_index])
                end = int(offsets[row_index + 1])
                tokens = flat_values[start:end]
                if not len(tokens):
                    raise ValueError(f"{self.path}: empty tokenized document")
                if int(tokens[-1]) != self.eos_id:
                    raise ValueError(
                        f"{self.path}: document {self.documents} does not end in EOS "
                        f"{self.eos_id}"
                    )
                if self.tokens + len(tokens) > self.token_budget:
                    self.boundary_document_tokens = len(tokens)
                    return
                self.documents += 1
                self.tokens += len(tokens)
                self.eos_tokens += 1
                self.max_document_tokens = max(self.max_document_tokens, len(tokens))
                yield tokens
        if self.tokens != self.token_budget:
            raise ValueError(
                f"{self.path}: exhausted source at {self.tokens:,}/"
                f"{self.token_budget:,} budgeted tokens"
            )


class BestFitPacker:
    """Previous indexed BestFit policy without special-token injection."""

    def __init__(
        self,
        documents: Iterator[np.ndarray],
        *,
        capacity: int,
        buffer_size: int,
        eos_id: int,
    ):
        self.documents = documents
        self.capacity = capacity
        self.buffer_size = buffer_size
        self.eos_id = eos_id
        self.doc_buffer: list[np.ndarray | None] = []
        self.active_documents = 0
        self.indices_by_length: dict[int, deque[int]] = {}
        self.sorted_lengths: list[int] = []
        self.source_exhausted = False
        self.source_documents = 0
        self.source_tokens = 0
        self.source_eos_tokens = 0
        self.selected_documents = 0
        self.cropped_documents = 0
        self.discarded_tail_tokens = 0
        self.discarded_tail_eos_tokens = 0
        self.discarded_incomplete_row_tokens = 0
        self.discarded_incomplete_row_eos_tokens = 0
        self.output_rows = 0
        self.output_eos_tokens = 0
        self.compactions = 0
        self.max_document_tokens = 0
        self.long_documents = 0

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

    def _append_document(self, document: np.ndarray) -> None:
        index = len(self.doc_buffer)
        self.doc_buffer.append(document)
        length = len(document)
        if length not in self.indices_by_length:
            self.indices_by_length[length] = deque()
            insort(self.sorted_lengths, length)
        self.indices_by_length[length].append(index)
        self.active_documents += 1
        self.source_documents += 1
        self.source_tokens += length
        self.source_eos_tokens += 1
        self.max_document_tokens = max(self.max_document_tokens, length)
        self.long_documents += int(length > self.capacity)

    def _refill(self) -> None:
        self._maybe_compact()
        while self.active_documents < self.buffer_size and not self.source_exhausted:
            try:
                document = next(self.documents)
            except StopIteration:
                self.source_exhausted = True
                break
            self._append_document(document)

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
        tail = document[remaining:]
        self.cropped_documents += 1
        self.discarded_tail_tokens += len(tail)
        self.discarded_tail_eos_tokens += int(document[-1] == self.eos_id)
        return document[:remaining]

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            row = np.empty(self.capacity, dtype=PACK_DTYPE)
            position = 0
            row_eos_tokens = 0
            while position < self.capacity:
                self._refill()
                if self.active_documents == 0:
                    if position:
                        self.discarded_incomplete_row_tokens += position
                        self.discarded_incomplete_row_eos_tokens += row_eos_tokens
                    return
                document = self._pop_best_fit(self.capacity - position)
                row[position : position + len(document)] = document
                position += len(document)
                row_eos_tokens += int(document[-1] == self.eos_id)
            self.output_rows += 1
            self.output_eos_tokens += row_eos_tokens
            yield row

    def stats(self) -> dict:
        return {
            "source_documents": self.source_documents,
            "source_tokens": self.source_tokens,
            "source_eos_tokens": self.source_eos_tokens,
            "selected_documents": self.selected_documents,
            "max_document_tokens": self.max_document_tokens,
            "long_documents_gt_capacity": self.long_documents,
            "cropped_documents": self.cropped_documents,
            "discarded_tail_tokens": self.discarded_tail_tokens,
            "discarded_tail_eos_tokens": self.discarded_tail_eos_tokens,
            "discarded_incomplete_row_tokens": self.discarded_incomplete_row_tokens,
            "discarded_incomplete_row_eos_tokens": (
                self.discarded_incomplete_row_eos_tokens
            ),
            "padding_tokens": 0,
            "output_rows": self.output_rows,
            "output_capacity_tokens": self.output_rows * self.capacity,
            "output_eos_tokens": self.output_eos_tokens,
            "compactions": self.compactions,
        }


def static_lane_expected(plan: dict, lane: dict) -> dict:
    return {
        "complete": True,
        "algorithm": plan["algorithm"],
        "lane_id": lane["lane_id"],
        "num_lanes": plan["num_lanes"],
        "source_path": lane["source_path"],
        "source_token_budget": lane["source_token_budget"],
        "capacity": plan["capacity"],
        "buffer_size": plan["buffer_size"],
        "dtype": plan["dtype"],
        "add_bos_during_packing": False,
        "add_eos_during_packing": False,
        "padding_tokens": 0,
    }


def validate_lane_output(output: Path, stats_path: Path, expected: dict) -> dict:
    if not output.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Incomplete lane output: {output}")
    stats = json.loads(stats_path.read_text())
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(
                f"{stats_path}: {key}={stats.get(key)!r}, expected {value!r}"
            )
    expected_bytes = int(stats["output_rows"]) * int(expected["capacity"]) * 2
    if output.stat().st_size != expected_bytes:
        raise ValueError(
            f"{output}: size={output.stat().st_size}, expected={expected_bytes}"
        )
    return stats


def make_packer(plan: dict, lane: dict) -> tuple[PrefixDocumentStream, BestFitPacker]:
    stream = PrefixDocumentStream(
        Path(lane["source_path"]),
        int(lane["source_token_budget"]),
        int(plan["source_eos_token_id"]),
    )
    packer = BestFitPacker(
        iter(stream),
        capacity=int(plan["capacity"]),
        buffer_size=int(plan["buffer_size"]),
        eos_id=int(plan["source_eos_token_id"]),
    )
    return stream, packer


def pack_lane(args: argparse.Namespace) -> dict:
    plan = load_plan(args)
    if args.lane_id < 0 or args.lane_id >= plan["num_lanes"]:
        raise IndexError(f"lane {args.lane_id} outside [0, {plan['num_lanes']})")
    lane = plan["lanes"][args.lane_id]
    output = Path(lane["output_path"])
    stats_path = lane_stats_path(args.output_root, args.lane_id, plan["num_lanes"])
    expected = static_lane_expected(plan, lane)
    if output.exists() and stats_path.exists() and not args.force:
        stats = validate_lane_output(output, stats_path, expected)
        print(f"SKIP valid lane: {output}", flush=True)
        return stats

    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    stats_path.unlink(missing_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getenv('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    temporary.unlink(missing_ok=True)

    stream, packer = make_packer(plan, lane)
    write_buffer = np.empty((args.write_batch_rows, args.capacity), dtype=PACK_DTYPE)
    buffered_rows = 0
    started = time.time()
    last_report = started
    try:
        with temporary.open("wb", buffering=32 * 1024 * 1024) as handle:
            for row in packer:
                write_buffer[buffered_rows] = row
                buffered_rows += 1
                if buffered_rows == args.write_batch_rows:
                    payload = write_buffer.tobytes(order="C")
                    handle.write(payload)
                    buffered_rows = 0
                now = time.time()
                if now - last_report >= 30:
                    elapsed = max(now - started, 1e-9)
                    print(
                        f"lane={args.lane_id}/{plan['num_lanes']} "
                        f"source={packer.source_tokens:,}/"
                        f"{lane['source_token_budget']:,} rows={packer.output_rows:,} "
                        f"output_tokens/s="
                        f"{packer.output_rows * args.capacity / elapsed:,.0f}",
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
    os.replace(temporary, output)

    stats = packer.stats()
    if stats["source_tokens"] != stream.tokens:
        raise ValueError("Packer and stream source-token counts differ")
    if stats["source_documents"] != stream.documents:
        raise ValueError("Packer and stream source-document counts differ")
    accounted_tokens = (
        stats["output_capacity_tokens"]
        + stats["discarded_tail_tokens"]
        + stats["discarded_incomplete_row_tokens"]
    )
    if accounted_tokens != stats["source_tokens"]:
        raise ValueError(
            f"Token conservation failed: {accounted_tokens:,} != "
            f"{stats['source_tokens']:,}"
        )
    accounted_eos = (
        stats["output_eos_tokens"]
        + stats["discarded_tail_eos_tokens"]
        + stats["discarded_incomplete_row_eos_tokens"]
    )
    if accounted_eos != stats["source_eos_tokens"]:
        raise ValueError(
            f"EOS conservation failed: {accounted_eos:,} != "
            f"{stats['source_eos_tokens']:,}"
        )

    elapsed = time.time() - started
    payload = {
        **expected,
        **stats,
        "source_row_groups_read": stream.row_groups,
        "source_budget_shortfall_tokens": stream.budget_shortfall_tokens,
        "boundary_document_tokens": stream.boundary_document_tokens,
        "source_tokens_dropped_by_bestfit": (
            stats["discarded_tail_tokens"] + stats["discarded_incomplete_row_tokens"]
        ),
        "packing_efficiency": (
            stats["output_capacity_tokens"] / max(stats["source_tokens"], 1)
        ),
        "output_path": str(output.resolve()),
        "output_size_bytes": output.stat().st_size,
        "row_bytes": args.capacity * PACK_DTYPE.itemsize,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": (
            stats["output_capacity_tokens"] / max(elapsed, 1e-9)
        ),
        "crop_tail_when_no_document_fits": True,
        "drop_final_incomplete_row": True,
    }
    write_json(stats_path, payload)
    validate_lane_output(output, stats_path, expected)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def replay_lane(plan: dict, lane: dict, rows: int) -> None:
    output = Path(lane["output_path"])
    stats = json.loads(
        lane_stats_path(
            Path(plan["output_root"]), lane["lane_id"], plan["num_lanes"]
        ).read_text()
    )
    compare_rows = min(rows, int(stats["output_rows"]))
    if not compare_rows:
        return
    memmap = np.memmap(
        output,
        mode="r",
        dtype=PACK_DTYPE,
        shape=(int(stats["output_rows"]), int(plan["capacity"])),
    )
    _, packer = make_packer(plan, lane)
    for row_index, expected_row in enumerate(islice(iter(packer), compare_rows)):
        if not np.array_equal(memmap[row_index], expected_row):
            mismatch = int(np.flatnonzero(memmap[row_index] != expected_row)[0])
            raise ValueError(
                f"Replay mismatch lane {lane['lane_id']} row {row_index} token {mismatch}"
            )
    del memmap


def finalize(args: argparse.Namespace) -> dict:
    plan = load_plan(args)
    stats_list = []
    for lane in plan["lanes"]:
        output = Path(lane["output_path"])
        stats_path = lane_stats_path(
            args.output_root, lane["lane_id"], plan["num_lanes"]
        )
        stats = validate_lane_output(
            output, stats_path, static_lane_expected(plan, lane)
        )
        replay_lane(plan, lane, args.replay_rows)
        stats_list.append(stats)
        print(
            f"validated lane={lane['lane_id']} rows={int(stats['output_rows']):,}",
            flush=True,
        )

    aggregate_keys = (
        "source_token_budget",
        "source_budget_shortfall_tokens",
        "source_documents",
        "source_tokens",
        "source_eos_tokens",
        "selected_documents",
        "long_documents_gt_capacity",
        "cropped_documents",
        "discarded_tail_tokens",
        "discarded_tail_eos_tokens",
        "discarded_incomplete_row_tokens",
        "discarded_incomplete_row_eos_tokens",
        "source_tokens_dropped_by_bestfit",
        "padding_tokens",
        "output_rows",
        "output_capacity_tokens",
        "output_eos_tokens",
        "output_size_bytes",
    )
    totals = {
        key: sum(int(stats[key]) for stats in stats_list) for key in aggregate_keys
    }
    if totals["source_token_budget"] != int(plan["target_source_tokens"]):
        raise ValueError("Aggregated source-token budget differs from plan")
    if (
        totals["source_tokens"] + totals["source_budget_shortfall_tokens"]
        != totals["source_token_budget"]
    ):
        raise ValueError("Source-token budget conservation failed")
    if totals["padding_tokens"]:
        raise ValueError("BestFit output unexpectedly contains padding")

    manifest = {
        "complete": args.write_success,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "algorithm": plan["algorithm"],
        "source_root": plan["source_root"],
        "source_special_tokens_policy": plan["source_special_tokens_policy"],
        "source_eos_token_id": plan["source_eos_token_id"],
        "reference_manifest": plan["reference_manifest"],
        "reference_algorithm": plan["reference_algorithm"],
        "output_root": plan["output_root"],
        "capacity": plan["capacity"],
        "training_sequence_length": plan["training_sequence_length"],
        "row_layout": f"input_ids[{plan['capacity']}]",
        "dtype": plan["dtype"],
        "buffer_size": plan["buffer_size"],
        "num_lanes": plan["num_lanes"],
        "target_source_tokens": plan["target_source_tokens"],
        "source_tokens_accepted": totals["source_tokens"],
        "source_budget_shortfall_tokens": totals["source_budget_shortfall_tokens"],
        "output_rows": totals["output_rows"],
        "output_capacity_tokens": totals["output_capacity_tokens"],
        "output_size_bytes": totals["output_size_bytes"],
        "source_tokens_dropped_by_bestfit": totals["source_tokens_dropped_by_bestfit"],
        "packing_efficiency": (
            totals["output_capacity_tokens"] / max(totals["source_tokens"], 1)
        ),
        "add_bos_during_packing": False,
        "add_eos_during_packing": False,
        "document_separator": "EOS already present at every source document end",
        "padding_tokens": 0,
        "crop_tail_when_no_document_fits": True,
        "drop_final_incomplete_row": True,
        "replay_rows_per_lane": args.replay_rows,
        "packing_plan": str(plan_path(args.output_root).resolve()),
        "totals": totals,
        "lane_stats": stats_list,
    }
    write_json(args.output_root / "manifest.json", manifest)
    if args.write_success:
        write_json(
            args.output_root / "_SUCCESS",
            {
                "manifest": "manifest.json",
                "output_rows": totals["output_rows"],
                "output_capacity_tokens": totals["output_capacity_tokens"],
            },
        )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def self_test() -> None:
    documents = [
        np.asarray([11, 12, 0], dtype=PACK_DTYPE),
        np.asarray([21, 22, 23, 24, 0], dtype=PACK_DTYPE),
        np.asarray([31, 32, 33, 34, 35, 36, 37, 0], dtype=PACK_DTYPE),
        np.asarray([41, 42, 43, 44, 45, 46, 47, 48, 49, 0], dtype=PACK_DTYPE),
        np.asarray([51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0], dtype=PACK_DTYPE),
    ]
    first = BestFitPacker(iter(documents), capacity=9, buffer_size=3, eos_id=0)
    rows = list(first)
    stats = first.stats()
    assert rows and all(row.shape == (9,) for row in rows)
    assert stats["source_tokens"] == sum(len(document) for document in documents)
    assert (
        stats["output_capacity_tokens"]
        + stats["discarded_tail_tokens"]
        + stats["discarded_incomplete_row_tokens"]
        == stats["source_tokens"]
    )
    second = BestFitPacker(iter(documents), capacity=9, buffer_size=3, eos_id=0)
    replay = list(second)
    assert len(rows) == len(replay)
    assert all(np.array_equal(left, right) for left, right in zip(rows, replay))
    assert stats["padding_tokens"] == 0
    print(json.dumps(stats, indent=2, sort_keys=True))
    print("SELF TEST PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--capacity", type=int, default=4097)
    parser.add_argument("--buffer-size", type=int, default=1000)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("plan-count")
    pack = commands.add_parser("pack-lane")
    pack.add_argument("--lane-id", type=int, required=True)
    pack.add_argument("--write-batch-rows", type=int, default=512)
    pack.add_argument("--force", action="store_true")
    finish = commands.add_parser("finalize")
    finish.add_argument("--replay-rows", type=int, default=16)
    finish.add_argument("--write-success", action="store_true")
    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command != "self-test":
        missing = [
            name
            for name in ("source_root", "output_root", "reference_manifest")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("required outside self-test: " + ", ".join(missing))
    return args


def main() -> None:
    args = parse_args()
    if args.capacity < 2:
        raise ValueError("capacity must be at least 2")
    if args.buffer_size < 1:
        raise ValueError("buffer-size must be positive")
    if args.command == "plan":
        build_plan(args)
    elif args.command == "plan-count":
        print(load_plan(args)["num_lanes"])
    elif args.command == "pack-lane":
        pack_lane(args)
    elif args.command == "finalize":
        finalize(args)
    elif args.command == "self-test":
        self_test()
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
