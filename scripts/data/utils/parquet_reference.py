"""Parquet reference reader used only to validate offline data preparation.

Derived from the original Parcae/Lightning data reader (Apache-2.0).
"""
import random
import logging
from pathlib import Path
import torch
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

class ParquetStreamPure(IterableDataset):
    """datasets-free version of (mostly) the same thing - shuffle not across files though
    a bit ironic to keep it in this file"""

    def __init__(
        self,
        dataset_folder_path="",
        seed=12345,
        shuffle=True,
        num_processes=1,
        process_rank=0,
        prefix="",
        verbose=False,
        shuffle_filenames=True,
        data_signature: dict[str, list[str] | str] = {"keys": ["input_ids"]},
        repetitions=None,
        return_data_id=False,
        data_id=None,
        broadcast_glob=True,
        stateful=True,  # this dataset is by default (suprisingly) stateful, even outside of iter(dataset)!
        plan_for_later_rank_expansion_to=8,
        tokenizer=None,  # Optional tokenizer for raw text data
        interleave_files=1,
        filename_seed=None,
    ):
        # Make rank visible for any early error handling
        self.process_rank = process_rank
        # Get file list, with distributed broadcast if needed
        if broadcast_glob and torch.distributed.is_initialized():
            if process_rank == 0:
                filenames = sorted(str(p) for p in Path(dataset_folder_path).rglob(f"{prefix}*.parquet"))
            else:
                filenames: list[str] = None  # type: ignore
            obj = [filenames]
            torch.distributed.broadcast_object_list(obj, 0)
            parquet_files = obj[0]
        else:
            parquet_files = sorted(str(p) for p in Path(dataset_folder_path).rglob(f"{prefix}*.parquet"))
        if shuffle_filenames:
            random.Random(seed if filename_seed is None else filename_seed).shuffle(parquet_files)

        # Shard files for distributed training
        if plan_for_later_rank_expansion_to > 0:
            ranks_for_file_selection = max(plan_for_later_rank_expansion_to, num_processes)
        else:
            ranks_for_file_selection = num_processes
        self.parquet_files = (
            parquet_files[process_rank::ranks_for_file_selection] if ranks_for_file_selection > 1 else parquet_files
        )
        if len(self.parquet_files) < 1:
            raise ValueError(f"Empty dataset on rank {self.process_rank}")

        # Store data signature for column name lookup
        self.data_signature = data_signature
        self.token_column = data_signature.get("keys", ["input_ids"])[0] if isinstance(data_signature.get("keys"), list) else data_signature.get("keys", "input_ids")
        self.tokenizer = tokenizer  # Store tokenizer for raw text data
        self.interleave_files = int(interleave_files)
        if self.interleave_files < 1:
            raise ValueError(f"interleave_files must be at least 1, got {self.interleave_files}")

        if verbose:
            logger.info(
                f"Rank {process_rank}/{num_processes} has {len(self.parquet_files)} parquet files"
            )
            examples = pq.read_table(self.parquet_files[0], columns=[self.token_column]).slice(0, 3).to_pylist()  # Get 3 rows
            for i, example in enumerate(examples):
                logger.info(f"Example {i}: {example[self.token_column][:12]}")  # First 12 tokens of each row
        self.shuffle = shuffle
        self.seed = seed
        self.process_rank = process_rank
        self.stateful = stateful
        # Initialize default state
        self._state_init()

    def _state_init(self):
        self._state = {
            "rng": random.Random(self.seed),
            "rng_state": (-1, [-1], None),
            "buffer": [],
            "file_idx": 0,
            "row_group_idx": 0,
            "interleave_origin_file_idx": 0 if self.interleave_files > 1 else -1,
            "interleave_origin_row_group_idx": 0 if self.interleave_files > 1 else -1,
            "interleave_plan_idx": 0,
        }
        self._interleave_plan = None

    def __iter__(self):
        if not self.stateful:
            self._state_init()

        if self.interleave_files > 1:
            yield from self._iter_interleaved()
            return

        while self._state["file_idx"] < len(self.parquet_files):
            if not self._state["buffer"]:
                pf = pq.ParquetFile(self.parquet_files[self._state["file_idx"]])
                if self._state["row_group_idx"] >= pf.num_row_groups:
                    self._state["file_idx"] += 1
                    self._state["row_group_idx"] = 0
                    continue

                self._read_buffer(pf)
                self._state["row_group_idx"] += 1

            while self._state["buffer"]:
                yield torch.as_tensor(self._state["buffer"].pop(), dtype=torch.long)

    def _install_buffer(self, raw_data) -> None:
        if self.tokenizer is not None and raw_data and isinstance(raw_data[0], str):
            self._state["buffer"] = [
                self.tokenizer.encode(text, return_tensors=False)
                for text in raw_data
            ]
        else:
            self._state["buffer"] = raw_data

        if self.shuffle:
            self._state["rng_state"] = self._state["rng"].getstate()
            self._state["rng"].shuffle(self._state["buffer"])

    def _build_interleave_plan(self):
        origin_file_idx = self._state["interleave_origin_file_idx"]
        origin_row_group_idx = self._state["interleave_origin_row_group_idx"]
        plan = []
        for window_start in range(origin_file_idx, len(self.parquet_files), self.interleave_files):
            window = list(range(window_start, min(window_start + self.interleave_files, len(self.parquet_files))))
            starts = {
                file_idx: origin_row_group_idx if file_idx == origin_file_idx else 0
                for file_idx in window
            }
            counts = {
                file_idx: pq.ParquetFile(self.parquet_files[file_idx]).num_row_groups
                for file_idx in window
            }
            rounds = max(counts[file_idx] - starts[file_idx] for file_idx in window)
            for offset in range(rounds):
                for file_idx in window:
                    row_group_idx = starts[file_idx] + offset
                    if row_group_idx < counts[file_idx]:
                        plan.append((file_idx, row_group_idx))
        self._interleave_plan = plan

    def _iter_interleaved(self):
        while self._state["buffer"]:
            yield torch.as_tensor(self._state["buffer"].pop(), dtype=torch.long)

        if self._interleave_plan is None:
            self._build_interleave_plan()

        while self._state["interleave_plan_idx"] < len(self._interleave_plan):
            file_idx, row_group_idx = self._interleave_plan[self._state["interleave_plan_idx"]]
            parquet_file = pq.ParquetFile(self.parquet_files[file_idx])
            self._state["file_idx"] = file_idx
            self._state["row_group_idx"] = row_group_idx
            self._read_buffer(parquet_file)
            self._state["row_group_idx"] = row_group_idx + 1
            self._state["interleave_plan_idx"] += 1
            while self._state["buffer"]:
                yield torch.as_tensor(self._state["buffer"].pop(), dtype=torch.long)

    def _read_buffer(self, parquet_file):
        batch = parquet_file.read_row_group(self._state["row_group_idx"])
        raw_data = batch.column(self.token_column).to_pylist()
        self._install_buffer(raw_data)

    def state_dict(self):
        # Pack all basic state and RNG state into one tensor for a single gather
        rng_0, rng_1, rng_2 = self._state["rng_state"]
        # Save row_group_idx - 1 because load_state_dict will read the buffer and then increment
        saved_row_group_idx = self._state["row_group_idx"] - 1
        buffer_remaining = len(self._state["buffer"])

        local_state = torch.tensor(
            [
                self._state["file_idx"],
                saved_row_group_idx,
                buffer_remaining,
                0,  # Reserved slot keeps the packed RNG-state layout unchanged.
                rng_0,
                *rng_1,
                rng_2 if rng_2 is not None else -1,
            ],
            device="cuda",
        )

        # Single gather for all state
        gathered_states = [torch.zeros_like(local_state) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_states, local_state)

        result = {
            "file_idx": [s[0].item() for s in gathered_states],
            "row_group_idx": [s[1].item() for s in gathered_states],
            "row_idx": [s[2].item() for s in gathered_states],
            "rng_state": gathered_states,  # Full tensors for unpacking in load
        }

        if self.interleave_files > 1:
            interleave_state = torch.tensor(
                [
                    self._state["interleave_origin_file_idx"],
                    self._state["interleave_origin_row_group_idx"],
                    self._state["interleave_plan_idx"],
                ],
                device="cuda",
            )
            gathered_interleave_states = [
                torch.zeros_like(interleave_state) for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered_interleave_states, interleave_state)
            result.update(
                {
                    "interleave_origin_file_idx": [s[0].item() for s in gathered_interleave_states],
                    "interleave_origin_row_group_idx": [s[1].item() for s in gathered_interleave_states],
                    "interleave_plan_idx": [s[2].item() for s in gathered_interleave_states],
                }
            )

        return result

    def load_state_dict(self, state_dict, offset_ranks=False):
        rank = torch.distributed.get_rank()

        def get_value(key):  # helper for backward compat
            effective_rank = rank % len(state_dict["file_idx"])
            return state_dict[key][effective_rank]

        self._state["file_idx"] = max(get_value("file_idx"), 0)
        self._state["row_group_idx"] = max(get_value("row_group_idx"), 0)
        row_idx = max(get_value("row_idx"), 0)
        if offset_ranks:
            row_idx += rank % 1000

        if state_dict["rng_state"] is not None:
            # New packed format
            rng_state = state_dict["rng_state"][rank % len(state_dict["rng_state"])]
            # RNG state starts at index 4
            rng_state = (rng_state[4].item(), tuple(x.item() for x in rng_state[5:-1]), rng_state[-1].item())
            if rng_state[2] == -1:
                rng_state = (rng_state[0], rng_state[1], None)

            self._state["rng"] = random.Random()
            self._state["rng"].setstate(rng_state)
            self._state["rng_state"] = rng_state

        saved_interleave = (
            self.interleave_files > 1
            and "interleave_origin_file_idx" in state_dict
            and get_value("interleave_origin_file_idx") >= 0
        )
        if saved_interleave:
            self._state["interleave_origin_file_idx"] = get_value("interleave_origin_file_idx")
            self._state["interleave_origin_row_group_idx"] = get_value("interleave_origin_row_group_idx")
            self._state["interleave_plan_idx"] = get_value("interleave_plan_idx")
            self._build_interleave_plan()
            if self._state["interleave_plan_idx"] > 0:
                file_idx, row_group_idx = self._interleave_plan[self._state["interleave_plan_idx"] - 1]
                self._state["file_idx"] = file_idx
                self._state["row_group_idx"] = row_group_idx
                pf = pq.ParquetFile(self.parquet_files[file_idx])
                self._read_buffer(pf)
                self._state["row_group_idx"] = row_group_idx + 1
                self._state["buffer"] = self._state["buffer"][:row_idx]
        else:
            pf = pq.ParquetFile(self.parquet_files[self._state["file_idx"]])
            self._read_buffer(pf)
            # Increment row_group_idx AFTER reading, matching what __iter__ does
            # This prevents re-reading the same row_group when iteration resumes
            self._state["row_group_idx"] += 1
            # Truncate buffer to restore position within the row_group
            self._state["buffer"] = self._state["buffer"][:row_idx]
            if self.interleave_files > 1:
                self._state["interleave_origin_file_idx"] = self._state["file_idx"]
                self._state["interleave_origin_row_group_idx"] = self._state["row_group_idx"]
                self._state["interleave_plan_idx"] = 0
                self._interleave_plan = None
