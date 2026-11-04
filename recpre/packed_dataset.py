"""Stateful fixed-width packed binary reader for training and validation."""
import logging
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


class PackedBinaryDataset(IterableDataset):
    """Read fixed-width uint16 token rows produced by the offline BestFit packer."""

    # row_idx is advanced before each row is yielded, so a saved state already
    # points at the next unread row. The trainer must not discard another batch
    # after restoring this dataset.
    requires_resume_batch_skip = False

    def __init__(
        self,
        dataset_folder_path: str,
        *,
        block_size: int,
        seed: int = 12345,
        shuffle_filenames: bool = True,
        num_processes: int = 1,
        process_rank: int = 0,
        stateful: bool = True,
        plan_for_later_rank_expansion_to: int = 8,
    ):
        self.dataset_folder_path = Path(dataset_folder_path)
        self.block_size = int(block_size)
        self.process_rank = int(process_rank)
        self.stateful = stateful

        filenames = sorted(str(path) for path in self.dataset_folder_path.glob("*.bin"))
        if not filenames:
            raise FileNotFoundError(
                f"No packed binary files found under {self.dataset_folder_path}"
            )
        if shuffle_filenames:
            random.Random(seed).shuffle(filenames)

        self._row_offset = 0
        self._row_stride = 1
        if int(num_processes) > len(filenames):
            # More ranks than files: ranks sharing a file consume disjoint rows.
            # This is needed for the released validation split (8 lanes) on a
            # 16-rank job.
            file_slot = self.process_rank % len(filenames)
            self._row_offset = self.process_rank // len(filenames)
            self._row_stride = (
                (int(num_processes) - 1 - file_slot) // len(filenames)
            ) + 1
            self.primary_files = [filenames[file_slot]]
            self.overflow_files = []
        else:
            ranks_for_file_selection = max(
                int(plan_for_later_rank_expansion_to), int(num_processes)
            )
            self.primary_files = filenames[
                self.process_rank :: ranks_for_file_selection
            ]
            if not self.primary_files:
                raise ValueError(
                    f"Empty packed dataset on rank {self.process_rank}; "
                    f"found {len(filenames)} files for "
                    f"{ranks_for_file_selection} planned ranks"
                )

            # Preserve the old r8 data prefix, then consume virtual shards that
            # would otherwise be unused by a smaller physical world size.
            self.overflow_files = []
            for virtual_rank in range(
                self.process_rank + int(num_processes),
                ranks_for_file_selection,
                int(num_processes),
            ):
                self.overflow_files.extend(
                    filenames[virtual_rank::ranks_for_file_selection]
                )
        self.files = self.primary_files + self.overflow_files
        self.primary_file_count = len(self.primary_files)

        self.dtype = np.dtype("<u2")
        self.row_bytes = self.block_size * self.dtype.itemsize
        self.rows_per_file = []
        for filename in self.files:
            size = Path(filename).stat().st_size
            if size % self.row_bytes:
                raise ValueError(
                    f"{filename} has {size} bytes, which is not divisible by "
                    f"the packed row size {self.row_bytes}"
                )
            self.rows_per_file.append(size // self.row_bytes)

        self._state_init()
        logger.info(
            "Rank %s/%s has %s primary + %s overflow packed files and %s rows",
            self.process_rank,
            num_processes,
            len(self.primary_files),
            len(self.overflow_files),
            len(self),
        )

    def _state_init(self) -> None:
        self._state = {
            "file_idx": 0,
            "row_idx": self._row_offset,
        }

    def __iter__(self):
        if not self.stateful:
            self._state_init()

        while self._state["file_idx"] < len(self.files):
            file_idx = self._state["file_idx"]
            rows = self.rows_per_file[file_idx]
            if self._state["row_idx"] >= rows:
                self._state["file_idx"] += 1
                self._state["row_idx"] = self._row_offset
                continue

            mapped = np.memmap(
                self.files[file_idx],
                mode="c",
                dtype=self.dtype,
                shape=(rows, self.block_size),
            )
            while self._state["row_idx"] < rows:
                row_idx = self._state["row_idx"]
                self._state["row_idx"] += self._row_stride
                yield torch.from_numpy(mapped[row_idx]).long()
            del mapped

    def __len__(self) -> int:
        return sum(
            max(0, (rows - self._row_offset + self._row_stride - 1) // self._row_stride)
            for rows in self.rows_per_file
        )

    def state_dict(self) -> dict:
        local_state = torch.tensor(
            [
                self._state["file_idx"],
                self._state["row_idx"],
            ],
            device="cuda" if torch.distributed.is_initialized() else "cpu",
            dtype=torch.long,
        )
        if torch.distributed.is_initialized():
            gathered_states = [
                torch.zeros_like(local_state)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered_states, local_state)
        else:
            gathered_states = [local_state]
        return {
            "file_idx": [state[0].item() for state in gathered_states],
            "row_idx": [state[1].item() for state in gathered_states],
        }

    def load_state_dict(self, state_dict: dict, offset_ranks: bool = False) -> None:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else self.process_rank
        )
        effective_rank = rank % len(state_dict["file_idx"])

        self._state["file_idx"] = max(
            int(state_dict["file_idx"][effective_rank]), 0
        )
        self._state["row_idx"] = max(
            int(state_dict["row_idx"][effective_rank]), 0
        )
        if offset_ranks:
            self._state["row_idx"] += rank % 1000
