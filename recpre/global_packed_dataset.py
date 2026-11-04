"""Global random access to fixed-width offline-packed binary sequences."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import fcntl

import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

GLOBAL_PACKED_READER_VERSION = "global-packed-catalog-epoch-chunk-permutation-v6"
PHYSICAL_CATALOG_VERSION = "global-packed-physical-catalog-v1"
SEQUENCE_SHUFFLE_VERSION = "pcg64-epoch-chunk-permutation-v2"


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize cache creation across ranks on a shared filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class GlobalPackedBinaryDataset(Dataset):
    """Expose every packed row through one deterministic global ordinal space.

    The catalog contains every ``.bin`` file in the source. Files are mmap'd
    lazily and retained until the dataset is closed. Retaining mappings is
    important for shuffled access across many files: mmap does not read the
    complete payload into memory, while repeatedly mapping multi-GB files is
    prohibitively expensive.
    When sequence shuffling is disabled, indices repeat globally with
    ``source_ordinal % len(dataset)``. When enabled, every logical source epoch
    has a different deterministic permutation from logical positions to
    physical packed rows. The permutation is cached as a shared mmap-able
    ``.npy`` file; packed payload files are never rewritten.
    """

    def __init__(
        self,
        dataset_folder_path: str | Path,
        *,
        block_size: int,
        seed: int = 12345,
        shuffle_filenames: bool = True,
        shuffle_sequences: bool = False,
        sequence_shuffle_chunk_size: int = 1,
        permutation_cache_root: str | Path | None = None,
    ) -> None:
        self.dataset_folder_path = Path(dataset_folder_path).resolve()
        self.block_size = int(block_size)
        self.dtype = np.dtype("<u2")
        self.row_bytes = self.block_size * self.dtype.itemsize

        manifest_path = self.dataset_folder_path.parent / "manifest.json"
        success_path = self.dataset_folder_path.parent / "_SUCCESS"
        if not manifest_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(
                "Offline packed dataset is incomplete under "
                f"{self.dataset_folder_path.parent}"
            )
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        if not manifest.get("complete"):
            raise ValueError(f"Packed dataset manifest is incomplete: {manifest_path}")
        if manifest.get("dtype") != "uint16-le":
            raise ValueError(
                f"Unsupported packed dtype {manifest.get('dtype')!r}; "
                "expected 'uint16-le'"
            )
        if int(manifest.get("capacity", -1)) != self.block_size:
            raise ValueError(
                f"Packed row capacity is {manifest.get('capacity')}, "
                f"but loader block size is {self.block_size}"
            )

        entries = self._catalog_entries(manifest)
        if shuffle_filenames:
            random.Random(seed).shuffle(entries)
        self.files = [str(path) for path, _ in entries]
        self.rows_per_file = [int(rows) for _, rows in entries]
        self.cumulative_rows = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(self.rows_per_file, dtype=np.int64),
            )
        )
        self.total_rows = int(self.cumulative_rows[-1])
        if self.total_rows <= 0:
            raise ValueError(
                f"Global packed dataset has no rows: {self.dataset_folder_path}"
            )

        tokenizer_sha256 = manifest.get("source_tokenizer_sha256") or manifest.get(
            "tokenizer_sha256"
        )
        physical_descriptor: dict[str, Any] = {
            "version": PHYSICAL_CATALOG_VERSION,
            "data_dir": str(self.dataset_folder_path),
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "capacity": self.block_size,
            "dtype": "uint16-le",
            "tokenizer_sha256": tokenizer_sha256,
            "shuffle_filenames": bool(shuffle_filenames),
            "seed": int(seed) if shuffle_filenames else None,
            "files": [
                {"name": Path(filename).name, "rows": rows}
                for filename, rows in zip(self.files, self.rows_per_file)
            ],
            "total_rows": self.total_rows,
        }
        physical_serialized = json.dumps(
            physical_descriptor, sort_keys=True, separators=(",", ":")
        )
        self.physical_catalog_fingerprint = hashlib.sha256(
            physical_serialized.encode()
        ).hexdigest()

        self.seed = int(seed)
        self.shuffle_sequences = bool(shuffle_sequences)
        self.requested_sequence_shuffle_chunk_size = int(
            sequence_shuffle_chunk_size
        )
        if self.requested_sequence_shuffle_chunk_size <= 0:
            raise ValueError("sequence_shuffle_chunk_size must be positive")
        self.sequence_shuffle_chunk_size = self.requested_sequence_shuffle_chunk_size
        if (
            self.shuffle_sequences
            and self.total_rows // self.sequence_shuffle_chunk_size < 2
        ):
            # A one-chunk permutation is the identity. Tiny sources are cheap
            # enough to shuffle row-by-row, so retain per-epoch randomness for
            # them instead of silently reading the same order every epoch.
            self.sequence_shuffle_chunk_size = 1
        if self.shuffle_sequences and permutation_cache_root is None:
            raise ValueError(
                "permutation_cache_root is required when shuffle_sequences=true"
            )
        # Shuffle complete fixed-size chunks and leave the final partial chunk
        # in place. This keeps every access within a chunk sequential while
        # preserving every packed row exactly once per source epoch.
        self.num_shuffle_chunks = self.total_rows // self.sequence_shuffle_chunk_size
        self.shuffled_rows = (
            self.num_shuffle_chunks * self.sequence_shuffle_chunk_size
        )
        self.permutation_dtype = np.dtype(
            "<u4"
            if self.num_shuffle_chunks <= np.iinfo(np.uint32).max + 1
            else "<u8"
        )
        self.permutation_cache_dir = (
            Path(permutation_cache_root)
            / self.physical_catalog_fingerprint
            / f"seed-{self.seed}"
            / f"chunk-{self.sequence_shuffle_chunk_size:08d}"
            if permutation_cache_root is not None
            else None
        )

        descriptor = dict(physical_descriptor)
        descriptor.update(
            {
                "reader_version": GLOBAL_PACKED_READER_VERSION,
                "physical_catalog_fingerprint": self.physical_catalog_fingerprint,
                "shuffle_sequences": self.shuffle_sequences,
                "sequence_shuffle_version": (
                    SEQUENCE_SHUFFLE_VERSION if self.shuffle_sequences else None
                ),
                "sequence_shuffle_seed": self.seed if self.shuffle_sequences else None,
                "sequence_shuffle_chunk_size": (
                    self.sequence_shuffle_chunk_size
                    if self.shuffle_sequences
                    else None
                ),
                "requested_sequence_shuffle_chunk_size": (
                    self.requested_sequence_shuffle_chunk_size
                    if self.shuffle_sequences
                    else None
                ),
                "num_shuffle_chunks": (
                    self.num_shuffle_chunks if self.shuffle_sequences else None
                ),
                "unshuffled_tail_rows": (
                    self.total_rows - self.shuffled_rows
                    if self.shuffle_sequences
                    else None
                ),
                "permutation_dtype": (
                    self.permutation_dtype.str if self.shuffle_sequences else None
                ),
            }
        )
        serialized = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        self.catalog_fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
        self.catalog_description = descriptor

        self._mapped_file_idx: int | None = None
        self._mapped: np.memmap | None = None
        self._mapped_files: dict[int, np.memmap] = {}
        self._permutation_epoch: int | None = None
        self._permutation: np.ndarray | None = None

        logger.info(
            "Global packed source has %s files and %s rows | "
            "sequence_shuffle=%s chunk_size=%s requested_chunk_size=%s catalog=%s",
            len(self.files),
            self.total_rows,
            self.shuffle_sequences,
            self.sequence_shuffle_chunk_size,
            self.requested_sequence_shuffle_chunk_size,
            self.catalog_fingerprint,
        )

    def _permutation_seed(self, epoch: int) -> int:
        payload = (
            f"{SEQUENCE_SHUFFLE_VERSION}\0{self.seed}\0"
            f"{self.physical_catalog_fingerprint}\0"
            f"{self.sequence_shuffle_chunk_size}\0{epoch}"
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")

    def _permutation_path(self, epoch: int) -> Path:
        if self.permutation_cache_dir is None:
            raise RuntimeError("Sequence permutation cache is not configured")
        return self.permutation_cache_dir / f"epoch-{epoch:08d}.npy"

    def _load_valid_permutation(self, path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        try:
            permutation = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            return None
        if (
            permutation.shape != (self.num_shuffle_chunks,)
            or permutation.dtype != self.permutation_dtype
        ):
            del permutation
            return None
        return permutation

    def _initialize_permutation_cache(self) -> None:
        if self.permutation_cache_dir is None:
            raise RuntimeError("Sequence permutation cache is not configured")
        description_path = self.permutation_cache_dir / "description.json"
        success_path = self.permutation_cache_dir / "_SUCCESS"
        lock_path = self.permutation_cache_dir / ".metadata.lock"
        with _exclusive_file_lock(lock_path):
            expected = {
                "version": SEQUENCE_SHUFFLE_VERSION,
                "physical_catalog_fingerprint": self.physical_catalog_fingerprint,
                "seed": self.seed,
                "total_rows": self.total_rows,
                "chunk_size": self.sequence_shuffle_chunk_size,
                "num_shuffle_chunks": self.num_shuffle_chunks,
                "unshuffled_tail_rows": self.total_rows - self.shuffled_rows,
                "dtype": self.permutation_dtype.str,
            }
            if description_path.is_file() and success_path.is_file():
                try:
                    if json.loads(description_path.read_text()) == expected:
                        return
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
                raise ValueError(
                    "Packed-sequence permutation cache metadata does not match "
                    f"the requested source: {self.permutation_cache_dir}"
                )

            self.permutation_cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="description.",
                suffix=".tmp",
                dir=self.permutation_cache_dir,
                delete=False,
            ) as temp_file:
                temp_description = Path(temp_file.name)
                json.dump(expected, temp_file, sort_keys=True, indent=2)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_description, description_path)
            success_path.touch()

    def _build_or_load_permutation(self, epoch: int) -> np.ndarray:
        if epoch < 0:
            raise IndexError(f"Negative packed source epoch: {epoch}")
        self._initialize_permutation_cache()
        target = self._permutation_path(epoch)
        permutation = self._load_valid_permutation(target)
        if permutation is not None:
            return permutation

        lock_path = target.with_suffix(target.suffix + ".lock")
        with _exclusive_file_lock(lock_path):
            permutation = self._load_valid_permutation(target)
            if permutation is not None:
                return permutation

            logger.info(
                "Building packed-sequence chunk permutation epoch=%s "
                "chunks=%s chunk_size=%s tail_rows=%s cache=%s",
                epoch,
                self.num_shuffle_chunks,
                self.sequence_shuffle_chunk_size,
                self.total_rows - self.shuffled_rows,
                target,
            )
            values = np.arange(self.num_shuffle_chunks, dtype=self.permutation_dtype)
            rng = np.random.Generator(np.random.PCG64(self._permutation_seed(epoch)))
            rng.shuffle(values)
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f"{target.stem}.",
                    suffix=".tmp",
                    dir=target.parent,
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    np.save(temp_file, values, allow_pickle=False)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                os.replace(temp_path, target)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
            del values

        permutation = self._load_valid_permutation(target)
        if permutation is None:
            raise RuntimeError(f"Failed to create packed-sequence permutation: {target}")
        return permutation

    def _physical_row(self, source_ordinal: int) -> int:
        epoch, logical_row = divmod(source_ordinal, self.total_rows)
        if not self.shuffle_sequences:
            return logical_row
        if logical_row >= self.shuffled_rows:
            return logical_row
        if self._permutation_epoch != epoch or self._permutation is None:
            if self._permutation is not None:
                del self._permutation
            self._permutation = self._build_or_load_permutation(epoch)
            self._permutation_epoch = epoch
        logical_chunk, within_chunk = divmod(
            logical_row, self.sequence_shuffle_chunk_size
        )
        physical_chunk = int(self._permutation[logical_chunk])
        return physical_chunk * self.sequence_shuffle_chunk_size + within_chunk

    def _catalog_entries(self, manifest: dict[str, Any]) -> list[tuple[Path, int]]:
        lane_stats = manifest.get("lane_stats")
        if isinstance(lane_stats, list) and lane_stats:
            by_name: dict[str, int] = {}
            usable = True
            for stat in lane_stats:
                if not isinstance(stat, dict):
                    usable = False
                    break
                output_path = stat.get("output_path")
                output_rows = stat.get("output_rows")
                if output_path is None or output_rows is None:
                    usable = False
                    break
                name = Path(str(output_path)).name
                if name in by_name:
                    raise ValueError(
                        f"Duplicate packed filename {name!r} in manifest lane_stats"
                    )
                by_name[name] = int(output_rows)
            if usable:
                entries = [
                    (self.dataset_folder_path / name, rows)
                    for name, rows in sorted(by_name.items())
                ]
                if entries:
                    return entries

        filenames = sorted(self.dataset_folder_path.glob("*.bin"))
        if not filenames:
            raise FileNotFoundError(
                f"No packed binary files found under {self.dataset_folder_path}"
            )
        entries = []
        for path in filenames:
            size = path.stat().st_size
            if size % self.row_bytes:
                raise ValueError(
                    f"{path} has {size} bytes, which is not divisible by "
                    f"the packed row size {self.row_bytes}"
                )
            entries.append((path, size // self.row_bytes))
        return entries

    def validate_files(self) -> None:
        """Validate the canonical catalog against the shared filesystem."""

        for filename, rows in zip(self.files, self.rows_per_file):
            path = Path(filename)
            if not path.is_file():
                raise FileNotFoundError(f"Missing packed binary file: {path}")
            expected_size = int(rows) * self.row_bytes
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"Packed file size mismatch for {path}: "
                    f"catalog={expected_size}, actual={actual_size}"
                )

    def __len__(self) -> int:
        return self.total_rows

    def _map_file(self, file_idx: int) -> np.memmap:
        mapped = self._mapped_files.get(file_idx)
        if mapped is None:
            rows = self.rows_per_file[file_idx]
            mapped = np.memmap(
                self.files[file_idx],
                mode="c",
                dtype=self.dtype,
                shape=(rows, self.block_size),
            )
            self._mapped_files[file_idx] = mapped
        self._mapped = mapped
        self._mapped_file_idx = file_idx
        return mapped

    def __getitem__(self, source_ordinal: int) -> torch.Tensor:
        source_ordinal = int(source_ordinal)
        if source_ordinal < 0:
            raise IndexError(f"Negative global packed index: {source_ordinal}")
        actual_row = self._physical_row(source_ordinal)
        file_idx = bisect_right(self.cumulative_rows, actual_row) - 1
        file_local_row = actual_row - int(self.cumulative_rows[file_idx])
        mapped = self._map_file(file_idx)
        # Conversion to int64 owns its storage, so the returned tensor does not
        # retain a view into the cached mmap.
        return torch.from_numpy(mapped[file_local_row]).long()

    def close(self) -> None:
        if getattr(self, "_mapped_files", None):
            self._mapped_files.clear()
        self._mapped = None
        self._mapped_file_idx = None
        if getattr(self, "_permutation", None) is not None:
            del self._permutation
        self._permutation = None
        self._permutation_epoch = None

    def __del__(self) -> None:
        self.close()
