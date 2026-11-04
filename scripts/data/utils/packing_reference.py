"""Reference packers for offline preparation parity checks, not training loaders."""
import threading
from bisect import bisect_right, insort
from collections import deque
from typing import Iterator, Optional, List
import torch
from torch.utils.data import IterableDataset
from loopedlm.tokenizer import Tokenizer

class BestFitPackingCollator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int,
        add_bos: bool = True,
        add_eos: bool = True,
        buffer_size: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.buffer_size = buffer_size
        self.doc_buffer: List[torch.Tensor] = []
        self.row_capacity = block_size

    def state_dict(self) -> dict:
        return {
            "doc_buffer": [doc.tolist() for doc in self.doc_buffer],
            "num_docs": len(self.doc_buffer),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict is None:
            return
        doc_buffer_lists = state_dict.get("doc_buffer", [])
        self.doc_buffer = [torch.tensor(doc, dtype=torch.long) for doc in doc_buffer_lists]

    def __call__(self, batch) -> tuple[torch.Tensor, torch.Tensor, list]:
        for row in batch:
            tokens = self._tokenize_row(row)
            if tokens is not None and len(tokens) > 0:
                self.doc_buffer.append(tokens)

        batch_size = len(batch)
        row_buffer = torch.empty((batch_size, self.row_capacity), dtype=torch.long)

        for row_idx in range(batch_size):
            pos = 0
            while pos < self.row_capacity:
                if not self.doc_buffer:
                    row_buffer[row_idx, pos:] = self.tokenizer.pad_id or 0
                    break

                remaining = self.row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(self.doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = self.doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(self.doc_buffer)), key=lambda i: len(self.doc_buffer[i]))
                    doc = self.doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        input_ids = row_buffer[:, :-1].contiguous()
        label_ids = row_buffer[:, 1:].contiguous()

        metadata = [None] * batch_size
        return input_ids, label_ids, metadata

    def _tokenize_row(self, row) -> Optional[torch.Tensor]:
        if isinstance(row, torch.Tensor):
            return row

        if isinstance(row, dict):
            data_sig = row.get("data_signature", {"format_fn": "pass_text", "keys": ["text"]})
            format_fn_name = data_sig.get("format_fn", "pass_text")

            local_add_bos = data_sig.get("add_bos", self.add_bos)
            local_add_eos = data_sig.get("add_eos", self.add_eos)

            if format_fn_name == "pass_text":
                text = row.get("text", "")
                if text:
                    return self.tokenizer.encode(text, bos=local_add_bos, eos=local_add_eos)
            elif format_fn_name == "concat_input_target":
                text = row.get("input", "") + row.get("target", "")
                if text:
                    return self.tokenizer.encode(text, bos=local_add_bos, eos=local_add_eos)
            else:
                text = row.get("text", "")
                if text:
                    return self.tokenizer.encode(text, bos=local_add_bos, eos=local_add_eos)

        return None


class NanochatBestFitPackingDataset(IterableDataset):
    """BOS-aligned best-fit packing with a continuously refilled document buffer."""

    requires_resume_batch_skip = False

    def __init__(
        self,
        source_dataset: IterableDataset,
        tokenizer: Tokenizer,
        block_size: int,
        buffer_size: int = 1000,
        stateful: bool = True,
        use_length_index: bool = False,
        prefetch_rows: int = 0,
    ):
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be at least 1, got {buffer_size}")
        if prefetch_rows < 0:
            raise ValueError(f"prefetch_rows must be non-negative, got {prefetch_rows}")
        self.source_dataset = source_dataset
        self.tokenizer = tokenizer
        self.row_capacity = block_size
        self.buffer_size = buffer_size
        self.stateful = stateful
        self.use_length_index = use_length_index
        self.prefetch_rows = prefetch_rows
        self.doc_buffer: List[Optional[torch.Tensor]] = []
        self._active_documents = 0
        self._indices_by_length: dict[int, deque[int]] = {}
        self._sorted_lengths: list[int] = []
        self.source_exhausted = False
        self.bos_id = tokenizer.get_bos_token_id()
        self._prefetch_condition = threading.Condition()
        self._prefetched_rows: deque[torch.Tensor] = deque()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_done = False
        self._prefetch_error: Optional[BaseException] = None
        self._prefetch_stop = False

    @property
    def _state(self):
        return getattr(self.source_dataset, "_state", {})

    def _prepare_document(self, row) -> Optional[torch.Tensor]:
        if isinstance(row, torch.Tensor):
            tokens = row.to(dtype=torch.long).flatten()
        elif isinstance(row, dict):
            data_signature = row.get(
                "data_signature", {"format_fn": "pass_text", "keys": ["text"]}
            )
            format_fn_name = data_signature.get("format_fn", "pass_text")
            if format_fn_name == "concat_input_target":
                text = row.get("input", "") + row.get("target", "")
            else:
                text = row.get("text", "")
            if not text:
                return None
            tokens = self.tokenizer.encode(text, bos=False, eos=False)
        else:
            return None

        bos = torch.tensor([self.bos_id], dtype=torch.long)
        return torch.cat((bos, tokens))

    def _document_buffer_size(self) -> int:
        return self._active_documents if self.use_length_index else len(self.doc_buffer)

    def _buffered_documents(self) -> list[torch.Tensor]:
        return [document for document in self.doc_buffer if document is not None]

    def _rebuild_length_index(self) -> None:
        documents = self._buffered_documents()
        self.doc_buffer = list(documents)
        self._active_documents = len(documents)
        self._indices_by_length = {}
        self._sorted_lengths = []
        for index, document in enumerate(documents):
            document_length = len(document)
            if document_length not in self._indices_by_length:
                self._indices_by_length[document_length] = deque()
                insort(self._sorted_lengths, document_length)
            self._indices_by_length[document_length].append(index)

    def _set_document_buffer(self, documents: list[torch.Tensor]) -> None:
        self.doc_buffer = list(documents)
        if self.use_length_index:
            self._rebuild_length_index()

    def _append_document(self, document: torch.Tensor) -> None:
        self.doc_buffer.append(document)
        if not self.use_length_index:
            return
        index = len(self.doc_buffer) - 1
        document_length = len(document)
        if document_length not in self._indices_by_length:
            self._indices_by_length[document_length] = deque()
            insort(self._sorted_lengths, document_length)
        self._indices_by_length[document_length].append(index)
        self._active_documents += 1

    def _maybe_compact_document_buffer(self) -> None:
        if (
            self.use_length_index
            and len(self.doc_buffer) > max(2 * self.buffer_size, 2 * self._active_documents)
        ):
            self._rebuild_length_index()

    def _refill_buffer(self, source_iterator: Iterator) -> None:
        self._maybe_compact_document_buffer()
        while self._document_buffer_size() < self.buffer_size and not self.source_exhausted:
            try:
                row = next(source_iterator)
            except StopIteration:
                self.source_exhausted = True
                break
            document = self._prepare_document(row)
            if document is not None and document.numel() > 0:
                self._append_document(document)

    def _pop_best_fit(self, remaining: int) -> torch.Tensor:
        if self.use_length_index:
            length_position = bisect_right(self._sorted_lengths, remaining) - 1
            fits = length_position >= 0
            if not fits:
                length_position = 0
            document_length = self._sorted_lengths[length_position]
            indices = self._indices_by_length[document_length]
            document_index = indices.popleft()
            if not indices:
                del self._indices_by_length[document_length]
                del self._sorted_lengths[length_position]
            document = self.doc_buffer[document_index]
            if document is None:
                raise RuntimeError("nanochat length index referenced a removed document")
            self.doc_buffer[document_index] = None
            self._active_documents -= 1
            return document if fits else document[:remaining]

        best_idx = -1
        best_len = 0
        for index, document in enumerate(self.doc_buffer):
            if document is None:
                continue
            document_len = len(document)
            if document_len <= remaining and document_len > best_len:
                best_idx = index
                best_len = document_len

        if best_idx >= 0:
            return self.doc_buffer.pop(best_idx)

        live_indices = [
            index for index, document in enumerate(self.doc_buffer) if document is not None
        ]
        shortest_idx = min(live_indices, key=lambda index: len(self.doc_buffer[index]))  # type: ignore[arg-type]
        document = self.doc_buffer.pop(shortest_idx)
        if document is None:
            raise RuntimeError("nanochat buffer unexpectedly contained an empty document")
        return document[:remaining]

    def _next_packed_row(self, source_iterator: Iterator) -> torch.Tensor:
        while True:
            row = torch.empty(self.row_capacity, dtype=torch.long)
            position = 0
            while position < self.row_capacity:
                self._refill_buffer(source_iterator)
                if self._document_buffer_size() == 0:
                    if position == 0:
                        raise StopIteration
                    row[position:] = self.tokenizer.pad_id or 0
                    position = self.row_capacity
                    break

                remaining = self.row_capacity - position
                document = self._pop_best_fit(remaining)
                row[position : position + len(document)] = document
                position += len(document)

            return row

    def _prefetch_worker(self) -> None:
        source_iterator = iter(self.source_dataset)
        while True:
            with self._prefetch_condition:
                while (
                    len(self._prefetched_rows) >= self.prefetch_rows
                    and not self._prefetch_stop
                ):
                    self._prefetch_condition.wait()
                if self._prefetch_stop:
                    return
                try:
                    row = self._next_packed_row(source_iterator)
                except StopIteration:
                    self._prefetch_done = True
                    self._prefetch_condition.notify_all()
                    return
                except BaseException as exc:
                    self._prefetch_error = exc
                    self._prefetch_done = True
                    self._prefetch_condition.notify_all()
                    return
                self._prefetched_rows.append(row)
                self._prefetch_condition.notify_all()

    def _start_prefetch_thread(self) -> None:
        with self._prefetch_condition:
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return
            self._prefetch_stop = False
            self._prefetch_done = False
            self._prefetch_error = None
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                name="nanochat-packing-prefetch",
                daemon=True,
            )
            self._prefetch_thread.start()

    def _stop_prefetch_thread(self) -> None:
        with self._prefetch_condition:
            thread = self._prefetch_thread
            if thread is None:
                return
            self._prefetch_stop = True
            self._prefetch_condition.notify_all()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("nanochat packing prefetch thread did not stop")
        with self._prefetch_condition:
            self._prefetch_thread = None

    def __iter__(self):
        if not self.stateful:
            self._stop_prefetch_thread()
            self._set_document_buffer([])
            self.source_exhausted = False
            self._prefetched_rows.clear()

        if self.prefetch_rows == 0:
            source_iterator = iter(self.source_dataset)
            while True:
                try:
                    yield self._next_packed_row(source_iterator)
                except StopIteration:
                    return

        self._start_prefetch_thread()
        while True:
            with self._prefetch_condition:
                while (
                    not self._prefetched_rows
                    and not self._prefetch_done
                    and self._prefetch_error is None
                ):
                    self._prefetch_condition.wait()
                if self._prefetch_error is not None:
                    raise RuntimeError("nanochat packing prefetch failed") from self._prefetch_error
                if self._prefetched_rows:
                    row = self._prefetched_rows.popleft()
                    self._prefetch_condition.notify_all()
                elif self._prefetch_done:
                    return
                else:
                    continue
            yield row

    def state_dict(self) -> dict:
        with self._prefetch_condition:
            source_state = self.source_dataset.state_dict()
            local_state = {
                "doc_buffer": [
                    document.tolist() for document in self._buffered_documents()
                ],
                "source_exhausted": self.source_exhausted,
                "prefetched_rows": [
                    row.tolist() for row in self._prefetched_rows
                ],
            }
        if torch.distributed.is_initialized():
            rank_states = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rank_states, local_state)
        else:
            rank_states = [local_state]
        return {
            "version": 2,
            "source_dataset": source_state,
            "rank_states": rank_states,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._stop_prefetch_thread()
        self.source_dataset.load_state_dict(state_dict["source_dataset"])
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        rank_states = state_dict.get("rank_states", [])
        if not rank_states:
            self._set_document_buffer([])
            self.source_exhausted = False
            self._prefetched_rows.clear()
            return
        local_state = rank_states[rank % len(rank_states)]
        self._set_document_buffer(
            [
                torch.tensor(document, dtype=torch.long)
                for document in local_state.get("doc_buffer", [])
            ]
        )
        self.source_exhausted = bool(local_state.get("source_exhausted", False))
        self._prefetched_rows = deque(
            torch.tensor(row, dtype=torch.long)
            for row in local_state.get("prefetched_rows", [])
        )
        self._prefetch_done = False
        self._prefetch_error = None


def collate_packed_rows(
    batch: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list]:
    row_buffer = torch.stack(batch)
    input_ids = row_buffer[:, :-1].contiguous()
    label_ids = row_buffer[:, 1:].contiguous()
    return input_ids, label_ids, [None] * len(batch)
