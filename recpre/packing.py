"""Shared uint16 best-fit packing for offline preparation and streaming training."""
from bisect import bisect_right, insort
from collections import deque
from typing import Iterator
import numpy as np

PACK_DTYPE = np.dtype("<u2")

class BestFitPacker:
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
        if capacity < 1 or buffer_size < 1:
            raise ValueError("capacity and buffer_size must be positive")
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


    def state_dict(self) -> dict:
        return {"documents": [doc.copy() for doc in self._buffered_documents()],
                "source_exhausted": self.source_exhausted,
                "counters": {key: getattr(self, key) for key in self.stats() if key != "output_capacity_tokens"}}

    def load_state_dict(self, state: dict) -> None:
        self.doc_buffer = [np.asarray(doc, dtype=PACK_DTYPE).copy() for doc in state["documents"]]
        self._rebuild_index()
        self.source_exhausted = state["source_exhausted"]
        for key, value in state["counters"].items():
            setattr(self, key, value)
