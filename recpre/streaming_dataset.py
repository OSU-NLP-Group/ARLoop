"""Online HF text datasets with the same packing rules as offline control data."""
import copy
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from recpre.packing import BestFitPacker, PACK_DTYPE

def streaming_source(dataset, *, dataset_config=None, dataset_split="train",
                     data_files=None, text_column="text", seed=1337, world_size=1, rank=0):
    from datasets import IterableDataset as HFIterableDataset, load_dataset_builder

    builder = load_dataset_builder(dataset, name=dataset_config, data_files=data_files)
    if builder.info.builder_name == "parquet":
        # Use synchronous reads: Arrow scanner prefetch can survive early training exit.
        files = sorted(str(path) for path in builder.config.data_files[dataset_split])
        random.Random(seed).shuffle(files)
        if not 0 <= rank < world_size <= len(files):
            raise ValueError("stream requires at least one source shard per data rank")
        return HFIterableDataset.from_generator(_read_parquet,
            gen_kwargs={"files": files[rank::world_size], "text_column": text_column})
    source = builder.as_streaming_dataset(split=dataset_split)
    if not 0 <= rank < world_size <= source.num_shards:
        raise ValueError("stream requires at least one source shard per data rank")
    source = source.shard(num_shards=world_size, index=rank)
    if text_column != "text":
        source = source.select_columns([text_column]).rename_column(text_column, "text")
    return source


def _read_parquet(files, text_column="text"):
    import fsspec
    import pyarrow.parquet as pq

    # Synchronous range reads avoid scanner prefetch surviving an early training exit.
    for path in files:
        with fsspec.open(path, "rb") as handle:
            with pq.ParquetFile(handle) as parquet:
                for batch in parquet.iter_batches(batch_size=1024, columns=[text_column], use_threads=False):
                    yield from ({"text": text} for text in batch.column(0).to_pylist())


class StreamingPackedDataset(IterableDataset):
    """One stateful iterator per rank; no worker processes or hidden prefetch queue."""

    requires_resume_batch_skip = False

    def __init__(self, source, tokenizer, *, block_size, buffer_size=1000,
                 shuffle_buffer_size=0, seed=1337, source_id=None, stateful=True):
        if tokenizer.vocab_size > 65536:
            raise ValueError("Online/offline control packing requires a uint16 vocabulary")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be nonnegative")
        self.source = source
        self.tokenizer = tokenizer
        self.shuffle_buffer_size = shuffle_buffer_size
        self.rng = random.Random(seed)
        self.shuffle_buffer = []
        self.source_exhausted = False
        self.seed = seed
        self.packer = BestFitPacker(iter(()), capacity=block_size, buffer_size=buffer_size,
            bos_id=int(tokenizer.get_bos_token_id()), pad_id=tokenizer.pad_id)
        self.initial_source_state = None if stateful else copy.deepcopy(source.state_dict())
        self.initial_packer_state = None if stateful else self.packer.state_dict()
        self.settings = (source_id, block_size, buffer_size, shuffle_buffer_size, seed,
                         str(tokenizer.checkpoint_dir), int(tokenizer.get_bos_token_id()), tokenizer.pad_id)

    def _documents(self):
        source = iter(self.source)
        try:
            yield from self._tokenized_documents(source)
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()

    def _tokenized_documents(self, source):
        while True:
            if not self.shuffle_buffer_size:
                for row in source:
                    yield np.asarray(self.tokenizer.encode(row["text"], return_tensors=False), dtype=PACK_DTYPE)
                return
            while len(self.shuffle_buffer) < self.shuffle_buffer_size and not self.source_exhausted:
                try:
                    self.shuffle_buffer.append(next(source)["text"])
                except StopIteration:
                    self.source_exhausted = True
            if not self.shuffle_buffer:
                return
            text = self.shuffle_buffer.pop(self.rng.randrange(len(self.shuffle_buffer)))
            yield np.asarray(self.tokenizer.encode(text, return_tensors=False), dtype=PACK_DTYPE)

    def __iter__(self):
        if get_worker_info() is not None:
            raise ValueError("stream requires num_workers=0")
        if self.initial_source_state is not None:
            self.close()
            self.source.load_state_dict(copy.deepcopy(self.initial_source_state))
            self.packer.load_state_dict(self.initial_packer_state)
            self.shuffle_buffer = []
            self.rng = random.Random(self.seed)
            self.source_exhausted = False
        self.packer.documents = self._documents()
        for row in self.packer:
            yield torch.from_numpy(row.astype(np.int64))

    def close(self):
        close = getattr(self.packer.documents, "close", None)
        if close is not None:
            close()
        self.packer.documents = iter(())

    def state_dict(self):
        local = copy.deepcopy({"source": self.source.state_dict(), "packer": self.packer.state_dict(),
            "shuffle_buffer": self.shuffle_buffer, "rng": self.rng.getstate(),
            "source_exhausted": self.source_exhausted, "settings": self.settings})
        if torch.distributed.is_initialized():
            states = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(states, local)
        else:
            states = [local]
        return {"rank_states": states}

    def load_state_dict(self, state):
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if len(state["rank_states"]) != world_size:
            raise ValueError("Streaming resume requires the same number of data ranks")
        local = copy.deepcopy(state["rank_states"][rank])
        if local["settings"] != self.settings:
            raise ValueError("Streaming resume requires the same tokenizer and packing settings")
        self.close()
        self.source.load_state_dict(local["source"])
        self.packer.load_state_dict(local["packer"])
        self.shuffle_buffer = local["shuffle_buffer"]
        self.rng.setstate(local["rng"])
        self.source_exhausted = local["source_exhausted"]
