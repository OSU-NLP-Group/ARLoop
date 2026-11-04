"""Collation of prepacked token rows."""
import torch
from torch.utils.data._utils.collate import collate_tensor_fn
from loopedlm.tokenizer import Tokenizer

def shift_inputs_and_labels(inputs_batch: torch.Tensor, labels_batch: torch.Tensor, tokenizer: Tokenizer):
    seq_len = inputs_batch.shape[1]

    input_ids = inputs_batch[:, 0 : (seq_len - 1)].contiguous().long()
    label_ids = labels_batch[:, 1:(seq_len)].contiguous().long()

    if tokenizer.pad_id is not None and tokenizer.eos_id is not None:
        input_ids[input_ids == tokenizer.pad_id] = tokenizer.eos_id  # type: ignore
    return input_ids, label_ids



def collate_packed_rows(batch: list[torch.Tensor], tokenizer: Tokenizer):
    rows = collate_tensor_fn(batch)
    input_ids, labels = shift_inputs_and_labels(rows, rows.clone(), tokenizer)
    return input_ids, labels, [None] * len(batch)
