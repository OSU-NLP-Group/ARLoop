"""Token-weighted validation of the fixed-loop release models."""
from __future__ import annotations
import os
from dataclasses import asdict, is_dataclass
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import torch
from recpre.path_utils import PROJECT_ROOT, resolve_project_path
from recpre.validation_cache import (
    get_cached_validation_batches, get_validation_batch_cache_source, rank_sequence_budget,
)

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader
    from loopedlm.tokenizer import Tokenizer
    from recpre.settings import CLISettings
    from recpre.utils import LightningFabric, SimpleFabric
    Fabric = LightningFabric | SimpleFabric

VALIDATION_BATCH_CACHE_DIR_ENV = "ARLOOP_VALIDATION_BATCH_CACHE_DIR"
DEFAULT_VALIDATION_BATCH_CACHE_DIR = PROJECT_ROOT / ".runtime" / "validation_batch_cache"

def get_validation_batch_cache_path(cfg: CLISettings, fabric: Fabric) -> Optional[Path]:
    if not cfg.cache_validation_batches:
        return None

    configured_cache_dir = cfg.validation_batch_cache_dir or os.getenv(VALIDATION_BATCH_CACHE_DIR_ENV)
    cache_dir = resolve_project_path(
        configured_cache_dir or DEFAULT_VALIDATION_BATCH_CACHE_DIR,
        label="validation batch cache directory",
    )
    val_entries = [asdict(entry) if is_dataclass(entry) else entry for entry in cfg.data_config.get("val_data", [])]
    fixed_sequence_budget = cfg.eval_num_sequences is not None
    local_eval_num_sequences = (
        rank_sequence_budget(
            cfg.eval_num_sequences, fabric.global_rank, fabric.world_size
        )
        if fixed_sequence_budget
        else None
    )
    signature = {
        "version": 2,
        "val_data": val_entries,
        "tokenizer_path": str(Path(cfg.tokenizer_path).expanduser().resolve(strict=False)),
        "seed": cfg.seed,
        "rank": fabric.global_rank,
        "world_size": fabric.world_size,
        "micro_batch_size": None if fixed_sequence_budget else cfg.micro_batch_size,
        "loader_block_size": cfg.loader_block_size,
        "eval_iters": None if fixed_sequence_budget else cfg.eval_iters,
        "eval_num_sequences": cfg.eval_num_sequences,
        "local_eval_num_sequences": local_eval_num_sequences,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "shuffle_blocks": cfg.shuffle_blocks,
        "shuffle_filenames": cfg.shuffle_filenames,
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"validation-v2-{digest}-rank{fabric.global_rank}-of-{fabric.world_size}.pt"



@torch.no_grad()
def validate(
    fabric: Fabric, model: nn.Module, val_dataloader: DataLoader, tokenizer: Tokenizer, cfg
) -> dict[str, torch.Tensor]:
    local_eval_num_sequences = (
        rank_sequence_budget(
            cfg.eval_num_sequences, fabric.global_rank, fabric.world_size
        )
        if cfg.eval_num_sequences is not None
        else None
    )
    if cfg.eval_num_sequences is not None:
        fabric.print(
            f"Validating on {cfg.eval_num_sequences} global sequences "
            f"({local_eval_num_sequences} on rank {fabric.global_rank}) ..."
        )
    else:
        fabric.print(f"Validating for {cfg.eval_iters} steps ...")
    model.eval()

    persistent_cache_path = get_validation_batch_cache_path(cfg, fabric)
    validation_batches, cache_created = get_cached_validation_batches(
        val_dataloader,
        cfg.eval_iters,
        persistent_cache_path=persistent_cache_path,
        num_sequences=local_eval_num_sequences,
        batch_size=cfg.micro_batch_size,
    )
    cache_source = get_validation_batch_cache_source(val_dataloader)
    if cache_created:
        cache_bytes = sum(
            tensor.numel() * tensor.element_size()
            for batch in validation_batches
            for tensor in batch
        )
        fabric.print(
            f"Cached {len(validation_batches)} validation batches "
            f"({cache_bytes / (1024 ** 2):.1f} MiB of CPU tensors per rank) at {persistent_cache_path}"
        )
    elif cache_source == "disk":
        fabric.print(f"Loaded validation batches from persistent cache: {persistent_cache_path}")

    eval_dtype = get_eval_precision_dtype(cfg.eval_precision, cfg.fabric_precision)
    autocast_dtype = get_eval_autocast_dtype(cfg.eval_precision)
    saved_state_dict = None

    if eval_dtype is not None and autocast_dtype is None:
        first_param = next(model.parameters())
        original_dtype = first_param.dtype
        if original_dtype != eval_dtype:
            fabric.print(f"Saving state dict and casting model from {original_dtype} to {eval_dtype} for evaluation")
            saved_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            model.to(eval_dtype)

    if autocast_dtype is not None:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
        fabric.print(f"Using autocast with dtype {autocast_dtype} for evaluation")
    else:
        autocast_ctx = nullcontext()

    ignore_index = model.objective["ignore_index"]

    def loss_sum_fn(logits, labels):
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )

    local_token_count = torch.tensor(
        sum(int(labels.ne(ignore_index).sum()) for _, labels in validation_batches),
        dtype=torch.int64,
        device=fabric.device,
    )
    local_sequence_count = torch.tensor(
        sum(input_ids.shape[0] for input_ids, _ in validation_batches),
        dtype=torch.int64,
        device=fabric.device,
    )
    global_token_count = fabric.all_reduce(local_token_count, reduce_op="sum")
    global_sequence_count = fabric.all_reduce(local_sequence_count, reduce_op="sum")
    if global_token_count.item() == 0:
        raise RuntimeError("Validation contains no non-padding target tokens")

    metrics = {
        "val_num_tokens": global_token_count,
        "val_num_sequences": global_sequence_count,
    }

    local_loss_sum = torch.zeros((), dtype=torch.float64, device=fabric.device)
    for input_ids, labels in validation_batches:
        input_ids = input_ids.to(fabric.device, non_blocking=True)
        labels = labels.to(fabric.device, non_blocking=True)
        with autocast_ctx:
            outputs = model(input_ids, return_logits=True)
        local_loss_sum += loss_sum_fn(outputs["logits"], labels).double()

    global_loss_sum = fabric.all_reduce(local_loss_sum, reduce_op="sum")
    global_val_loss = (global_loss_sum / global_token_count).float()
    metrics["val_loss"] = global_val_loss
    metrics["val_ppl"] = global_val_loss.exp()

    if saved_state_dict is not None:
        fabric.print(f"Restoring original model weights after evaluation")
        model.load_state_dict(saved_state_dict)
        del saved_state_dict

    model.train()
    return metrics



def get_eval_precision_dtype(eval_precision: str, train_precision: str) -> Optional[torch.dtype]:
    if eval_precision == "same":
        return None

    precision_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "bf16-mixed": torch.bfloat16,
        "fp16-mixed": torch.float16,
    }

    return precision_map.get(eval_precision, None)



def get_eval_autocast_dtype(eval_precision: str) -> Optional[torch.dtype]:
    if eval_precision in ["bf16-mixed", "fp16-mixed"]:
        return torch.bfloat16 if "bf16" in eval_precision else torch.float16
    return None
