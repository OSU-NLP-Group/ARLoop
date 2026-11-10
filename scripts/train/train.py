import time
import sys
import signal
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# These must be set before importing torch, recpre, or W&B. Persistent
# offline-service state stays in the checkout. Compiler caches and temporary
# files intentionally use the container defaults (normally node-local /tmp).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_ROOT = _PROJECT_ROOT / ".runtime"
_RUNTIME_PATHS = {
    "WANDB_CACHE_DIR": _RUNTIME_ROOT / "wandb" / "cache",
    "WANDB_CONFIG_DIR": _RUNTIME_ROOT / "wandb" / "config",
    "WANDB_DATA_DIR": _RUNTIME_ROOT / "wandb" / "data",
    "HF_HOME": _RUNTIME_ROOT / "huggingface",
    "HF_DATASETS_CACHE": _RUNTIME_ROOT / "huggingface" / "datasets",
    "TRANSFORMERS_CACHE": _RUNTIME_ROOT / "huggingface" / "transformers",
    "TORCHELASTIC_ERROR_FILE": _RUNTIME_ROOT / "torchelastic" / "error.json",
}
for _name, _path in _RUNTIME_PATHS.items():
    _path.parent.mkdir(parents=True, exist_ok=True)
    if _name != "TORCHELASTIC_ERROR_FILE":
        _path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_name, str(_path))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["WANDB_ERROR_REPORTING"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from recpre.path_utils import PROJECT_ROOT, resolve_project_path

from loopedlm.tokenizer import Tokenizer

global_start_time = time.time()
import math
import socket

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import json

import torch


if TYPE_CHECKING:
    import torch.distributed
    import torch.version
    import torch._dynamo.config
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy, SingleDeviceStrategy
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

import warnings

warnings.filterwarnings("ignore", message="The config.capture_autograd_function flag is deprecated")  # pytorch nightly
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")  # our weights

from recpre.settings import CLISettings
from recpre.blended_dataset import create_blended_dataloader
from recpre.checkpoint_utils import prune_old_checkpoints
from recpre.training_checkpoint import load_checkpoint
from loopedlm.models.arloop.config import ARLoopConfig
from recpre.runtime import set_torch_flags, derive_precision, warmup_or_early_fail_allreduce, _get_time_from_slurm
from recpre.schedulers import get_lr_scheduler

from recpre.packed_dataset import PackedBinaryDataset
from recpre.data_loading_utils import collate_packed_rows
from recpre.validation import validate
import recpre.utils
from recpre.monitor import enable_monitoring_on_step, disable_monitoring_and_retrieve_metrics, track_gradient_metrics, get_MFU_metrics
from cost import estimate_flops_gpt, FLOPBreakdown


def compute_flops_per_token_at_recurrence(flop_breakdown: FLOPBreakdown, forward_only: int, backprop_depth: int) -> int:
    if not flop_breakdown.is_recurrent:
        return 6 * flop_breakdown.total_params + flop_breakdown.total_attn_flops

    total_recurrence = forward_only + backprop_depth

    non_core_flops = 6 * flop_breakdown.non_core_params + flop_breakdown.non_core_attn_flops
    core_fwd = 2 * flop_breakdown.core_block_params * total_recurrence
    core_bwd = 4 * flop_breakdown.core_block_params * backprop_depth
    core_attn_fwd = flop_breakdown.core_attn_fwd_per_step * total_recurrence
    core_attn_bwd = 2 * flop_breakdown.core_attn_fwd_per_step * backprop_depth

    return int(non_core_flops + core_fwd + core_bwd + core_attn_fwd + core_attn_bwd)

from dataclasses import asdict, is_dataclass
from jsonargparse import CLI

RETRY_CACHE_INDUCTOR = False

if RETRY_CACHE_INDUCTOR:
    import torch._inductor.codecache

    torch._inductor.codecache.PyCodeCache.load_by_key_path = classmethod(recpre.utils.load_by_key_path_with_retry)

end_time = time.time()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")

Fabric = recpre.utils.LightningFabric | recpre.utils.SimpleFabric


def setup_fabric(cfg: CLISettings, wandb_run_id: Optional[str] = None) -> Fabric:
    if cfg.logger_name == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        logger_kwargs = {
            "project": cfg.logger_project,
            "name": cfg.run_name,
            "save_dir": cfg.out_dir
        }
        if cfg.logger_entity:
            logger_kwargs["entity"] = cfg.logger_entity
        if cfg.logger_group is not None:
            logger_kwargs["group"] = cfg.logger_group
        if wandb_run_id is not None:
            logger_kwargs["id"] = wandb_run_id
            logger_kwargs["resume"] = "allow"
        loggers = [WandbLogger(**logger_kwargs)]
    elif cfg.logger_name in {"none", "disabled"}:
        loggers = []
    else:
        raise ValueError(f"`logger={cfg.logger_name}` is not a valid option; choose 'none' or 'wandb'.")

    if cfg.fabric_strategy == "simple-ddp":
        fabric = recpre.utils.SimpleFabric(precision=cfg.fabric_precision, loggers=loggers)
        fabric.print("Using simple fabric.")
    else:
        if "fsdp" in cfg.fabric_strategy:
            if "grad" in cfg.fabric_strategy:
                sharding_strategy = "SHARD_GRAD_OP"
            elif "full" in cfg.fabric_strategy:
                sharding_strategy = "FULL_SHARD"
            elif "hybrid2" in cfg.fabric_strategy:
                sharding_strategy = "_HYBRID_SHARD_ZERO2"
            else:
                sharding_strategy = "HYBRID_SHARD"
            precision_strategy = derive_precision(cfg.fabric_precision, cfg.fabric)
            strategy = FSDPStrategy(
                auto_wrap_policy={cfg.model_config.Block},
                mixed_precision=precision_strategy,
                activation_checkpointing_policy=(
                    {cfg.model_config.Block}
                    if cfg.gradient_checkpointing and not isinstance(cfg.model_config, ARLoopConfig)
                    else None
                ),
                state_dict_type="full",
                sharding_strategy=sharding_strategy,  # type: ignore
                param_init_fn=((lambda x: x.to_empty(recurse=False)) if cfg.model_impl == "huggingface" else None),
            )
        elif cfg.fabric_strategy == "ddp":
            ddp_kwargs = {
                "find_unused_parameters": cfg.fabric.ddp_find_unused_parameters,
                "static_graph": cfg.fabric.ddp_static_graph,
                "gradient_as_bucket_view": cfg.fabric.ddp_gradient_as_bucket_view,
            }
            if cfg.fabric.ddp_bucket_cap_mb is not None:
                ddp_kwargs["bucket_cap_mb"] = cfg.fabric.ddp_bucket_cap_mb
            strategy = DDPStrategy(**ddp_kwargs)
        elif cfg.fabric_strategy == "single":
            strategy = SingleDeviceStrategy(device=torch.device("cuda:0") if torch.cuda.is_available() else "cpu")
        elif cfg.fabric_strategy == "axonn_tp":
            from axonn.lightning import AxonnStrategy
            strategy = AxonnStrategy(
                G_intra_r=cfg.fabric.row_tensor_parallel_size,
                G_intra_c=cfg.fabric.col_tensor_parallel_size,  # this needs more integration!
                G_intra_d=cfg.fabric.depth_tensor_parallel_size,
                overlap_communication=cfg.fabric.optimize_communication,
            )
        else:
            raise ValueError(f"`fabric_strategy={cfg.fabric_strategy}` is not a valid option.")

        fabric = recpre.utils.LightningFabric(
            devices=cfg.devices,
            strategy=strategy,
            precision=cfg.fabric_precision,
            loggers=loggers,
            num_nodes=cfg.num_nodes,
            use_lightning_environment=cfg.fabric_use_lightning_environment,
        )
        fabric.print(f"Using Lightning Fabric with strategy {cfg.fabric_strategy} ")
        fabric.launch()

    fabric.print(f"> gradient_accumulation_steps = {cfg.gradient_accumulation_steps}")
    fabric.print(f"> micro_batch_size = {cfg.micro_batch_size}")
    fabric.print(f"> global_batch_size = {cfg.world_batch_size}")

    return fabric


def define_wandb_metrics(fabric: Fabric, cfg: CLISettings) -> None:
    if fabric.global_rank != 0 or cfg.logger_name != "wandb" or not hasattr(fabric.logger, "experiment"):
        return
    experiment = fabric.logger.experiment
    if not hasattr(experiment, "define_metric"):
        return
    experiment.define_metric("global_step")
    experiment.define_metric("val_ppl_global_step", step_metric="global_step")


def log_validation_global_step_metrics(
    fabric: Fabric,
    state: dict,
    val_metrics: dict[str, torch.Tensor],
    cfg: CLISettings,
) -> None:
    if "val_ppl" not in val_metrics:
        return
    fabric.log_dict(
        {
            "global_step": state["optimizer_step"],
            "val_ppl_global_step": val_metrics["val_ppl"],
        },
        step=state["microbatch_step"],
    )


def startup(fabric: Fabric, cfg: CLISettings):
    start_time = time.time()

    if cfg.save_n_min_before_job_done is not None:
        if fabric.global_rank == 0:
            global_total_time = _get_time_from_slurm()
            fabric.print(f"Total job time: {global_total_time:.02f} seconds.")
        else:
            global_total_time = 0

        global_total_time = fabric.broadcast(global_total_time, 0)
        cfg.global_total_time = global_total_time

    if fabric.global_rank == 0:
        if cfg.resume_checkpoint_path is not None and cfg.resume_use_checkpoint_out_dir:
            checkpoint_path = Path(cfg.resume_checkpoint_path)
            if "checkpoints-" in str(checkpoint_path):
                cfg.out_dir = str(checkpoint_path.parent.parent)
                fabric.print(f"Using output directory from checkpoint path: {cfg.out_dir}")
    if torch.distributed.is_initialized():
        out_dir_list = [cfg.out_dir]
        torch.distributed.broadcast_object_list(out_dir_list, src=0)
        cfg.out_dir = out_dir_list[0]
    cfg.out_dir = str(resolve_project_path(cfg.out_dir, label="out_dir"))

    checkpoint_error = [None]
    if fabric.global_rank == 0 and not cfg.resume and Path(cfg.out_dir).exists():
        for subdir in ["checkpoints-DDPStrategy", "checkpoints-FSDPStrategy", "checkpoints-SingleDeviceStrategy"]:
            checkpoint_dir = Path(cfg.out_dir) / subdir
            if checkpoint_dir.exists():
                # Check for checkpoint files/directories (with or without .pth extension)
                checkpoints = list(checkpoint_dir.glob("step-*"))
                if checkpoints:
                    checkpoint_error[0] = f"Output directory {cfg.out_dir} already contains checkpoints. Use --resume=True to resume."
                    break
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(checkpoint_error, src=0)
    if checkpoint_error[0] is not None:
        raise ValueError(checkpoint_error[0])

    tokenizer = Tokenizer(cfg.tokenizer_path)
    cfg.model_config.ensure_vocab_size(len(tokenizer))

    if fabric.global_rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.out_dir) / fabric.get_prefix_for_checkpoint()).mkdir(parents=True, exist_ok=True)
        with open(f"{cfg.out_dir}/run_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=4)
        with open(f"{cfg.out_dir}/model_config.json", "w") as f:
            json.dump(asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config, f, indent=4)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    t0 = time.time()
    if not cfg.ignore_block_size_mismatch:
        assert cfg.block_size == cfg.model_config.block_size, "cfg.block_size must match config.block_size"
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=cfg.micro_batch_size,
        block_size=cfg.loader_block_size,
        fabric=fabric,
        seed=(cfg.seed + fabric.global_rank),
        cfg=cfg,
        tokenizer=tokenizer,
    )

    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate and setup dataloaders: {time.time() - t0:.02f} seconds.")

    fabric.seed_everything(cfg.seed)  # same seed for every process to init model (FSDP)
    fabric.print(f"Loading model with {cfg.model_config.__dict__}")

    loss_ignore_index = cfg.loss_ignore_index
    if loss_ignore_index is None:
        loss_ignore_index = tokenizer.pad_id if tokenizer.pad_id is not None else -100
    fabric.print(
        f"Using loss_ignore_index={loss_ignore_index} "
        f"(configured={cfg.loss_ignore_index}, tokenizer.pad_id={tokenizer.pad_id}, "
        f"tokenizer.eos_id={tokenizer.eos_id})"
    )
    objective = {
        "op": recpre.utils.chunked_cross_entropy,
        "label_smoothing": cfg.label_smoothing,
        "ignore_index": loss_ignore_index,
        "z_regularization": cfg.z_regularization,
    }

    t0 = time.time()
    with fabric.init_module(empty_init="fsdp" in cfg.fabric_strategy):
        model = cfg.model_config.construct_model(
            objective=objective,
            gradient_checkpointing=cfg.gradient_checkpointing and (
                isinstance(cfg.model_config, ARLoopConfig) or "fsdp" not in cfg.fabric_strategy
            ),
        )
        if cfg.drop_unused_tied_lm_head:
            if not getattr(cfg.model_config, "tie_embeddings", False):
                raise ValueError("drop_unused_tied_lm_head requires tie_embeddings=true")
            if not hasattr(model, "lm_head"):
                raise ValueError("drop_unused_tied_lm_head requested but model has no lm_head")
            del model.lm_head
    if cfg.drop_unused_tied_lm_head:
        fabric.print(
            "Removed unreachable lm_head; tied logits use transformer.wte.weight directly."
        )
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate model: {time.time() - t0:.02f} seconds.")
    num_params = recpre.utils.num_parameters(model)
    rec_params = recpre.utils.num_recurrent_parameters(model)
    fabric.log_to_summary({"num_parameters": num_params, "device": torch.cuda.get_device_name()})
    fabric.log_to_summary({"num_recurrent_parameters": rec_params})

    flop_breakdown = estimate_flops_gpt(model, cfg.model_config)
    flops_per_token = flop_breakdown.flops_per_token(use_curriculum_adjusted=False)

    if cfg.max_flops is not None and cfg.max_tokens == 0:
        cfg.max_tokens = int(cfg.max_flops / flops_per_token)
        fabric.print(f"Derived max_tokens={cfg.max_tokens:,} from max_flops={cfg.max_flops:.2e}")
    if cfg.token_to_param_ratio is not None and cfg.max_tokens == 0:
        cfg.max_tokens = int(cfg.token_to_param_ratio * num_params)
        fabric.print(f"Derived max_tokens={cfg.max_tokens:,} from {cfg.token_to_param_ratio}:1 token-to-param ratio ({num_params:,} params)")

    if cfg.max_steps is None:
        cfg.max_tokens_per_device = cfg.max_tokens // fabric.world_size
        cfg.tokens_per_step = cfg.micro_batch_size * cfg.block_size * cfg.gradient_accumulation_steps
        cfg.max_steps = cfg.max_tokens_per_device // cfg.tokens_per_step
        fabric.print(
            f"Based on block size {cfg.block_size}, expecting to take {cfg.max_steps} optimizer steps to reach "
            f"{cfg.max_tokens / 1e12:4.2f}T tokens.\nRunning {cfg.tokens_per_step * fabric.world_size} tok/step "
            f"for {cfg.max_tokens_per_device / 1e9:4.2f}B in total per device."
        )

    t0 = time.time()
    pre_ddp_regional_backends = {"selective_pre_ddp", "regional_attnres_pre_ddp"}
    if cfg.compile_model and cfg.compile_backend in pre_ddp_regional_backends:
        fabric.print(f"{time.ctime()[:-5]}: Regionally compiling transformer blocks BEFORE DDP wrapping...")
        compile_fn = lambda m: torch.compile(m, backend="inductor", mode=cfg.compile_mode, dynamic=False)

        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "h"):
                model.transformer.h = torch.nn.ModuleList([compile_fn(block) for block in model.transformer.h])
            if hasattr(model.transformer, "prelude"):
                model.transformer.prelude = torch.nn.ModuleList([compile_fn(block) for block in model.transformer.prelude])
            if hasattr(model.transformer, "core_block"):
                model.transformer.core_block = torch.nn.ModuleList([compile_fn(block) for block in model.transformer.core_block])
            if hasattr(model.transformer, "coda"):
                model.transformer.coda = torch.nn.ModuleList([compile_fn(block) for block in model.transformer.coda])
        if cfg.compile_backend == "regional_attnres_pre_ddp":
            if not hasattr(model, "_attn_res_from_values"):
                raise ValueError("regional_attnres_pre_ddp requires an AttnRes model")
            model._attn_res_from_values = compile_fn(model._attn_res_from_values)
        fabric.print(f"{time.ctime()[:-5]}: Pre-DDP regional compilation setup complete!")
    model = fabric.setup(model, compile=False)
    if cfg.fabric_strategy == "ddp" and cfg.fabric.all_reduce_dtype is not None:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        reduce_dtype = cfg.fabric.all_reduce_dtype.lower()
        if reduce_dtype not in {"bf16", "bf16-mixed"}:
            raise ValueError(
                "DDP communication compression currently supports only "
                f"bf16, got {cfg.fabric.all_reduce_dtype!r}"
            )
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            raise TypeError("DDP communication compression requires a DistributedDataParallel model")
        model.register_comm_hook(model.process_group, default_hooks.bf16_compress_hook)
        fabric.print("Registered BF16 DDP gradient communication hook.")
    if cfg.compile_model and cfg.compile_backend not in pre_ddp_regional_backends:
        if cfg.compile_backend == "selective":
            fabric.print(f"{time.ctime()[:-5]}: Selectively compiling transformer blocks with inductor...")
            compile_fn = lambda m: torch.compile(m, backend="inductor", mode=cfg.compile_mode, dynamic=False)
            inner_model = model.module if hasattr(model, "module") else model
            if hasattr(inner_model, "transformer"):
                if hasattr(inner_model.transformer, "h"):
                    inner_model.transformer.h = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.h])
                if hasattr(inner_model.transformer, "prelude"):
                    inner_model.transformer.prelude = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.prelude])
                if hasattr(inner_model.transformer, "core_block"):
                    inner_model.transformer.core_block = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.core_block])
                if hasattr(inner_model.transformer, "coda"):
                    inner_model.transformer.coda = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.coda])
            fabric.print(f"{time.ctime()[:-5]}: Selective compilation complete!")
        elif cfg.compile_backend == "selective_eager":
            fabric.print(f"{time.ctime()[:-5]}: Selectively compiling transformer blocks with EAGER (no AOT autograd)...")
            compile_fn = lambda m: torch.compile(m, backend="eager", dynamic=False)

            inner_model = model.module if hasattr(model, "module") else model
            if hasattr(inner_model, "transformer"):
                if hasattr(inner_model.transformer, "h"):
                    inner_model.transformer.h = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.h])
                if hasattr(inner_model.transformer, "prelude"):
                    inner_model.transformer.prelude = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.prelude])
                if hasattr(inner_model.transformer, "core_block"):
                    inner_model.transformer.core_block = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.core_block])
                if hasattr(inner_model.transformer, "coda"):
                    inner_model.transformer.coda = torch.nn.ModuleList([compile_fn(block) for block in inner_model.transformer.coda])
            fabric.print(f"{time.ctime()[:-5]}: Selective eager compilation complete!")
        elif cfg.compile_backend == "torchscript":
            fabric.print(f"{time.ctime()[:-5]}: Compiling transformer blocks with TorchScript...")

            inner_model = model.module if hasattr(model, "module") else model
            if hasattr(inner_model, "transformer"):
                if hasattr(inner_model.transformer, "prelude"):
                    for i, block in enumerate(inner_model.transformer.prelude):
                        try:
                            inner_model.transformer.prelude[i] = torch.jit.script(block)
                        except Exception as e:
                            fabric.print(f"Warning: Could not script prelude block {i}: {e}")
                if hasattr(inner_model.transformer, "core_block"):
                    for i, block in enumerate(inner_model.transformer.core_block):
                        try:
                            inner_model.transformer.core_block[i] = torch.jit.script(block)
                        except Exception as e:
                            fabric.print(f"Warning: Could not script core_block {i}: {e}")
                if hasattr(inner_model.transformer, "coda"):
                    for i, block in enumerate(inner_model.transformer.coda):
                        try:
                            inner_model.transformer.coda[i] = torch.jit.script(block)
                        except Exception as e:
                            fabric.print(f"Warning: Could not script coda block {i}: {e}")
            fabric.print(f"{time.ctime()[:-5]}: TorchScript compilation complete!")
        elif cfg.compile_backend == "post_ddp":
            fabric.print(f"{time.ctime()[:-5]}: Compiling model AFTER DDP wrapping...")
            model = torch.compile(
                model,
                backend="inductor",
                mode=cfg.compile_mode,
                fullgraph=cfg.compile_fullgraph,
                dynamic=cfg.compile_dynamic,
            )
            fabric.print(f"{time.ctime()[:-5]}: Post-DDP compilation complete!")
        else:
            fabric.print(f"{time.ctime()[:-5]}: Compiling with backend={cfg.compile_backend}...")
            model = torch.compile(
                model,
                backend=cfg.compile_backend,
                mode=cfg.compile_mode,
                fullgraph=cfg.compile_fullgraph,
                dynamic=cfg.compile_dynamic,
            )
            fabric.print(f"{time.ctime()[:-5]}: Model compiled!")
    if cfg.compile_model:
        fabric.print(f"{time.ctime()[:-5]}: Model compilation complete!")
    fabric.print(f"Model with full setup is {model}")
    fabric.print(f"Total parameters: {num_params:,}")
    fabric.print(f"{time.ctime()[:-5]}: Time to setup model: {time.time() - t0:.02f} seconds.")
    t0 = time.time()
    fabric.print(f"{time.ctime()[:-5]}: Setting up optimizer...")
    verbose_param_groups = fabric.global_rank == 0
    param_groups = recpre.optim.get_param_groups(
        model.named_parameters(),
        cfg.no_weight_decay_for_bias_and_norm_params,
        weight_lr_scale=1 / getattr(model.config, "mup_model_scaling_factor", 1.0),
        verbose=verbose_param_groups,
    )
    optimizer = recpre.optim.get_optimizer(
        cfg.optimizer,
        allow_fusion=not torch.version.hip and "bf16" in cfg.fabric_precision,
    )(param_groups, **cfg.optim_config)
    optimizer = fabric.setup_optimizers(optimizer)
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate and setup optimizers: {time.time() - t0:.02f} seconds.")

    assert cfg.max_steps is not None, "max_steps must be set before creating LR scheduler"
    warmup_steps = int(cfg.warmup_steps * cfg.max_steps) if cfg.warmup_steps < 1 else cfg.warmup_steps
    cooldown_steps = int(cfg.cooldown_steps * cfg.max_steps) if cfg.cooldown_steps < 1 else cfg.cooldown_steps
    lr_scheduler = get_lr_scheduler(
        schedule_type=cfg.lr_schedule,
        base_lr=cfg.optim_config["lr"],
        min_lr=cfg.min_lr,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        max_steps=cfg.max_steps,
    )

    max_flops = cfg.max_tokens * flops_per_token
    fabric.log_to_summary({"max_flops": max_flops, "flops_per_token": flops_per_token})
    fabric.print(f"Estimated max FLOPs: {max_flops/1e18:.5f} exaFLOPs ({flops_per_token:,} FLOPs/token)")


    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "lr_scheduler": lr_scheduler,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "microbatch_step": 0,
        "optimizer_step": 0,
        "is_accumulating": False,
        "metrics": {"lr": 0.0, "grad_norm": 0.0, "current_batch_size": 0},
        "should_exit_training": False,
        "model_config": asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config,
        "wandb_run_id": None,
        "flops_per_token": flops_per_token,
        "flop_breakdown": flop_breakdown,
        "cumulative_estimated_flops": 0,
        "_current_recurrence_split": None,
    }


    fabric.print(f"{time.ctime()[:-5]}: Checking for checkpoints...")
    load_checkpoint(
        fabric,
        state,
        cfg.out_dir,
        cfg.run_name,
        cfg.model_checkpoint,
        cfg.resume,
        cfg.resume_checkpoint_path,
        cfg.resume_model_only,
        cfg.resume_data_state,
    )

    if not cfg.resume_wandb_run:
        state["wandb_run_id"] = None

    if state.get("wandb_run_id") and fabric.global_rank == 0 and cfg.logger_name == "wandb" and hasattr(fabric.logger, "experiment"):
        if fabric.logger.experiment.id != state["wandb_run_id"]:
            import wandb
            fabric.logger.experiment.finish()
            init_kwargs = dict(id=state["wandb_run_id"], resume="allow", project=cfg.logger_project, name=cfg.run_name, dir=cfg.out_dir)
            if cfg.logger_entity:
                init_kwargs["entity"] = cfg.logger_entity
            fabric.logger._experiment = wandb.init(**init_kwargs)

    if asdict(cfg.model_config) != state["model_config"]:
        fabric.print("-------------Warning, model config difference between checkpoint and model config!-------------")

    fabric.print(f"{time.ctime()[:-5]}: Logging hyperparameters...")
    fabric.print(f"cmdline + derived cfg:\n{json.dumps(cfg.__dict__, default=lambda x: x.__dict__, indent=4)}")
    if fabric.global_rank == 0 and cfg.logger_name == "wandb":
        try:
            fabric.logger.log_hyperparams(cfg.__dict__)
        except Exception as e:
            fabric.print(f"Warning: W&B logging failed: {e}")

    fabric.print(f"{time.ctime()[:-5]}: Waiting at barrier before starting training...")
    fabric.barrier()
    end_time = time.time()
    fabric.print(f"{time.ctime()[:-5]}: Total time to run main func setups: {end_time - start_time:.02f} seconds.")

    return state


def train_step(input_ids, labels, fabric, state, running_loss, running_ppl, cfg):
    model = state["model"]
    optimizer = state["optimizer"]
    metrics = state["metrics"]

    state["microbatch_step"] += 1
    model.step = state["microbatch_step"]

    input_ids = input_ids.to(fabric.device, non_blocking=True)
    labels = labels.to(fabric.device, non_blocking=True)

    if state["microbatch_step"] < cfg.shape_watching_steps:
        bsz, seq_len = input_ids.shape
        fabric.print(f"bsz: {bsz} | seq_len: {seq_len}")
        fabric.print(f"input_ids.shape: {input_ids.shape} | labels.shape: {labels.shape}")
    elif state["microbatch_step"] == cfg.shape_watching_steps and cfg.shape_watching_steps > 0:
        fabric.print("Silencing shape watching ...")
    state["is_accumulating"] = state["microbatch_step"] % cfg.gradient_accumulation_steps != 0
    monitor_step = cfg.model_telemetry and state["microbatch_step"] % cfg.log_step_interval == 0
    if monitor_step and not state["is_accumulating"]:
        model.module.apply(partial(enable_monitoring_on_step, extreme=cfg.extreme_telemetry))

    def tightly_scoped_fwd_bwd(model, input_ids, labels):
        if state["microbatch_step"] == 1:
            fabric.print(f"{time.ctime()[:-5]}: About to run first forward pass (this may take 10-30 min if compiling)...")
        with fabric.no_backward_sync(model, enabled=state["is_accumulating"]):
            if cfg.compile_model and cfg.compile_mode == "reduce-overhead":
                torch.compiler.cudagraph_mark_step_begin()
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            if state["microbatch_step"] == 1:
                fabric.print(f"{time.ctime()[:-5]}: First forward pass complete, starting backward...")
            fabric.backward(loss / cfg.gradient_accumulation_steps, model=model)
            if state["microbatch_step"] == 1:
                fabric.print(f"{time.ctime()[:-5]}: First backward pass complete!")
            return loss.detach(), outputs["log_ppl"].detach()

    loss, log_ppl = tightly_scoped_fwd_bwd(model, input_ids, labels)
    metrics["mbs_loss"] = loss
    running_loss.update(loss)
    running_ppl.update(log_ppl)

    if not cfg.allow_nonfinite_loss and not torch.isfinite(loss):
        fabric.print(f"Loss is {loss} on {socket.gethostname()}. Terminating ...")
        state["should_exit_training"] = True

    if not state["is_accumulating"]:
        current_step_lr = state["lr_scheduler"].get_lr(state["optimizer_step"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = torch.as_tensor(current_step_lr * param_group["base_lr"])


        metrics["grad_norm"] = fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_clip, error_if_nonfinite=False)
        if torch.isfinite(metrics["grad_norm"]):
            if state["optimizer_step"] > 0:  # Skip first step if compiling or autotuning
                if cfg.compile_optimizer:
                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            if param.grad is not None:
                                torch._dynamo.decorators.mark_static_address(param.grad)
                    torch.compile(optimizer.step, mode="max-autotune-no-cudagraphs")()
                else:
                    optimizer.step()
        else:
            if cfg.skip_nonfinite_grads:
                fabric.print(f"Grad norm non-finite! Optimizer step {state['optimizer_step'] + 1} skipped.")
            else:
                fabric.print(f"Grad norm non-finite! Optimizer step {state['optimizer_step'] + 1}. Terminating ...")
                state["should_exit_training"] = True

        if monitor_step:
            track_gradient_metrics(model, optimizer, metrics)
            model.module.apply(partial(disable_monitoring_and_retrieve_metrics, metrics=metrics))
        optimizer.zero_grad(set_to_none=not (cfg.compile_optimizer))
        state["optimizer_step"] += 1
        cfg.gradient_accumulation_steps = get_batch_size(state["microbatch_step"], cfg)

        metrics["lr"] = current_step_lr
        metrics["current_batch_size"] = cfg.gradient_accumulation_steps * cfg.micro_batch_size * cfg.replicas

    if "flop_breakdown" in state and state["flop_breakdown"] is not None:
        tokens_this_step = cfg.micro_batch_size * cfg.block_size * fabric.world_size
        t, s = 0, 0
        state["_current_recurrence_split"] = (t, s)
        flops_this_step = compute_flops_per_token_at_recurrence(state["flop_breakdown"], t, s) * tokens_this_step
        state["cumulative_estimated_flops"] += flops_this_step


def train(fabric, state, cfg):
    fabric.print(f"{time.ctime()[:-5]}: Starting warmup...")
    warmup_or_early_fail_allreduce(fabric)
    define_wandb_metrics(fabric, cfg)
    fabric.print(f"{time.ctime()[:-5]}: Warmup complete, creating iterator...")
    state["initial_step"] = state["last_logged_step"] = state["microbatch_step"]
    if state["microbatch_step"] > 0:
        fabric.print(f"=== CHECKPOINT RESUME DIAGNOSTICS ===")
        fabric.print(f"  Resuming from microbatch_step: {state['microbatch_step']}")
        fabric.print(f"  optimizer_step: {state.get('optimizer_step', 'N/A')}")
        # Check LR at resume point (LR scheduler uses optimizer_step)
        expected_lr = state["lr_scheduler"].get_lr(state.get("optimizer_step", 0))
        fabric.print(f"  Expected LR at optimizer_step {state.get('optimizer_step', 0)}: {expected_lr:.6e}")
        # Check optimizer has state
        if hasattr(state["optimizer"], "state"):
            params_with_state = sum(1 for p in state["optimizer"].state if state["optimizer"].state[p])
            fabric.print(f"  Optimizer params with state: {params_with_state}")
        fabric.print(f"=====================================")

    train_iterator = iter(state["train_dataloader"])
    # Skip one batch on resume to fix StatefulDataLoader off-by-one
    resume_batch_skip_required = getattr(
        state["train_dataloader"].dataset, "requires_resume_batch_skip", True
    )
    if state["microbatch_step"] > 0 and resume_batch_skip_required:
        next(train_iterator, None)
        fabric.print(f"Skipped one batch to fix StatefulDataLoader off-by-one on resume")
    fabric.print(f"{time.ctime()[:-5]}: Iterator created, setting up metrics...")

    running_loss = RunningMean(window=cfg.log_step_interval, sync_on_compute=False).to(fabric.device)
    running_log_ppl = RunningMean(window=cfg.log_step_interval, sync_on_compute=False).to(fabric.device)
    first_validation_passed = False
    fabric.barrier()
    state["total_t0"] = time.time()
    fabric.print(f"{time.ctime()[:-5]}: Training preparations finished, starting to iterate train data now.")

    step_time = 0
    first_batch = True
    resumed_from_checkpoint = state["microbatch_step"] > 0
    pre_resume_loss = None
    for input_ids, labels, _ in train_iterator:
        if first_batch:
            fabric.print(f"{time.ctime()[:-5]}: Got first batch, shape: {input_ids.shape}, starting train_step...")
            first_batch = False

        step_before_train = state["microbatch_step"]
        is_save_boundary = (step_before_train > 0 and step_before_train % cfg.save_step_interval == 0) or \
                           (step_before_train > 0 and (step_before_train - 1) % cfg.save_step_interval == 0) or \
                           (step_before_train > 0 and (step_before_train + 1) % cfg.save_step_interval == 0)
        if is_save_boundary or (resumed_from_checkpoint and step_before_train <= state["initial_step"] + 2):
            batch_hash = hash(tuple(input_ids.flatten()[:100].tolist())) % (10**8)
            fabric.print(f"[BATCH_HASH] Step {step_before_train}: hash={batch_hash}, input_ids[0,:5]={input_ids[0, :5].tolist()}")
        t0 = time.time()
        train_step(input_ids, labels, fabric, state, running_loss, running_log_ppl, cfg=cfg)
        step_time += time.time() - t0
        step = state["microbatch_step"]
        if resumed_from_checkpoint and pre_resume_loss is None:
            pre_resume_loss = state["metrics"].get("mbs_loss", None)
            if pre_resume_loss is not None:
                # Compute batch hash for comparison (to verify data matches)
                batch_hash = hash(tuple(input_ids.flatten()[:100].tolist())) % (10**8)
                fabric.print(f"=== FIRST STEP AFTER RESUME ===")
                fabric.print(f"  Step {step}, Loss: {pre_resume_loss.item():.4f}")
                fabric.print(f"  LR: {state['metrics'].get('lr', 'N/A')}")
                fabric.print(f"  Batch hash (first 100 tokens): {batch_hash}")
                fabric.print(f"================================")

        validate_regular = (
            not state["is_accumulating"]
            and state["optimizer_step"] % cfg.eval_step_interval == 0
        )
        validate_at_the_end = state["optimizer_step"] >= cfg.max_steps - 1
        if validate_regular or validate_at_the_end:
            t0 = time.time()
            val_metrics = validate(fabric, state["model"], state["val_dataloader"], state["tokenizer"], cfg=cfg)
            td = time.time() - t0
            val_metrics["val_time"] = torch.as_tensor(td)

            fabric.print(f"Step {step}: Val loss {val_metrics['val_loss'].item():.4f}, Val time: {td:.2f}s")
            log_validation_global_step_metrics(fabric, state, val_metrics, cfg)
            state["metrics"] |= val_metrics
            if not first_validation_passed:
                fabric.log_to_summary({"first_validation_passed": time.time() - global_start_time})
                first_validation_passed = True
            fabric.barrier()

            if torch.distributed.is_initialized() and (state["microbatch_step"] % 32) == 0:
                exit_tensor = torch.as_tensor([int(state["should_exit_training"])], device=fabric.device)
                torch.distributed.all_reduce(exit_tensor, torch.distributed.ReduceOp.MIN, async_op=False)
                state["should_exit_training"] = bool(exit_tensor.item())  # always cast back to bool

        # Log at an interval.
        if step % cfg.log_step_interval == 0 or (state["should_exit_training"] and (step % 32) == 0):
            log_step(fabric, state, running_loss, running_log_ppl, step_time, cfg)
            step_time = 0

        if state["should_exit_training"] and (state["microbatch_step"] % 32) == 0:
            fabric.print(f"{time.ctime()[:-5]}: Exiting training early in step {step} due to error signal received.")
            break

        maybe_save_checkpoint(fabric, state, cfg, is_accumulating=state["is_accumulating"])

        exit_step = cfg.exit_at_step if cfg.exit_at_step is not None else cfg.max_steps
        if state["optimizer_step"] >= exit_step - 1:
            fabric.print(f"{time.ctime()[:-5]}: Exiting training orderly after completion of {step + 1} steps.")
            break


def log_step(
    fabric: Fabric,
    state: dict,
    running_loss: RunningMean,
    running_log_ppl: RunningMean,
    accumulated_step_time: float,
    cfg: CLISettings,
):
    loss = running_loss.compute()
    log_ppl = running_log_ppl.compute()
    t1 = time.time()

    metrics = state["metrics"]

    avg_time_per_step = accumulated_step_time / (state["microbatch_step"] - state["last_logged_step"])
    tokens_per_step = cfg.micro_batch_size * cfg.block_size * fabric.world_size
    tokens_per_second = tokens_per_step / avg_time_per_step

    metrics |= {
        "local_loss": loss,
        "local_ppl": log_ppl.exp(),
        "microbatch_step": state["microbatch_step"],
        "optimizer_step": state["optimizer_step"],
        "global_step": state["optimizer_step"],
        "steps/second": 1 / avg_time_per_step,
        "seconds/step": avg_time_per_step,
        "tokens/second": tokens_per_second,
        "remaining_time": (
            (t1 - state["total_t0"])
            / (state["microbatch_step"] - state["initial_step"])
            * (cfg.max_steps * cfg.gradient_accumulation_steps - state["microbatch_step"])
        ),
        "total_tokens": state["microbatch_step"] * tokens_per_step,
        "total_time": t1 - state["total_t0"],
    }
    if cfg.measure_utilization:
        max_memory_allocated_per_gpu = torch.cuda.max_memory_allocated(fabric.device) / 1024**3
        max_mem_reserved_per_gpu = torch.cuda.max_memory_reserved(fabric.device) / 1024**3
        torch.cuda.reset_peak_memory_stats(fabric.device)
        model_flops, tflops, mfu = get_MFU_metrics(tokens_per_second, fabric, state["model"], cfg.fabric_precision)
        metrics |= {
            "total_FLOPs": state["microbatch_step"] * tokens_per_step * model_flops,
            "FLOP/S": tflops,
            "model_flop_utilization": mfu,
            "max_mem_per_gpu": max_memory_allocated_per_gpu,
            "max_mem_reserved_per_gpu": max_mem_reserved_per_gpu,
        }

    if "flops_per_token" in state:
        # Simple estimate using target (post-ramp) flops_per_token
        metrics["total_estimated_FLOPs"] = metrics["total_tokens"] * state["flops_per_token"]
    if "cumulative_estimated_flops" in state:
        metrics["total_estimated_FLOPs_curriculum"] = state["cumulative_estimated_flops"]
        if "_current_recurrence_split" in state:
            t, s = state["_current_recurrence_split"]
            metrics["expected_forward_only_steps"] = t
            metrics["expected_backprop_depth"] = s
            metrics["expected_total_recurrence"] = t + s

    if "grad_norm" in metrics and metrics["grad_norm"] is not None:
        grad_norm = fabric.all_reduce(metrics["grad_norm"])
        metrics["global_grad_norm"] = grad_norm
    else:
        metrics["global_grad_norm"] = None
    metrics["global_loss"] = fabric.all_reduce(loss)

    metrics["global_train_ppl"] = fabric.all_reduce(log_ppl).exp()


    fabric.log_dict(metrics, step=state["microbatch_step"])
    state["last_logged_step"] = state["microbatch_step"]

    step_timing = (
        f" steps/sec: {metrics['steps/second']:4.2f}  |"
        if metrics["steps/second"] >= 1.0
        else f" secs/step: {metrics['seconds/step']:4.2f}  |"
    )
    lr_str = f"{metrics['lr']:2.4e}" if "lr" in metrics and metrics["lr"] is not None else ""
    grad_norm_str = f"{metrics['global_grad_norm']:6.4e}" if metrics["global_grad_norm"] is not None else ""

    fabric.print(
        f"{time.ctime()[:-5]}\n"
        f"Step {metrics['microbatch_step']:>8}    | Loss: {metrics['global_loss']:7.4f} | {metrics['global_train_ppl']:9.2f} PPL     |"
        f" Update {metrics['optimizer_step']:>8}     |\n"
        f"{'(optimizer.step)' if not state['is_accumulating'] else ' ' * 16}"
        f" | LR: {lr_str:>10}| Grad norm: {grad_norm_str:>11} |{' ' * 19}|\n"
        f"                 | MFU : {metrics.get('model_flop_utilization', 0):6.2%}  | TFLOP/S : {metrics.get('FLOP/S', 0):5.2f}  |"
        f" tok/sec: {metrics['tokens/second']:8.1f} | {step_timing}\n"
        f"                 | Max mem allocated: {metrics.get('max_mem_per_gpu', 0):4.2f} GB       "
        f"| Max mem reserved: {metrics.get('max_mem_reserved_per_gpu', 0):4.2f} GB            |\n"
        f"                 | Tokens: {metrics['total_tokens'] / 1e9: 4.1f}B | exaFLOP: {metrics.get('total_FLOPs', 0) / 1e18:8.5f} |"
        f" Remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days             |"
    )
    state["metrics"] = {}


def create_dataloader(data_config, batch_size, block_size, fabric, seed=1337, *, cfg, tokenizer, stateful=True):
    """Build a packed or streaming reader, blending multiple packed training sources."""
    if not data_config:
        raise ValueError("No data sources configured")
    if any(entry.type == "stream" for entry in data_config):
        if len(data_config) != 1:
            raise ValueError("stream requires exactly one source per train/validation loader")
        if cfg.dataloader_num_workers != 0:
            raise ValueError("stream requires dataloader_num_workers=0")
        from recpre.streaming_dataset import StreamingPackedDataset, streaming_source
        entry = data_config[0]
        source = streaming_source(entry.dataset, dataset_config=entry.dataset_config,
            dataset_split=entry.dataset_split, data_files=entry.data_files, text_column=entry.text_column,
            seed=cfg.seed, world_size=fabric.world_size, rank=fabric.global_rank)
        dataset = StreamingPackedDataset(source, tokenizer, block_size=cfg.loader_block_size,
            buffer_size=entry.packing_buffer_size, shuffle_buffer_size=entry.shuffle_buffer_size,
            seed=cfg.seed, source_id=(entry.dataset, entry.dataset_config, entry.dataset_split,
                                      entry.data_files, entry.text_column), stateful=stateful)
        loader_class = StatefulDataLoader if stateful else DataLoader
        return loader_class(dataset, batch_size=batch_size, num_workers=0,
            pin_memory=True, collate_fn=partial(collate_packed_rows, tokenizer=tokenizer))
    if any(entry.type != "packed-bin" or not entry.data_dir for entry in data_config):
        raise ValueError("Every data source must be packed-bin with a data_dir")
    if stateful and len(data_config) > 1:
        return create_blended_dataloader(data_config, batch_size, fabric, cfg=cfg, tokenizer=tokenizer)
    if len(data_config) != 1:
        raise ValueError("Validation requires exactly one packed source")
    dataset = PackedBinaryDataset(
        dataset_folder_path=data_config[0].data_dir,
        block_size=cfg.loader_block_size,
        seed=cfg.seed,
        shuffle_filenames=cfg.shuffle_filenames,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
        stateful=stateful,
    )
    loader_class = StatefulDataLoader if stateful else DataLoader
    return loader_class(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        collate_fn=partial(collate_packed_rows, tokenizer=tokenizer),
        num_workers=cfg.dataloader_num_workers,
        prefetch_factor=4 if cfg.dataloader_num_workers > 0 else None,
    )


def create_dataloaders(batch_size, block_size, fabric, seed=1337, *, cfg, tokenizer):
    fabric.print(f"Creating dataloaders with seed: {seed}")
    kwargs = dict(batch_size=batch_size, block_size=block_size, fabric=fabric, seed=seed, cfg=cfg, tokenizer=tokenizer)
    train_loader = create_dataloader(cfg.data_config["train_data"], **kwargs)
    val_loader = (create_dataloader(cfg.data_config["val_data"], stateful=False, **kwargs)
                  if "val_data" in cfg.data_config else None)
    return train_loader, val_loader


def get_batch_size(step: int, cfg: CLISettings) -> int:
    if step > cfg.batch_size_ramp:
        gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    else:
        slope = step / cfg.batch_size_ramp
        gradient_accumulation_steps = math.ceil(slope * cfg.batch_size / cfg.micro_batch_size)
    return gradient_accumulation_steps


def maybe_save_checkpoint(fabric, state, cfg, is_accumulating=False, force_save=False):
    t0 = time.time()
    prefix = fabric.get_prefix_for_checkpoint()
    fully_qualified_checkpoint_path = f"{cfg.out_dir}/{prefix}/step-{state['microbatch_step']:08d}-{cfg.run_name}"

    if cfg.save_wallclock_interval_seconds is not None:
        now = time.monotonic()
        last_save = state.setdefault("_runtime_last_checkpoint_time", now)
        save_at_interval = False
        if not is_accumulating:
            due = torch.tensor(
                int(fabric.global_rank == 0 and now - last_save >= cfg.save_wallclock_interval_seconds),
                device=fabric.device, dtype=torch.int32,
            )
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(due, src=0)
            save_at_interval = bool(due.item())
    elif cfg.save_optimizer_step_interval is not None:
        save_at_interval = (
            not is_accumulating
            and state["optimizer_step"] > 0
            and state["optimizer_step"] % cfg.save_optimizer_step_interval == 0
        )
    else:
        save_at_interval = (
            not is_accumulating
            and state["microbatch_step"] % cfg.save_step_interval == 0
        )
    if cfg.save_n_min_before_job_done is not None and (state["microbatch_step"] % 32) == 0:
        time_spent = time.time() - global_start_time
        remaining_time = cfg.global_total_time - time_spent
        remaining_time = remaining_time / 60.0
        remaining_time = fabric.all_reduce(remaining_time, reduce_op="mean")  # slowdown?
        save_before_timeout = remaining_time <= cfg.save_n_min_before_job_done
        if save_before_timeout:
            fabric.print(f"{time.ctime()[:-5]}: Saving at {remaining_time:.02f} minutes left")
            cfg.save_n_min_before_job_done = None  # reset
    else:
        save_before_timeout = False

    save_at_first_step = cfg.save_first_step and (state["microbatch_step"] == 0)
    effective_max = cfg.exit_at_step if cfg.exit_at_step is not None else cfg.max_steps
    save_at_last_step = cfg.save_last_step and (state["optimizer_step"] >= (effective_max - 1))

    if save_at_interval or save_at_last_step or save_at_first_step or save_before_timeout or force_save:
        fabric.print(f"--------------------- {time.ctime()[:-5]} Model Save triggered --------")
        fabric.print(f"Saving to {str(fully_qualified_checkpoint_path)!r}")

        if fabric.global_rank == 0 and cfg.logger_name == "wandb" and hasattr(fabric.logger, "experiment"):
            state["wandb_run_id"] = fabric.logger.experiment.id

        import numpy as np
        import random
        state["rng_state_torch"] = torch.get_rng_state()
        state["rng_state_cuda"] = torch.cuda.get_rng_state(device=fabric.device)
        state["rng_state_numpy"] = np.random.get_state()
        state["rng_state_python"] = random.getstate()

        if fabric.global_rank == 0:
            opt = state["optimizer"]
            opt_state = opt.state if hasattr(opt, "state") else {}
            params_with_state = sum(1 for s in opt_state.values() if s)
            total_m_norm = sum(s.get('m', torch.zeros(1)).float().norm().item() for s in opt_state.values() if s and 'm' in s)
            total_v_norm = sum(s.get('v', torch.zeros(1)).float().norm().item() for s in opt_state.values() if s and 'v' in s)
            fabric.print(f"  Optimizer state: {params_with_state} params, m_norm={total_m_norm:.2f}, v_norm={total_v_norm:.2f}")

        dl = state["train_dataloader"]
        ds = dl.dataset if hasattr(dl, 'dataset') else None
        if ds is not None and hasattr(ds, '_state'):
            fabric.print(f"  Dataset position: file_idx={ds._state.get('file_idx')}, "
                         f"row_group_idx={ds._state.get('row_group_idx')}, "
                         f"buffer_len={len(ds._state.get('buffer', []))}")

        # Validation loaders, tokenizer, and scheduler are recreated from the recipe.
        save_state = {
            key: value for key, value in state.items()
            if not key.startswith("_runtime_") and key not in {"val_dataloader", "tokenizer", "lr_scheduler"}
        }
        fabric.save(fully_qualified_checkpoint_path, save_state)
        state["_runtime_last_checkpoint_time"] = time.monotonic()
        fabric.print(f"------------------- {time.ctime()[:-5]} Checkpoint saved ({time.time() - t0:.02f} seconds)")

        if cfg.max_checkpoints_to_keep is not None:
            fabric.barrier()
            if fabric.global_rank == 0:
                checkpoint_dir = Path(cfg.out_dir) / prefix
                try:
                    deleted = prune_old_checkpoints(
                        checkpoint_dir,
                        cfg.run_name,
                        cfg.max_checkpoints_to_keep,
                        allowed_root=PROJECT_ROOT,
                    )
                except OSError as exc:
                    fabric.print(f"WARNING: Failed to prune old checkpoints in {checkpoint_dir}: {exc}")
                else:
                    if deleted:
                        deleted_names = ", ".join(path.name for path in deleted)
                        fabric.print(
                            f"Pruned {len(deleted)} old checkpoint artifact(s), "
                            f"keeping the newest {cfg.max_checkpoints_to_keep} step(s): {deleted_names}"
                        )
            fabric.barrier()


import sys
import datetime


def main():
    cfg: CLISettings = CLI(CLISettings)  # type: ignore

    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    set_torch_flags(cfg)

    wandb_run_id = None
    if cfg.resume and cfg.resume_checkpoint_path and cfg.resume_wandb_run:
        try:
            ckpt = torch.load(cfg.resume_checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
            wandb_run_id = ckpt.get("wandb_run_id") if isinstance(ckpt, dict) else None
        except Exception:
            pass

    fabric = setup_fabric(cfg, wandb_run_id=wandb_run_id)
    fabric.print(f"{time.ctime()[:-5]}: Calling startup()...")
    state = startup(fabric, cfg)
    fabric.print(f"{time.ctime()[:-5]}: startup() complete, calling train()...")

    signal.signal(signal.SIGUSR1, lambda s, f: (fabric.print("SIGUSR1: saving checkpoint..."), state.__setitem__("should_exit_training", True), maybe_save_checkpoint(fabric, state, cfg, force_save=True)))

    train_time = time.time()
    try:
        train(fabric, state, cfg)
    except (KeyboardInterrupt, Exception) as e:
        fabric.print(f"Training interrupted: {e}. Saving emergency checkpoint...")
        maybe_save_checkpoint(fabric, state, cfg, is_accumulating=False, force_save=True)
        raise
    finally:
        close_dataset = getattr(state["train_dataloader"].dataset, "close", None)
        if close_dataset is not None:
            close_dataset()

    fabric.print("--------------------------------------------------------------------")
    fabric.print(f"Training time: {str(datetime.timedelta(seconds=time.time() - train_time))} ")
    fabric.log_to_summary(
        {"train_time": time.time() - global_start_time, "total_time": time.time() - global_start_time}
    )
    if fabric.device.type == "cuda":
        max_alloc = f"{torch.cuda.max_memory_allocated(fabric.device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(fabric.device) / float(1024**3):,.3f} GB"
        fabric.print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
    fabric.print("--------------------------------------------------------------------")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print(f"Run {cfg.run_name} finished without error.")
        print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
        print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        main()
    except BaseException:  # gate around hell to guarantee NCCL deconstruction
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if int(os.getenv("SLURM_PROCID", "0")) == 0:
            print("Run finished with errors.")
            print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
            print("-----------------Shutdown complete.--------------------------")

            raise


if __name__ == "__main__":
    guarded_main()
