# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Utility functions for training and inference."""

import math
import os

from typing import TYPE_CHECKING, Any, Dict, Optional, TypeVar, Union, TypedDict
from functools import partial
from contextlib import nullcontext, contextmanager
from datetime import timedelta

import lightning as L
import torch
import torch.nn as nn
import torch.utils._device
import numpy as np
import random
import time
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors

if TYPE_CHECKING:
    import torch.distributed


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total: int = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)  # type: ignore
            else:
                total += p.numel()
    return total

def num_recurrent_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total: int = 0
    if hasattr(module, "transformer") and hasattr(module.transformer, "core_block") and module.transformer.core_block is not None:
        for p in module.transformer.core_block.parameters():
            if requires_grad is None or p.requires_grad == requires_grad:
                if hasattr(p, "quant_state"):
                    # bitsandbytes 4bit layer support
                    total += math.prod(p.quant_state.shape)  # type: ignore
                else:
                    total += p.numel()
        return total

    return 0


T = TypeVar("T")


def slice_logits_remap_labels(
    logits: torch.Tensor, targets: torch.Tensor, target_range: tuple, ignore_indices: list[int] = [-1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slices logits to a specific index range, and remaps target labels to the new range, zero indexed."""
    assert min(target_range) >= 0, "smallest target_range value must be non-negative"
    assert max(target_range) < logits.size(-1), "largest target_range value must be less than the number of logits"
    assert sorted(target_range) == list(target_range), "target_range must be sorted already, sanity convention"

    label_id_to_tgt_id = {label_id: tgt_id for tgt_id, label_id in enumerate(target_range)}
    logits = logits[:, target_range]
    for label_id, tgt_id in label_id_to_tgt_id.items():
        targets[targets == label_id] = tgt_id

    return logits, targets


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_indices: list[int] = [-1],
    reduction: Optional[str] = None,
    training: bool = True,
    z_loss_eps=0.0,
    target_range: Optional[tuple[int]] = None,
    return_logits_targets: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ignore_index = ignore_indices[0] or -100
    for additional_ignore in ignore_indices[1:]:
        if additional_ignore is not None and additional_ignore != ignore_index:
            targets[targets == additional_ignore] = ignore_index

    cross_entropy_fn = partial(
        torch.nn.functional.cross_entropy,
        ignore_index=ignore_index,
    )

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    if target_range is not None:
        # print("Using slice loss!")
        logits, targets = slice_logits_remap_labels(logits, targets, target_range, ignore_indices)
    else:
        # print("Using normal loss!")
        pass

    if chunk_size == 0:
        # no chunking at all
        if reduction is not None:
            loss = cross_entropy_fn(input=logits, target=targets, reduction=reduction)
        else:
            loss = cross_entropy_fn(input=logits, target=targets)
    else:
        # chunk cross entropy
        logit_chunks = logits.split(chunk_size)
        target_chunks = targets.split(chunk_size)
        losses = torch.zeros_like(targets, dtype=logits.dtype, device=logits.device)  # prealloc required for compile

        for idx, (logit_chunk, target_chunk) in enumerate(zip(logit_chunks, target_chunks)):
            loss_chunk = cross_entropy_fn(input=logit_chunk, target=target_chunk, reduction="none")
            losses[idx * chunk_size : (idx + 1) * chunk_size] = loss_chunk

        non_masked_elems = (targets != ignore_index).sum().clamp(min=1.0)
        loss = losses.sum() / non_masked_elems

    if z_loss_eps > 0.0 and training:
        loss += z_loss_eps * torch.logsumexp(logits, dim=-1).pow(2).mean()

    if not return_logits_targets:
        return loss
    else:
        non_masked_mask = targets != ignore_index
        return loss, logits[non_masked_mask, :], targets[non_masked_mask]


# wrapper for lightning fabrics
class LightningFabric:
    def __init__(self, devices, strategy, precision, loggers=[], num_nodes: int = 1, use_lightning_environment=False):
        plugins = LightningEnvironment() if use_lightning_environment else None
        fabric = L.Fabric(
            devices=devices,
            strategy=strategy,
            precision=precision,
            loggers=loggers,
            num_nodes=num_nodes,
            plugins=plugins,
        )
        self._underlying_fabric = fabric
        self._precision = precision
        self.strategy_name = strategy.__class__.__name__

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._underlying_fabric, attr)

    def get_prefix_for_checkpoint(self):
        return f"checkpoints-{self.strategy_name}"

    def log_dict(self, metrics, step=None):
        """Log metrics to wandb, converting tensors to scalars."""
        if self.global_rank == 0:
            metrics = convert_tensors_to_scalars(metrics)
            for logger in self.loggers:
                logger.log_metrics(metrics=metrics, step=step)

    def log_to_summary(self, dict_of_values):
        if self.global_rank == 0 and self.loggers:
            for key, value in dict_of_values.items():
                self.loggers[0].experiment.summary[key] = value

    def log_chart(self, dict_of_charts):
        if self.global_rank == 0 and self.loggers:
            self.loggers[0].experiment.log(dict_of_charts)

    def setup(self, model: torch.nn.Module, compile: bool = False, compile_ddp: bool = True):
        """Compiling DDP turns out not to improve speed on the A6000 ada cards"""

        if "DDP" in self.strategy_name and compile and compile_ddp:
            # the default dtype needs to be set before the lightning fabric handles it, otherwise DDP cannot be compiled
            # due to default dtypes changes in the compiled program
            if "bf16-true" in self._precision:
                torch.set_default_dtype(torch.bfloat16)
            elif "16-true" in self._precision:
                torch.set_default_dtype(torch.float16)

            model = self._underlying_fabric.setup(model)
            if compile:
                model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")  # type: ignore
        else:
            if compile:
                # error on dynamic shape
                model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")  # type: ignore
            model = self._underlying_fabric.setup(model)

        return model


# @torch.compile() # funny hang on A100s + very large model?
# maybe the graph is too long
@torch.no_grad()
def _allreduce_chunk_stream(model, world_size=1, device=torch.device("cpu"), safety_goggles_on=False):
    """Simple implementation with fixed MB chunks that can span gradients"""
    chunk_size = 1024 * 1024 * 64 // 4  # 64MB fp32 as in warmup

    chunk = torch.empty(chunk_size, dtype=torch.float32, device=device)
    chunk_index = 0
    param_refs = []

    for p in model.parameters():
        if p.grad is not None:
            grad_index = 0
            while grad_index < p.grad.numel():
                # Fold
                n = min(chunk_size - chunk_index, p.grad.numel() - grad_index)
                chunk[chunk_index : chunk_index + n] = p.grad.view(-1)[grad_index : grad_index + n]
                param_refs.append((p, grad_index, chunk_index, n))
                chunk_index += n
                grad_index += n

                if chunk_index == chunk_size:
                    # Average over ranks
                    torch.distributed.all_reduce(chunk)
                    chunk.div_(world_size)
                    if safety_goggles_on:
                        torch.cuda.current_stream().synchronize()
                    # Unfold
                    for param, start_p, start_c, numel in param_refs:
                        param.grad.view(-1)[start_p : start_p + numel] = chunk[start_c : start_c + numel]

                    # Reset
                    chunk = torch.empty(chunk_size, dtype=torch.float32, device=device)
                    chunk_index = 0
                    param_refs = []
    # Handle final chunk:
    if chunk_index > 0:
        torch.distributed.all_reduce(chunk)  # keep consistent MB size
        chunk.div_(world_size)
        for param, start_p, start_c, numel in param_refs:
            param.grad.view(-1)[start_p : start_p + numel].copy_(chunk[start_c : start_c + numel])


class SimpleFabric:
    """Simple ddp-based fabric without lightning cruft, can be a template for other fabrics"""

    verbose = False

    def __init__(self, precision="bf16-true", loggers=None, local_device_init=True, use_dumb_allreduce=True):
        self.precision = precision
        self.rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
        self.local_device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        torch.distributed.init_process_group(
            backend="nccl",  # can disable to build both nccl and gloo backends, but gloo may be a gamble on frontier
            rank=self.rank,
            world_size=int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", -1))),
            device_id=self.local_device if local_device_init else None,  # this immediately forms the NCCL communicator
            timeout=timedelta(minutes=15),
        )
        loggers = loggers if loggers is not None else []
        self._loggers = loggers if isinstance(loggers, list) else [loggers]

        if precision == "bf16-true":
            self.dtype = torch.bfloat16
        elif precision in ["fp16-mixed", "bf16-mixed", "16-mixed"]:
            self.dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision type {precision} provided.")
        # Please be careful with vague constructors with this turned on, float32 constructors should say so:
        torch.set_default_dtype(self.dtype)
        torch.cuda.set_device(self.local_device)
        self.global_rank_for_creating_dataloader = self.rank
        self.world_size = torch.distributed.get_world_size()
        self.local_devices = min(int(os.getenv("SLURM_NTASKS_PER_NODE", torch.cuda.device_count())), self.world_size)
        self.local_rank = int(os.getenv("LOCAL_RANK", self.rank % self.local_devices))
        assert self.local_rank == self.rank % self.local_devices
        self.strategy_name = "SimpleDDP"
        self.use_dumb_allreduce = use_dumb_allreduce
        self.no_sync = False

    def launch(self):
        pass

    def print(self, msg, ranks=[0]):
        if torch.distributed.get_rank() in ranks:
            print(msg, flush=True)

    def log_dict(self, metrics, step=None):
        if self.rank == 0:
            metrics = convert_tensors_to_scalars(metrics)
            for logger in self._loggers:
                logger.log_metrics(metrics=metrics, step=step)

    def log_to_summary(self, dict_of_values):
        if self.rank == 0 and self._loggers:
            for key, value in dict_of_values.items():
                self.logger.experiment.summary[key] = value

    def log_chart(self, dict_of_charts):
        if self.rank == 0 and self._loggers:
            self.logger.experiment.log(dict_of_charts)

    @property
    def device(self):
        return self.local_device

    @property
    def global_rank(self):
        return self.rank

    @property
    def logger(self):
        return self._loggers[0]

    @property
    def strategy(self):
        return "SimpleDDP"

    def setup_dataloaders(self, train_dataloader, val_dataloader):
        return train_dataloader, val_dataloader

    def setup(self, model, compile: bool = False):
        model = model.to(self.device)
        # wrap DDP and AMP
        if not self.use_dumb_allreduce:
            self._ddp_model_ref = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
        else:
            self._ddp_model_ref = model
        self.model_ref = _FabricModule(
            forward_module=self._ddp_model_ref,
            original_module=model,
            precision=self.precision,
        )
        # compile after DDP to compile DDP calls
        if compile:
            self.model_ref = torch.compile(
                self.model_ref, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs"
            )
        return self.model_ref

    def setup_optimizers(self, optimizers):
        return optimizers

    def all_reduce(self, data, reduce_op="mean") -> torch.Tensor:
        """
        All reduce over data parallel groups
        """
        op = torch.distributed.ReduceOp.AVG
        if reduce_op == "sum":
            op = torch.distributed.ReduceOp.SUM
        data = convert_to_tensors(data, device=self.device)

        torch.distributed.all_reduce(data, op=op)
        return data

    def barrier(self):
        torch.distributed.barrier(device_ids=[self.device.index])

    def broadcast(self, obj, src: int = 0):
        if not torch.distributed.is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src)
        return obj[0]

    def backward(self, loss, model=None):
        loss.backward()
        if not self.no_sync and self.use_dumb_allreduce:
            _allreduce_chunk_stream(model, world_size=self.world_size, device=self.local_device)

    def clip_gradients(self, model, optimizer, max_norm, error_if_nonfinite=False):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm, norm_type=2.0, error_if_nonfinite=error_if_nonfinite
        )
        return grad_norm

    def save(self, checkpoint_file_location, state, overwrite=True, strict=False):
        """New strategy: First 8 local ranks save!"""
        save_state_dict = {}
        for key, value in state.items():
            if key in ["optimizer", "model"] or "train_dataloader" in key:
                save_state_dict[key] = value.state_dict()  # form states on all devices to prevent deadlock!
            else:
                save_state_dict[key] = value

        if self.rank < self.local_devices:  # first 8 ranks!!
            checkpoint_folder = os.path.dirname(checkpoint_file_location)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            checkpoint_file = f"{checkpoint_file_location}_{self.local_rank}.pth"
            if os.path.exists(checkpoint_file) and not overwrite:
                raise ValueError(f"Checkpoint {checkpoint_file} already exists")
            elif os.path.exists(checkpoint_file):
                print(f"Checkpoint {checkpoint_file} overwritten with new version.")
            torch.save(save_state_dict, checkpoint_file)
            if self.verbose:
                print(f"Rank {self.rank} - local {self.local_rank} saved {checkpoint_file}.")

    def load(self, checkpoint_file_location, state, strict=False):
        checkpoint_file = f"{checkpoint_file_location}_{self.local_rank}.pth"
        if self.verbose:
            print(f"Rank {self.rank} - local {self.local_rank} loaded {checkpoint_file}.")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"), weights_only=False)

        # Track what was loaded for debugging
        loaded_keys = []
        skipped_keys = []

        for key, value in checkpoint.items():
            if key in ["optimizer", "model"] or "train_dataloader" in key:
                try:
                    state[key].load_state_dict(value)
                    loaded_keys.append(f"{key} (state_dict)")
                except Exception as e:
                    print(f"WARNING: Failed to load {key}: {e}")
                    skipped_keys.append(key)
            elif key in ["tokenizer", "val_dataloader", "lr_scheduler", "flop_breakdown"]:
                # Skip objects that don't need to be restored or are recreated fresh
                skipped_keys.append(key)
            else:
                state[key] = value
                loaded_keys.append(key)

        if self.verbose or self.rank == 0:
            print(f"Checkpoint loaded. Keys restored: {len(loaded_keys)}, skipped: {len(skipped_keys)}")
            if skipped_keys:
                print(f"  Skipped keys: {skipped_keys}")

    @contextmanager
    def no_backward_sync(self, model, enabled=True):
        try:
            context = self._ddp_model_ref.no_sync() if enabled and not self.use_dumb_allreduce else nullcontext()
            with context:
                self.no_sync = bool(enabled)
                yield
        finally:
            self.no_sync = False

    def get_prefix_for_checkpoint(self):
        return "checkpoints-sane-ddp"

    @contextmanager
    def init_module(self, empty_init=False):
        yield None

    def seed_everything(self, seed: Optional[int] = None, workers: bool = False):
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        if seed is None:
            env_seed = os.environ.get("PL_GLOBAL_SEED")
            if env_seed is None:
                seed = 0
                self.print(f"Warning: No seed found, seed set to {seed}")
            else:
                try:
                    seed = int(env_seed)
                except ValueError:
                    seed = 0
                    self.print(f"Warning: Invalid seed found: {repr(env_seed)}, seed set to {seed}")
        elif not isinstance(seed, int):
            seed = int(seed)

        if not (min_seed_value <= seed <= max_seed_value):
            self.print(f"Warning: {seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
            seed = 0

        # print(rank_prefixed_message(f"Seed set to {seed}", self.global_rank))
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

        return seed


class _FabricModule(torch.nn.Module):
    def __init__(self, forward_module, precision, original_module=None) -> None:
        """The FabricModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            forward_module: The module to wrap the ``forward`` method on.
            strategy: Reference to the strategy for handling precision etc.
            original_module: The original, unmodified module as passed into the
                :meth:`lightning.fabric.fabric.Fabric.setup` method. This is needed when attribute lookup
                on this wrapper should pass through to the original module.

        """
        super().__init__()
        self._forward_module = forward_module
        self._original_module = original_module or forward_module
        self.precision = precision

        class AmpSettings(TypedDict):
            device_type: str
            dtype: torch.dtype
            enabled: bool

        self.amp_settings: AmpSettings = {
            "device_type": "cuda",
            "enabled": "mixed" in precision,
            "dtype": torch.bfloat16 if "bf16" in precision else torch.float16,
        }
        self._fabric_module_initialized = True

    @property
    def module(self):
        return self._original_module or self._forward_module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward method."""
        with torch.autocast(**self.amp_settings):
            output = self._forward_module(*args, **kwargs)
        return output

    def __getattr__(self, item: Any) -> Any:
        try:
            # __getattr__ gets called as a last resort if the attribute does not exist
            # call nn.Module's implementation first
            return super().__getattr__(item)
        except AttributeError:
            # If the attribute is not available on the _FabricModule wrapper, redirect to the wrapped nn.Module
            original_module = super().__getattr__("_original_module")
            attr = getattr(original_module, item)
            return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if not getattr(self, "_fabric_module_initialized", False):
            super().__setattr__(name, value)
            return

        original_has_attr = hasattr(self._original_module, name)
        fabric_has_attr = name in dir(self)

        if not (original_has_attr or fabric_has_attr):
            setattr(self._original_module, name, value)
            return

        if original_has_attr:
            setattr(self._original_module, name, value)
        if fabric_has_attr:
            super().__setattr__(name, value)


# Torch internal API compatibility: _reload_python_module_in_subproc was removed/renamed in newer torch versions.
try:
    from torch._inductor.codecache import _reload_python_module, _reload_python_module_in_subproc, ModuleType  # type: ignore[attr-defined]
except Exception:
    from torch._inductor.codecache import _reload_python_module  # type: ignore
    try:
        from torch._inductor.codecache import ModuleType  # type: ignore[attr-defined]
    except Exception:
        from types import ModuleType  # fallback to stdlib definition

    def _reload_python_module_in_subproc(key, path):
        # Fallback: reload in-process if subproc helper is unavailable in this torch version.
        return _reload_python_module(key, path)


def load_by_key_path_with_retry(
    cls,
    key: str,
    path: str,
    linemap: Optional[list[tuple[int, str]]] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> ModuleType:
    if linemap is None:
        linemap = []
    if key not in cls.cache:
        # Only retry the module load operation
        for attempt in range(10):
            try:
                mod = _reload_python_module(key, path)
                cls.cache.setdefault(key, mod)
                break
            except OSError:
                print("CACHE LOAD FAILURE")
                if attempt == 9:
                    raise
                time.sleep(min(0.1 * 2**attempt, 8.0))

        cls.linemaps[path] = list(zip(*linemap))
        if attrs is not None:
            for k, v in attrs.items():
                setattr(mod, k, v)
        if not (linemap or attrs):
            mod._reload_in_subproc = partial(_reload_python_module_in_subproc, key, path)  # type: ignore

    return cls.cache[key]


# original:
def load_by_key_path(
    cls,
    key: str,
    path: str,
    linemap: Optional[list[tuple[int, str]]] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> ModuleType:
    if linemap is None:
        linemap = []
    if key not in cls.cache:
        mod = _reload_python_module(key, path)

        # another thread might set this first
        cls.cache.setdefault(key, mod)
        # unzip into separate lines/nodes lists
        cls.linemaps[path] = list(zip(*linemap))

        if attrs is not None:
            for k, v in attrs.items():
                setattr(mod, k, v)

        if not (linemap or attrs):
            mod._reload_in_subproc = partial(_reload_python_module_in_subproc, key, path)  # type: ignore

    return cls.cache[key]
