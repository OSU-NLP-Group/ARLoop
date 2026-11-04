"""Restore training weights, AdamW, RNG, and packed data position."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import torch
import time

if TYPE_CHECKING:
    from torch import nn
    from recpre.utils import LightningFabric, SimpleFabric
    Fabric = LightningFabric | SimpleFabric

def _strip_checkpoint_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "_forward_module.")
    cleaned = {}
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        cleaned[key] = value
    return cleaned


def _load_model_weights_from_checkpoint(model: nn.Module, checkpoint_path: Path, fabric: Fabric) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint at {checkpoint_path} does not contain a model state dict")
    target = getattr(model, "_forward_module", model)
    target = getattr(target, "module", target)
    target_state = target.state_dict()
    cleaned = _strip_checkpoint_prefixes(state_dict)
    compatible = {}
    skipped = []
    for key, value in cleaned.items():
        if key not in target_state:
            skipped.append(key)
            continue
        if target_state[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible[key] = value
    missing, unexpected = target.load_state_dict(compatible, strict=False)
    fabric.print(
        f"Loaded model-only checkpoint from {checkpoint_path}: "
        f"{len(compatible)} tensors, {len(missing)} missing, {len(unexpected)} unexpected, {len(skipped)} skipped."
    )
    if skipped:
        fabric.print(f"  skipped first few keys: {skipped[:8]}")
    if missing:
        raise RuntimeError(f"model-only checkpoint missing required keys, first few: {missing[:8]}")


def load_checkpoint(
    fabric,
    state,
    out_dir,
    run_name,
    model_checkpoint,
    resume=True,
    resume_checkpoint_path=None,
    resume_model_only=False,
    resume_data_state=True,
):
    resume_ckpt = None
    t0 = time.time()
    if resume:
        fabric.print("-------------------- Model Load triggered ------------------------------")

        if resume_checkpoint_path:
            p = Path(resume_checkpoint_path)
            if p.exists():
                resume_ckpt = p
            else:
                resume_ckpt = p.parent / p.stem.rsplit("_", 1)[0] if "_" in p.stem else p.with_suffix("")
            fabric.print(f"Using checkpoint: {resume_ckpt}")
        else:
            base_for_glob = Path(out_dir) / fabric.get_prefix_for_checkpoint()
            fabric.print(f"Globbing for checkpoint files in {base_for_glob}")
            # DDPStrategy saves without extension, others may add rank suffix like {name}_0.pth
            if fabric.strategy_name == "axonn_tp":
                ckpt_pattern = f"*/*-{run_name}_*.pth"
            elif fabric.strategy_name == "DDPStrategy":
                # Try without extension first (Lightning Fabric default), then with .pth
                ckpt_pattern = f"step-*-{run_name}"
            else:
                ckpt_pattern = f"*-{run_name}_*.pth"
            ckpt_paths = list(base_for_glob.glob(ckpt_pattern))
            # Also try with .pth extension for DDPStrategy
            if len(ckpt_paths) == 0 and fabric.strategy_name == "DDPStrategy":
                ckpt_paths = list(base_for_glob.glob(f"step-*-{run_name}.pth"))
            fabric.print(f"Found {len(ckpt_paths)} checkpoint(s) matching pattern")
            if len(ckpt_paths) == 0:
                fabric.print(f"No checkpoint found in {out_dir} to resume from.")
            else:
                # Extract step number from filename like "step-00005000-{run_name}"
                def extract_step(p):
                    name = p.stem if p.suffix else p.name  # handle with/without extension
                    # Format: step-XXXXXXXX-{run_name}
                    parts = name.split("-")
                    if len(parts) >= 2:
                        try:
                            return int(parts[1])
                        except ValueError:
                            return 0
                    return 0
                resume_ckpt = max(ckpt_paths, key=extract_step)
                fabric.print(f"Selected checkpoint: {resume_ckpt}")
                # For DDPStrategy, use the path as-is (fabric.load handles both with/without extension)
                if fabric.strategy_name != "DDPStrategy":
                    filename, directory = str(resume_ckpt.name), resume_ckpt.parents[0]
                    filename = filename[filename.find("step") :]
                    filename = filename.split(f"-{run_name}_")[0] + f"-{run_name}"  # split off rank info and .pth
                    if fabric.strategy_name == "axonn_tp":
                        directory = Path(out_dir) / fabric.get_prefix_for_checkpoint()
                    resume_ckpt = directory / filename

        if resume_ckpt is not None:
            fabric.print(f"Resuming training from {resume_ckpt}")
            pre_load_step = state["microbatch_step"]

            train_dataloader = state["train_dataloader"]
            val_dataloader = state.get("val_dataloader")
            # Don't restore lr_scheduler from checkpoint — it depends on current config's max_steps
            lr_scheduler = state.pop("lr_scheduler", None)
            if not resume_data_state:
                state.pop("train_dataloader", None)
                state.pop("val_dataloader", None)

            ckpt_path = Path(resume_ckpt)
            if ckpt_path.is_dir():
                fabric.print(f"WARNING: resume_ckpt is a directory: {resume_ckpt}")
                possible_files = list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.pth")) + list(ckpt_path.glob("*.ckpt"))
                if possible_files:
                    ckpt_path = possible_files[0]
                    fabric.print(f"Using checkpoint file: {ckpt_path}")

            if resume_model_only:
                _load_model_weights_from_checkpoint(state["model"], ckpt_path, fabric)
                if lr_scheduler is not None:
                    state["lr_scheduler"] = lr_scheduler
                state["train_dataloader"] = train_dataloader
                state["val_dataloader"] = val_dataloader
                fabric.print("Model-only resume complete; optimizer, scheduler, dataloader, and step counters kept from current run.")
                return

            try:
                checkpoint_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                fabric.print(f"Loaded raw checkpoint, keys: {list(checkpoint_raw.keys())}")
                if "train_dataloader" in checkpoint_raw:
                    _raw_dl = checkpoint_raw['train_dataloader']
                    fabric.print(f"  train_dataloader type: {type(_raw_dl)}")
                    if isinstance(_raw_dl, dict):
                        fabric.print(f"  train_dataloader keys: {list(_raw_dl.keys())}")
                    else:
                        _raw_ds = getattr(_raw_dl, 'dataset', None)
                        _raw_ds_state = getattr(_raw_ds, '_state', None) if _raw_ds is not None else None
                        if _raw_ds_state is not None:
                            fabric.print(f"  pickled dataset _state: file_idx={_raw_ds_state.get('file_idx')}, "
                                         f"row_group_idx={_raw_ds_state.get('row_group_idx')}, "
                                         f"buffer_len={len(_raw_ds_state.get('buffer', []))}")
                        else:
                            fabric.print(f"  pickled object has no dataset._state")
            except Exception as e:
                fabric.print(f"WARNING: Failed to load raw checkpoint: {e}")
                checkpoint_raw = {}

            _original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: _original_torch_load(*args, weights_only=False, **{k: v for k, v in kwargs.items() if k != 'weights_only'})
            try:
                fabric.load(resume_ckpt, state, strict=False)
            finally:
                torch.load = _original_torch_load

            if lr_scheduler is not None:
                state["lr_scheduler"] = lr_scheduler
                fabric.print(f"LR scheduler NOT restored from checkpoint (using current config's max_steps)")

            # Restore dataset state only (not full dataloader state, which causes double-positioning)
            state["train_dataloader"] = train_dataloader
            state["val_dataloader"] = val_dataloader
            if not resume_data_state:
                fabric.print("Skipping dataloader state restore because resume_data_state=False")
            elif "train_dataloader" in checkpoint_raw:
                saved_dl_state = checkpoint_raw["train_dataloader"]
                if isinstance(saved_dl_state, dict) and "dataset_state" in saved_dl_state:
                    ds_state = saved_dl_state["dataset_state"]
                    fabric.print(f"Restoring dataset state from checkpoint (dict path):")
                    fabric.print(f"  file_idx={ds_state.get('file_idx')}, "
                                 f"row_group_idx={ds_state.get('row_group_idx')}, "
                                 f"row_idx={ds_state.get('row_idx')}")
                    train_dataloader.dataset.load_state_dict(ds_state)
                elif not isinstance(saved_dl_state, dict) and hasattr(saved_dl_state, 'dataset'):
                    saved_ds = saved_dl_state.dataset
                    if hasattr(saved_ds, '_state') and saved_ds._state:
                        train_dataloader.dataset._state = saved_ds._state
                        fabric.print(f"Restoring dataset state from checkpoint (pickled path): "
                                     f"file_idx={saved_ds._state.get('file_idx')}, "
                                     f"row_group_idx={saved_ds._state.get('row_group_idx')}")
                    else:
                        fabric.print("WARNING: Pickled dataloader has no dataset state")
                else:
                    fabric.print(f"WARNING: Cannot extract dataset state from train_dataloader "
                                 f"(type={type(saved_dl_state).__name__}), data restarts from beginning")
            else:
                fabric.print("WARNING: No train_dataloader in checkpoint, data restarts from beginning")


            if "rng_state_torch" in checkpoint_raw and checkpoint_raw["rng_state_torch"] is not None:
                torch.set_rng_state(checkpoint_raw["rng_state_torch"])
                fabric.print(f"Restored PyTorch CPU RNG state")
            if "rng_state_cuda" in checkpoint_raw and checkpoint_raw["rng_state_cuda"] is not None:
                torch.cuda.set_rng_state(checkpoint_raw["rng_state_cuda"], device=fabric.device)
                fabric.print(f"Restored PyTorch CUDA RNG state")
            if "rng_state_numpy" in checkpoint_raw and checkpoint_raw["rng_state_numpy"] is not None:
                import numpy as np
                np.random.set_state(checkpoint_raw["rng_state_numpy"])
                fabric.print(f"Restored NumPy RNG state")
            if "rng_state_python" in checkpoint_raw and checkpoint_raw["rng_state_python"] is not None:
                import random
                random.setstate(checkpoint_raw["rng_state_python"])
                fabric.print(f"Restored Python RNG state")
            state["rng_state_torch"] = checkpoint_raw.get("rng_state_torch")
            state["rng_state_cuda"] = checkpoint_raw.get("rng_state_cuda")
            state["rng_state_numpy"] = checkpoint_raw.get("rng_state_numpy")
            state["rng_state_python"] = checkpoint_raw.get("rng_state_python")
            if state["microbatch_step"] == pre_load_step and state["microbatch_step"] == 0:
                fabric.print(f"WARNING: microbatch_step is still 0 after loading - checkpoint may not have loaded correctly!")

    if resume_ckpt is None and model_checkpoint is not None:
        fabric.print("-------------------- Pretrained Checkpoint Load triggered ------------------------------")
        fabric.print(f"Loaded full checkpoint (including optim and data state) from {model_checkpoint}")
        train_dataloader = state["train_dataloader"]

        lr_scheduler = state.pop("lr_scheduler", None)
        checkpoint_raw = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
        if not resume_data_state:
            state.pop("train_dataloader", None)

        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(*args, weights_only=False, **{k: v for k, v in kwargs.items() if k != 'weights_only'})
        try:
            fabric.load(model_checkpoint, state, strict=False)
        finally:
            torch.load = _original_torch_load
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler
            fabric.print(f"LR scheduler NOT restored from checkpoint (using current config's max_steps)")

        if not resume_data_state:
            fabric.print("Skipping dataloader state restore because resume_data_state=False")
        elif "train_dataloader" in checkpoint_raw:
            fabric.print("Restoring dataloader state from checkpoint...")
            dl_state = checkpoint_raw["train_dataloader"]
            if isinstance(dl_state, dict):
                dl_state_modified = dl_state.copy()
                dl_state_modified["_num_yielded"] = 0
                dl_state_modified["_sampler_iter_yielded"] = 0
                dl_state_modified["_iterator_finished"] = False
                train_dataloader.load_state_dict(dl_state_modified)
                fabric.print(f"  Dataloader state restored (with _num_yielded=0)")
            else:
                train_dataloader.load_state_dict(dl_state)
                fabric.print(f"  Restored full dataloader state")
        state["train_dataloader"] = train_dataloader

    if resume_ckpt or model_checkpoint:
        fabric.print(f"Loaded state is from step {state['microbatch_step']}")
        fabric.print(f"Loaded optimizer_step: {state.get('optimizer_step', 'NOT FOUND')}")

        opt = state["optimizer"]
        opt_state = opt.state if hasattr(opt, "state") else {}
        params_with_state = sum(1 for s in opt_state.values() if s)
        fabric.print(f"Optimizer has state for {params_with_state} parameters")

        total_m_norm = 0.0
        total_v_norm = 0.0
        adamw_steps = []
        for s in opt_state.values():
            if s:
                if 'm' in s:
                    total_m_norm += s['m'].float().norm().item()
                if 'v' in s:
                    total_v_norm += s['v'].float().norm().item()
                if 'step' in s:
                    adamw_steps.append(s['step'])

        fabric.print(f"  Total momentum (m) norm: {total_m_norm:.4f}")
        fabric.print(f"  Total second moment (v) norm: {total_v_norm:.4f}")
        if adamw_steps:
            fabric.print(f"  AdamW step counts: min={min(adamw_steps)}, max={max(adamw_steps)}")

        if total_m_norm < 1e-6 and total_v_norm < 1e-6 and state['microbatch_step'] > 100:
            fabric.print("  WARNING: Optimizer state appears empty despite training having progressed!")
            fabric.print("  This will cause training to restart from scratch optimizer-wise!")

        fabric.print(f"{time.ctime()[:-5]} : Time to load ckpt state: {time.time() - t0:.02f} seconds.")
        fabric.print("-------------------- Checkpoint loaded    ------------------------------")
    else:
        fabric.print("-------------------- No Checkpoint loaded ------------------------------")
    return resume_ckpt
