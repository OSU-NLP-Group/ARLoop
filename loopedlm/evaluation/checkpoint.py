from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from loopedlm import create_config
from loopedlm.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: torch.nn.Module
    tokenizer: Tokenizer
    config: Any
    checkpoint_path: Path
    compute_dtype: torch.dtype


def _enable_vendored_fla() -> None:
    vendor = PROJECT_ROOT / "third_party" / "flash-linear-attention"
    if (vendor / "fla").is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def _resolve_input(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


def _resolve_checkpoint(path: str | Path) -> Path:
    candidate = _resolve_input(path)
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")

    files = sorted(
        item
        for item in candidate.iterdir()
        if item.is_file() and item.suffix in {"", ".pt", ".pth", ".ckpt"}
    )
    if len(files) != 1:
        raise ValueError(
            f"Checkpoint directory must contain exactly one checkpoint file; "
            f"found {len(files)} in {candidate}"
        )
    return files[0]


def strip_checkpoint_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove wrappers added by DDP and torch.compile, including nested wrappers."""
    prefixes = ("module.", "_forward_module.", "_orig_mod.")
    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        if key in cleaned:
            raise ValueError(f"Checkpoint keys collide after prefix removal: {original_key!r} -> {key!r}")
        cleaned[key] = value
    return cleaned


def _extract_state_dict(checkpoint: Any, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict) or not all(isinstance(key, str) for key in state_dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a model state dict")
    tensor_state = {key: value for key, value in state_dict.items() if isinstance(value, torch.Tensor)}
    if not tensor_state:
        raise ValueError(f"Checkpoint at {checkpoint_path} contains no model tensors")
    return strip_checkpoint_prefixes(tensor_state)


def _parse_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower().replace("torch.", "")
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if normalized not in choices:
        raise ValueError(f"Unsupported dtype {dtype!r}; use auto, bf16, fp16, or fp32")
    result = choices[normalized]
    if device.type == "cpu" and result == torch.float16:
        raise ValueError("fp16 evaluation is not supported on CPU")
    return result


def load_checkpoint(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "auto",
    strict: bool = True,
    compile_model: bool = False,
) -> LoadedCheckpoint:
    """Construct a model from a training YAML and load its model weights.

    The checkpoint may be a raw state dict or a full Lightning Fabric training
    checkpoint with a ``model`` entry. Inputs are read-only; this function does
    not create or modify checkpoint/config/tokenizer files.
    """
    _enable_vendored_fla()
    config_file = _resolve_input(config_path)
    checkpoint_file = _resolve_checkpoint(checkpoint_path)
    with config_file.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {config_file}")

    model_name = raw_config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Training config is missing model_name: {config_file}")
    model_overwrite = raw_config.get("model_overwrite") or {}
    if not isinstance(model_overwrite, dict):
        raise ValueError("model_overwrite must be a mapping")

    tokenizer_value = tokenizer_path if tokenizer_path is not None else raw_config.get("tokenizer_path")
    if not tokenizer_value:
        raise ValueError("Tokenizer path is absent; pass --tokenizer or add tokenizer_path to the config")
    tokenizer_dir = _resolve_input(tokenizer_value)
    tokenizer = Tokenizer(tokenizer_dir)

    model_config = create_config(model_name, **model_overwrite)
    model_config.ensure_vocab_size(len(tokenizer))
    model = model_config.construct_model()

    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, checkpoint_file)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    required_missing = sorted(set(missing))
    if strict and (required_missing or unexpected):
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={required_missing[:12]}, unexpected={sorted(unexpected)[:12]}"
        )

    target_device = torch.device(device)
    target_dtype = _parse_dtype(dtype, target_device)
    # Preserve checkpoint parameter dtypes and use autocast in the adapter. In
    # particular, Module.to(dtype=...) would also destroy the complex RoPE
    # frequency buffer by casting it to a real low-precision dtype.
    model = model.to(device=target_device)
    model.eval()
    if compile_model:
        model = torch.compile(model, dynamic=False)
    return LoadedCheckpoint(model, tokenizer, model_config, checkpoint_file, target_dtype)
