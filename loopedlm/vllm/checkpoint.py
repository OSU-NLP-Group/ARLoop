"""Shared configuration and checkpoint readers for vLLM loading."""
from __future__ import annotations
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any
import torch
from safetensors.torch import load_file
from loopedlm.models.largescale_arloop.config import LargeScaleARLoopConfig
from loopedlm.models.arloop.config import ARLoopConfig

TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints-DDPStrategy"
    checkpoints = sorted(checkpoint_dir.glob("step-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    return checkpoints[-1]


def load_model_config(
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    LargeScaleARLoopConfig | ARLoopConfig,
]:
    with (run_dir / "model_config.json").open() as handle:
        raw_config = json.load(handle)
    model_class_name = raw_config.get("model_class_name")
    if model_class_name == "LargeScaleARLoop":
        config_class = LargeScaleARLoopConfig
    elif model_class_name == "ARLoop":
        config_class = ARLoopConfig
    else:
        raise ValueError(
            "vLLM loading supports "
            "LargeScaleARLoop, and ARLoop, got "
            f"{model_class_name!r}"
        )
    allowed = {field.name for field in fields(config_class)}
    config = config_class(
        **{key: value for key, value in raw_config.items() if key in allowed}
    )
    return raw_config, config


def build_vllm_config(
    raw: dict[str, Any],
    config: LargeScaleARLoopConfig | ARLoopConfig,
    special_tokens: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate the native dataclass into an AutoConfig-compatible JSON object."""
    special_tokens = special_tokens or {}
    physical_layers = int(config.n_layer)
    loop_count = int(config.loop_count)
    effective_layers = physical_layers * loop_count
    head_dim = int(config.n_embd // config.num_attention_heads)
    is_largescale = config.mlp_class_name == "GatedMLP"
    if not is_largescale:
        if config.mlp_class_name != "BaseMLP" or config.nonlin_name.lower() != "relu2":
            raise ValueError(
                "Standard ARLoop vLLM loading currently requires BaseMLP + ReLU2"
            )

    result = {
        "architectures": ["LargeScaleARLoopForCausalLM"],
        # Using a built-in config type lets AutoConfig parse the directory. The
        # architecture still dispatches to our registered vLLM model.
        "model_type": "llama",
        "hidden_size": int(config.n_embd),
        "intermediate_size": int(config.intermediate_size),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": head_dim,
        "num_hidden_layers": effective_layers,
        "arloop_n_layer": physical_layers,
        "loop_count": loop_count,
        "effective_sublayers": 2 * effective_layers,
        "vocab_size": int(config.vocab_size),
        "padded_vocab_size": int(config.padded_vocab_size),
        "padding_multiple": int(config.padding_multiple),
        "max_position_embeddings": int(config.block_size),
        "rope_theta": float(config.rope_settings.rope_base),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(config.rope_settings.rope_base),
        },
        "rope_condense_ratio": int(config.rope_settings.rope_condense_ratio),
        "rms_norm_eps": float(config.norm_eps),
        "hidden_act": "silu" if is_largescale else "relu2",
        "arloop_mlp_type": "swiglu" if is_largescale else "base_relu2",
        "attention_bias": bool(config.bias),
        "qk_bias": bool(config.qk_bias),
        "qk_norm": bool(config.qk_norm),
        "clip_qkv": config.clip_qkv,
        "tie_word_embeddings": bool(config.tie_embeddings),
        "embedding_scale": float(config.init.embedding_scale),
        "logit_scale": float(config.init.logit_scale),
        "logit_softcap": config.logit_softcap,
        "attnres_block_size": int(config.attnres_block_size),
        "attnres_sliding_window": bool(config.attnres_sliding_window),
        "sw_blocks": config.sw_blocks,
        "use_value_embeddings": bool(config.use_value_embeddings),
        "share_value_embeddings": config.shared_value_embeddings,
        "use_abacus": bool(config.use_abacus),
        "randomize_positions_from": config.randomize_positions_from,
        "torch_dtype": "bfloat16",
        "dtype": "bfloat16",
        "transformers_version": "5.0.0",
        "arloop_model_name": raw.get("name"),
        "arloop_model_class": raw.get("model_class_name"),
    }
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if special_tokens.get(key) is not None:
            result[key] = special_tokens[key]
    return result


def load_checkpoint_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    from recpre.path_utils import PROJECT_ROOT
    profiler_path = str(PROJECT_ROOT / "scripts/train")
    if profiler_path not in sys.path:
        sys.path.insert(0, profiler_path)
    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {checkpoint} does not contain a model state dict")
    return state_dict



def load_hf_source(source: str, revision: str | None = None) -> Path:
    """Resolve a local model directory or download data files only (no remote Python)."""
    local = Path(source).expanduser()
    if local.is_dir():
        if revision is not None:
            raise ValueError("--revision only applies to a Hugging Face repo ID")
        return local.resolve()
    if local.is_absolute() or source.startswith("."):
        raise FileNotFoundError(local)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=source, revision=revision or "main",
        allow_patterns=["config.json", "generation_config.json", *TOKENIZER_FILES,
                        "*.safetensors", "*.safetensors.index.json",
                        "pytorch_model*.bin", "pytorch_model.bin.index.json"],
    ))


def build_hf_vllm_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate the portable ARLoopForCausalLM release configuration."""
    if raw.get("model_type") != "arloop" or raw.get("architectures") != ["ARLoopForCausalLM"]:
        raise ValueError("HF input must be a portable ARLoopForCausalLM release")
    if raw.get("quantization_config") or not raw.get("tie_word_embeddings", True):
        raise ValueError("HF loading requires unquantized, tied-embedding ARLoop weights")
    config = LargeScaleARLoopConfig(
        name="largescale-arloop-hf",
        n_layer=raw["physical_layers"], loop_count=raw["loop_count"],
        n_embd=raw["hidden_size"], intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"], padded_vocab_size=raw["vocab_size"],
        padding_multiple=1, block_size=raw["max_position_embeddings"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        norm_eps=raw.get("rms_norm_eps", 1e-5),
        qk_norm=raw.get("qk_norm", True), bias=False, qk_bias=False,
        tie_embeddings=True, attnres_block_size=raw["attnres_block_size"],
        sw_blocks=raw["sw_blocks"],
        attnres_sliding_window=True,
    )
    result = build_vllm_config({"model_class_name": "LargeScaleARLoop"}, config, raw)
    if raw.get("num_hidden_layers", result["num_hidden_layers"]) != result["num_hidden_layers"]:
        raise ValueError("HF effective depth does not match physical_layers * loop_count")
    if raw.get("head_dim", result["head_dim"]) != result["head_dim"]:
        raise ValueError("HF head_dim does not match hidden_size / num_attention_heads")
    result.update(
        rope_theta=raw.get("rope_theta", 50000.0),
        rope_parameters={"rope_type": "default", "rope_theta": raw.get("rope_theta", 50000.0)},
        rope_condense_ratio=raw.get("rope_condense_ratio", 1),
        embedding_scale=raw.get("embedding_scale", 1.0), logit_scale=1.0,
        logit_softcap=raw.get("logit_softcap"),
    )
    return result


def load_hf_weights(folder: Path) -> dict[str, torch.Tensor]:
    """Read single-file or indexed safetensors/PyTorch weights on CPU."""
    weight_map = None
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = folder / index_name
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            filenames = sorted(set(weight_map.values()))
            break
    else:
        filenames = [next((name for name in ("model.safetensors", "pytorch_model.bin")
                          if (folder / name).is_file()), "")]
    if not filenames or not filenames[0]:
        raise FileNotFoundError(f"No supported HF weights in {folder}")
    state = {}
    for name in filenames:
        path = folder / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Weight shard escapes model directory: {name}")
        shard = (load_file(str(path), device="cpu") if name.endswith(".safetensors")
                 else torch.load(path, map_location="cpu", weights_only=True))
        if state.keys() & shard.keys():
            raise ValueError(f"Duplicate tensors in HF shard {name}")
        if weight_map is not None and any(weight_map.get(key) != name for key in shard):
            raise ValueError(f"HF shard {name} does not match its weight index")
        state.update(shard)
    if weight_map is not None and state.keys() != weight_map.keys():
        raise ValueError("HF weight index contains missing tensors")
    return state
