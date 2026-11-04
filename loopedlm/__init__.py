def create_config(name: str, **kwargs):
    """Create a model config by name.

    >>> config = create_config("arloop-small-140m")
    >>> config.block_size = 2048
    >>> model = config.construct_model()
    """
    from loopedlm.models.config import Config
    return Config.from_name(name, **kwargs)


def create_model(name: str, **kwargs):
    """Create a model by name, ready for forward passes.

    >>> model = create_model("arloop-small-140m")
    >>> model = create_model("gpt-small-140m")
    """
    config = create_config(name, **kwargs)
    return config.construct_model()


def from_pretrained(repo_id: str, device="cpu", dtype=None, **kwargs):
    """Load a pretrained model from a HuggingFace repository.

    Expects the repo to contain:
      - config.json: serialized model config (with _class_name field)
      - pytorch_model.bin or model.pt: PyTorch state dict

    >>> model = loopedlm.from_pretrained("your-organization/arloop-checkpoint")
    >>> model = loopedlm.from_pretrained("your-organization/arloop-checkpoint", device="cuda")
    """
    import json
    import torch
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id, "config.json", **kwargs)
    with open(config_path) as f:
        config_dict = json.load(f)

    class_name = config_dict.pop("_class_name", "GPTConfig")
    config_dict.pop("rope_settings", None)

    # Lazy import to avoid circular deps — Config.from_name handles dispatch internally
    if class_name == "ARLoopConfig":
        from loopedlm.models.arloop.config import ARLoopConfig as config_cls
    elif class_name == "LargeScaleARLoopConfig":
        from loopedlm.models.largescale_arloop.config import (
            LargeScaleARLoopConfig as config_cls,
        )
    elif class_name == "PreNormConfig":
        from loopedlm.models.prenorm.config import PreNormConfig as config_cls
    elif class_name == "GPTConfig":
        from loopedlm.models.gpt.config import GPTConfig as config_cls
    else:
        raise ValueError(f"Unsupported model config class: {class_name!r}")

    config = config_cls(**config_dict)

    model = config.construct_model()

    for weight_file in ["pytorch_model.bin", "model.pt"]:
        try:
            weights_path = hf_hub_download(repo_id, weight_file, **kwargs)
            break
        except Exception:
            continue
    else:
        raise FileNotFoundError(f"No weights found in {repo_id} (tried pytorch_model.bin, model.pt)")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device=device)

    return model


def save_pretrained(model, path: str):
    """Save a model's config and weights for later loading with from_pretrained.

    >>> loopedlm.save_pretrained(model, "./my-model")
    # Then upload the directory to HuggingFace
    """
    import json
    import torch
    from pathlib import Path
    from dataclasses import asdict

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(model.config)
    config_dict["_class_name"] = model.config.__class__.__name__
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    torch.save(model.state_dict(), out_dir / "model.pt")


def __getattr__(name):
    if name == "GPT":
        from loopedlm.models.gpt.gpt import GPT
        return GPT
    if name == "GPTConfig":
        from loopedlm.models.gpt.config import GPTConfig
        return GPTConfig
    if name == "ARLoop":
        from loopedlm.models.arloop.arloop import ARLoop
        return ARLoop
    if name == "ARLoopConfig":
        from loopedlm.models.arloop.config import ARLoopConfig
        return ARLoopConfig
    if name == "LargeScaleARLoopConfig":
        from loopedlm.models.largescale_arloop.config import (
            LargeScaleARLoopConfig,
        )

        return LargeScaleARLoopConfig
    if name == "PreNorm":
        from loopedlm.models.prenorm.prenorm import PreNorm
        return PreNorm
    if name == "PreNormConfig":
        from loopedlm.models.prenorm.config import PreNormConfig
        return PreNormConfig
    raise AttributeError(f"module 'loopedlm' has no attribute {name!r}")
