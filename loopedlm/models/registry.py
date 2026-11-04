import importlib
from pathlib import Path

configs: list[dict] = []
name_to_config: dict[str, dict] = {}


def register_config(name: str, config: dict) -> dict:
    """Register a model configuration."""
    if name in name_to_config:
        raise ValueError(f"Duplicate model configuration: {name}")
    configs.append(config)
    name_to_config[name] = config
    return config


def _load_configs():
    """Auto-discover and load all configs from loopedlm/configs/"""
    configs_path = Path(__file__).parent.parent / "configs"
    if not configs_path.exists():
        return
    for subdir in sorted(configs_path.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        for config_file in sorted(subdir.glob("*.py")):
            if config_file.name.startswith("_"):
                continue
            module_name = f"loopedlm.configs.{subdir.name}.{config_file.stem}"
            module = importlib.import_module(module_name)
            for conf in getattr(module, "configs", ()):
                register_config(conf["name"], conf)



_load_configs()
