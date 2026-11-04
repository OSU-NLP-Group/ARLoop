from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_evaluation_runtime(*, online: bool = False) -> Path:
    """Set cache paths before importing benchmark packages.

    ``ARLOOP_EVAL_CACHE_DIR`` overrides the default project-local
    ``.runtime/evaluation`` cache directory.
    """
    configured_runtime = os.environ.get("ARLOOP_EVAL_CACHE_DIR")
    runtime = (
        Path(configured_runtime).expanduser().resolve()
        if configured_runtime
        else PROJECT_ROOT / ".runtime" / "evaluation"
    )
    hf_home = runtime / "huggingface"
    xdg_cache = runtime / "xdg-cache"
    for directory in (runtime, hf_home, xdg_cache):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if online:
        # The selected network mode overrides inherited offline flags.
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return runtime
