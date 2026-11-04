"""Prepare metadata for direct HF/native loading without writing model weights."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer
from recpre.path_utils import PROJECT_ROOT
from .checkpoint import (
    build_hf_vllm_config, build_vllm_config, latest_checkpoint,
    load_hf_source, load_model_config,
)


def add_source_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", help="HF repo ID, local HF model, or existing vLLM export")
    group.add_argument("--run-dir", type=Path, help="Native training run with model_config.json")
    parser.add_argument("--checkpoint", type=Path, help="Native checkpoint; defaults to the latest in run-dir")
    parser.add_argument("--revision", help="HF branch, tag, or commit")


def local_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@contextmanager
def model_source(args):
    native = args.run_dir is not None or args.checkpoint is not None
    if native:
        if args.model or args.revision:
            raise ValueError("Native checkpoints cannot be combined with --model or --revision")
        checkpoint = local_path(args.checkpoint) if args.checkpoint else None
        run_dir = local_path(args.run_dir) if args.run_dir else None
        if run_dir is None:
            run_dir = next((p for p in checkpoint.parents if (p / "model_config.json").is_file()), None)
            if run_dir is None:
                raise ValueError("Cannot infer the training run; pass --run-dir")
        checkpoint = checkpoint or latest_checkpoint(run_dir).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Expected a single-file training checkpoint: {checkpoint}")
        raw, model_config = load_model_config(run_dir)
        run_config = json.loads((run_dir / "run_config.json").read_text())
        tokenizer_path = run_config.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError("No tokenizer_path in run_config.json; set it to the training tokenizer location")
        tokenizer_path = local_path(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
        special = {key: getattr(tokenizer, key) for key in ("bos_token_id", "eos_token_id", "pad_token_id")}
        config = build_vllm_config(raw, model_config, special)
        source = {"kind": "native", "path": str(checkpoint)}
    else:
        requested = args.model or "osunlp/ARLoop-1.2B"
        candidate = local_path(requested)
        folder = load_hf_source(str(candidate) if candidate.is_dir() else requested, args.revision)
        raw = json.loads((folder / "config.json").read_text())
        if raw.get("model_type") == "arloop":
            config = build_hf_vllm_config(raw)
        elif raw.get("architectures") == ["LargeScaleARLoopForCausalLM"]:
            config = raw.copy()
        else:
            raise ValueError("Expected an ARLoop HF model or an existing ARLoop vLLM export")
        tokenizer_path = folder
        source = {"kind": "hf", "path": str(folder.resolve())}
    config["arloop_weight_source"] = source
    with TemporaryDirectory(prefix="arloop-vllm-config-") as temporary:
        metadata = Path(temporary)
        (metadata / "config.json").write_text(json.dumps(config))
        # Only metadata is generated; the custom loader opens the original weights.
        if not native and (folder / "generation_config.json").is_file():
            generation = json.loads((folder / "generation_config.json").read_text())
        else:
            generation = {key: config[key] for key in ("bos_token_id", "eos_token_id", "pad_token_id") if key in config}
        (metadata / "generation_config.json").write_text(json.dumps(generation))
        yield {"pretrained": str(metadata), "tokenizer": str(tokenizer_path),
               "load_format": "arloop", "trust_remote_code": False}
