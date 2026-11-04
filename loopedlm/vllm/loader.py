"""vLLM loader for existing HF weights and single-file native checkpoints."""
from pathlib import Path
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from .checkpoint import load_checkpoint_state, load_hf_weights


class ARLoopModelLoader(BaseModelLoader):
    def download_model(self, model_config):
        # The evaluation entrypoint resolves remote snapshots before engine creation.
        source = model_config.hf_config.arloop_weight_source
        if not Path(source["path"]).exists():
            raise FileNotFoundError(source["path"])

    def load_weights(self, model, model_config):
        source = model_config.hf_config.arloop_weight_source
        if source["kind"] == "native":
            state = load_checkpoint_state(Path(source["path"]))
        elif source["kind"] == "hf":
            state = load_hf_weights(Path(source["path"]))
        else:
            raise ValueError(f"Unsupported ARLoop weight source: {source['kind']}")
        model.load_weights(state.items())
