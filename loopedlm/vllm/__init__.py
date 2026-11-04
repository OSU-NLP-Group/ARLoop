"""vLLM model and checkpoint-loader registration."""

_loader_registered = False


def register() -> None:
    """Register ARLoop models and loaders in every vLLM worker process."""
    global _loader_registered
    from vllm import ModelRegistry  # noqa: PLC0415
    from vllm.model_executor.model_loader import register_model_loader
    from .loader import ARLoopModelLoader

    if not _loader_registered:
        register_model_loader("arloop")(ARLoopModelLoader)
        _loader_registered = True

    for architecture in ("LargeScaleARLoopForCausalLM",):
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                architecture,
                "loopedlm.vllm.largescale_arloop:LargeScaleARLoopForCausalLM",
            )


__all__ = ["register"]
