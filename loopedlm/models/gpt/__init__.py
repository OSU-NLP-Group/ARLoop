from loopedlm.models.gpt.config import GPTConfig
from loopedlm.models.gpt.init import GPTInit


def __getattr__(name):
    if name == "GPT":
        from loopedlm.models.gpt.gpt import GPT
        return GPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GPTConfig", "GPT", "GPTInit"]
