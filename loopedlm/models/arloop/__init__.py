from loopedlm.models.arloop.config import ARLoopConfig
from loopedlm.models.arloop.init import ARLoopInit


def __getattr__(name):
    if name == "ARLoop":
        from loopedlm.models.arloop.arloop import ARLoop

        return ARLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ARLoopConfig", "ARLoop", "ARLoopInit"]
