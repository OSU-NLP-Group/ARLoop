from loopedlm.models.prenorm.config import PreNormConfig


def __getattr__(name):
    if name == "PreNorm":
        from loopedlm.models.prenorm.prenorm import PreNorm

        return PreNorm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PreNormConfig", "PreNorm"]
