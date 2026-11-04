# type: ignore
from typing import Protocol
from .base import LRScheduler
from .linear import LinearLRScheduler
from .cosine import CosineLRScheduler
from .constant import ConstantLRScheduler


def get_lr_scheduler(
    schedule_type: str,
    base_lr: float,
    min_lr: float,
    warmup_steps: int = 0,
    cooldown_steps: int = 0,
    max_steps: int = 0,
) -> LRScheduler:
    """
    Factory function to create a learning rate scheduler.

    Args:
        schedule_type: Type of scheduler ("linear", "cosine", "constant", or "trapezoid")
        base_lr: Base learning rate (peak LR after warmup)
        min_lr: Minimum learning rate (for cooldown and decay)
        warmup_steps: Number of steps for linear warmup from 0 to base_lr
        cooldown_steps: Number of steps for linear cooldown from current LR to min_lr
        max_steps: Maximum number of training steps

    Returns:
        An LRScheduler instance

    Raises:
        ValueError: If schedule_type is not supported
    """
    if schedule_type == "linear":
        return LinearLRScheduler(
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            max_steps=max_steps,
        )
    elif schedule_type == "cosine":
        return CosineLRScheduler(
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            max_steps=max_steps,
        )
    elif schedule_type in ["constant", "trapezoid"]:
        return ConstantLRScheduler(
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            max_steps=max_steps,
        )
    else:
        raise ValueError(f"Unsupported lr_schedule: {schedule_type}")


# Export all scheduler classes
__all__ = [
    "LRScheduler",
    "LinearLRScheduler",
    "CosineLRScheduler",
    "ConstantLRScheduler",
    "get_lr_scheduler",
]
