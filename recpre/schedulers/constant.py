# type: ignore
from .base import LRScheduler


class ConstantLRScheduler(LRScheduler):
    """
    Constant learning rate scheduler.
    Maintains base_lr throughout the decay period (no decay).
    Also handles "trapezoid" schedule which is the same as constant.
    """

    def _get_decay_lr(self, step: int) -> float:
        """
        Constant decay: always return base_lr (no decay).
        """
        return self.base_lr
