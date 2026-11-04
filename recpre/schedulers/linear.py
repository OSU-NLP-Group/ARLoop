# type: ignore
from .base import LRScheduler


class LinearLRScheduler(LRScheduler):
    """
    Linear learning rate decay scheduler.
    Linearly decays from base_lr to min_lr over the decay period.
    """

    def _get_decay_lr(self, step: int) -> float:
        """
        Linear decay: lr = base_lr - decay_ratio * (base_lr - min_lr)
        where decay_ratio goes from 0 to 1 over the decay period.
        """
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1, f"decay_ratio should be in [0, 1], got {decay_ratio}"
        return self.base_lr - decay_ratio * (self.base_lr - self.min_lr)
