# type: ignore
import math
from .base import LRScheduler


class CosineLRScheduler(LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    Uses cosine annealing to decay from base_lr to min_lr over the decay period.
    """

    def _get_decay_lr(self, step: int) -> float:
        """
        Cosine decay: lr = min_lr + coeff * (base_lr - min_lr)
        where coeff = 0.5 * (1 + cos(π * decay_ratio))
        and decay_ratio goes from 0 to 1 over the decay period.
        """
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1, f"decay_ratio should be in [0, 1], got {decay_ratio}"
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.base_lr - self.min_lr)
