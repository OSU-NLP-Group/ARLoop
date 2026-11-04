# type: ignore
from abc import ABC, abstractmethod
from typing import Protocol


class LRScheduler(ABC):
    """
    Base class for learning rate schedulers.
    Handles warmup and cooldown phases, delegates main decay logic to subclasses.
    """

    def __init__(
        self,
        base_lr: float,
        min_lr: float,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        max_steps: int = 0,
    ):
        """
        Args:
            base_lr: Base learning rate (peak LR after warmup)
            min_lr: Minimum learning rate (for cooldown and decay)
            warmup_steps: Number of steps for linear warmup from 0 to base_lr
            cooldown_steps: Number of steps for linear cooldown from current LR to min_lr
            max_steps: Maximum number of training steps
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        """
        Get the learning rate for the given step.
        Handles warmup, main decay, and cooldown phases.

        Args:
            step: Current training step

        Returns:
            Learning rate for this step
        """
        # Warmup phase: linear ramp from 0 to base_lr
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps

        # Cooldown phase: linear ramp from base_lr to 0, then clamp to min_lr
        # This matches the original implementation: base_lr * (max_steps - step) / cooldown_steps
        if step > (self.max_steps - self.cooldown_steps):
            if self.cooldown_steps == 0:
                return self._get_decay_lr(step)
            return max(self.base_lr * (self.max_steps - step) / self.cooldown_steps, self.min_lr)

        # If step > max_steps, return min learning rate
        if step > self.max_steps:
            return self.min_lr

        # Main decay phase: delegate to subclass
        return self._get_decay_lr(step)

    @abstractmethod
    def _get_decay_lr(self, step: int) -> float:
        """
        Get the learning rate during the main decay phase (between warmup and cooldown).
        Subclasses implement the specific decay schedule.

        Args:
            step: Current training step (guaranteed to be >= warmup_steps and <= max_steps - cooldown_steps)

        Returns:
            Learning rate for this step
        """
        pass
