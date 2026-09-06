from typing import override
from abc import ABC, abstractmethod
import numpy as np

class Schedule(ABC):
    """Learning-rate schedule: maps an epoch index to the learning rate to use."""

    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        pass

    @property
    @abstractmethod
    def state(self) -> dict[str, int|float]:
        pass

    @abstractmethod
    def set_state(self, dict) -> None:
        pass

class CosineSchedule(Schedule):
    """Cosine annealing from `max_lr` down to `min_lr` over `total_epochs`, then flat
    at `min_lr`."""

    def __init__(self, max_lr: float, min_lr: float, total_epochs: int):
        self.max_lr:float = max_lr
        self.min_lr:float = min_lr
        self.total_epochs:int = total_epochs

    @override
    def get_lr(self, epoch: int) -> float:
        # Boundary condition
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing formula
        cosine_factor:np.ndarray = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return float(self.min_lr + (self.max_lr - self.min_lr) * cosine_factor)

    @property
    @override
    def state(self) -> dict[str, int|float]:
        return {'max_lr': self.max_lr, 'min_lr': self.min_lr, 'total_epochs': self.total_epochs}
    
    @override
    def set_state(self, dict) -> None:
        self.max_lr, self.min_lr, self.total_epochs = dict['max_lr'], dict['min_lr'], dict['total_epochs']

class CosineRestartSchedule(Schedule):
    """Cosine annealing restarted at each entry of `boundaries`, then flat at `min_lr`.

    Each cycle runs the full `max_lr` -> `min_lr` sweep, so the epoch immediately
    before every boundary is trained at (essentially) `min_lr`. That is what makes
    an intermediate checkpoint comparable to a run of that budget: with a single
    cosine spanning the whole run, a checkpoint harvested early would still sit at
    a high learning rate and could not stand in for a completed shorter run.

    `boundaries` holds the epoch counts at which a cycle ends (exclusive), e.g.
    `[64, 128, 256]` for checkpoints taken after epochs 63, 127 and 255.
    """

    def __init__(self, max_lr: float, min_lr: float, boundaries: list[int]):
        cuts = sorted({int(b) for b in boundaries})
        assert len(cuts) > 0, "at least one boundary is required"
        assert cuts[0] > 0, "boundaries are epoch counts, so they must be positive"

        self.max_lr:float = max_lr
        self.min_lr:float = min_lr
        self.boundaries:list[int] = cuts

    @property
    def total_epochs(self) -> int:
        """Epoch at which the last cycle ends."""
        return self.boundaries[-1]

    @override
    def get_lr(self, epoch: int) -> float:
        start = 0
        for end in self.boundaries:
            if epoch < end:
                cosine_factor = (1 + np.cos(np.pi * (epoch - start) / (end - start))) / 2
                return float(self.min_lr + (self.max_lr - self.min_lr) * cosine_factor)
            start = end

        # Past the last boundary the schedule holds at its floor.
        return self.min_lr

    @property
    @override
    def state(self) -> dict[str, int|float]:
        return {'max_lr': self.max_lr, 'min_lr': self.min_lr, 'boundaries': self.boundaries}

    @override
    def set_state(self, dict) -> None:
        self.max_lr, self.min_lr, self.boundaries = dict['max_lr'], dict['min_lr'], list(dict['boundaries'])
