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
