from typing import override
from abc import ABC, abstractmethod
import numpy as np

class Schedule(ABC):
    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        pass

    @abstractmethod
    def get_state(self) -> dict[str, int|float]:
        pass


class CosineSchedule(Schedule):
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

    @override
    def get_state(self) -> dict[str, int|float]:
        return {'max_lr': self.max_lr, 'min_lr': self.min_lr, 'total_epochs': self.total_epochs}
