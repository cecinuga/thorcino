from abc import ABC, abstractmethod
from typing import override
from thorcino.tensor import Tensor


class Layer(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Compute the layer output."""
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @property
    def parameters(self) -> list[Tensor]:
        return []
    
    @property
    def state(self) -> dict:
        """Return the weights and bias of the layer"""
        return {self.__repr__(): {}}
    
    @override
    def set_state(self, state: dict) -> None:
        pass