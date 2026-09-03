from abc import ABC, abstractmethod
from typing import override
from thorcino.tensor import Tensor


class Layer(ABC):
    """Base layer: subclasses implement `forward` plus the `train`/`eval` mode switch.
    The parameter and state accessors default to empty, so a layer without weights
    needs no override."""

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
        """Picklable snapshot of this layer's parameters, keyed by short names."""
        return {}
    
    @override
    def set_state(self, state: dict) -> None:
        pass