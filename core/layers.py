from abc import ABC, abstractmethod
from pathlib import Path
from typing import override, TYPE_CHECKING
import numpy as np
from core.consts import BIAS_ROLE, WEIGHTS_ROLE
from core.tensor import Tensor

if TYPE_CHECKING:
    from core.graph import ComputationalGraph

class Layer(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Compute the layer output."""
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @property
    def parameters(self) -> list[Tensor]:
        return []


    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

class Linear(Layer):
    def __init__(self, in_feature:int, out_feature:int, bias:bool=True):
        self.training: bool = True
        self.in_feature = in_feature
        self.out_feature = out_feature

        # Xavier initialization
        scale = np.sqrt(1.0 / in_feature)
        weights_data:np.ndarray = np.random.randn(in_feature, out_feature) * scale
        self.weights:Tensor = Tensor(weights_data, role=WEIGHTS_ROLE)

        if bias:
            self.bias = Tensor(np.zeros(out_feature), role=BIAS_ROLE)
        else:
            self.bias = None

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature}, out_feature={self.out_feature}, bias={self.bias})"

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Compute the layer output: y = xW + b"""
        output = x.matmul(self.weights)

        if self.bias is not None:
            output = output + self.bias

        return output

    @override
    def train(self) -> None:
        self.training = True
        self.weights.requires_grad=True
        if self.bias is not None:
            self.bias.requires_grad=True

    @override
    def eval(self) -> None:
        self.training = False
        self.weights.requires_grad=False
        if self.bias is not None:
            self.bias.requires_grad=False

    @property
    @override
    def parameters(self):
        """Return the list of trainable parameters in the layer"""
        params = [self.weights]

        if self.bias is not None:
            params.append(self.bias)

        return params

class Dropout(Layer):
    def __init__(self, p:float=0.5):
        self.p: float = p
        self.training: bool=True

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        # 1. Create Mask
        keep_prob = 1.0 - self.p
        mask = np.random.random(x.data.shape) < keep_prob

        # 2. Scale Factor (Inverted Dropout)
        scale = 1.0 / keep_prob

        # 3. Apply
        return x * Tensor(mask) * Tensor(scale)


    @override
    def train(self) -> None:
        self.training = True

    @override
    def eval(self) -> None:
        self.training = False

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.training: bool = True
        self.layers: list[Layer] = list(layers)
        self._graph: ComputationalGraph | None = None

    @override
    def __repr__(self) -> str:
        return f"Sequential=({[l for l in self.layers]})"

    @override
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    @override
    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params

    @override
    def train(self) -> None:
        self.training = True
        for layer in self.layers:
            layer.train()

    @override
    def eval(self) -> None:
        self.training = False
        for layer in self.layers:
            layer.eval()

    def build_graph(self, arch: bool = True, forward: bool = False,
                    backward: bool = False) -> None:
        """Build the graph of this model.

        Args:
            arch: include the network architecture (layer info).
            forward: include the forward computational graph (data flow).
            backward: include the backward computational graph (autograd ops).
        """
        # Imported lazily to avoid a circular import (core.graph imports layers).
        from core.graph import ComputationalGraph
        self._graph = ComputationalGraph(self)
        self._graph.build(arch=arch, forward=forward, backward=backward)

    def save_graph(self, path: str | Path, arch: bool = True, forward: bool = False,
                   backward: bool = False) -> None:
        """Render the graph to a .png image at ``path``."""
        self.build_graph(arch, forward, backward)
        assert self._graph is not None
        self._graph.render(path)

    def destroy_graph(self) -> None:
        self._graph = None
