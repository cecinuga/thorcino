from abc import ABC, abstractmethod
from pathlib import Path
from typing import override, TYPE_CHECKING
import numpy as np
from thorcino.consts import BIAS_ROLE, WEIGHTS_ROLE
from thorcino.layers import Layer
from thorcino.tensor import Tensor

if TYPE_CHECKING:
    from thorcino.graph import ComputationalGraph


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
        from thorcino.graph import ComputationalGraph
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
