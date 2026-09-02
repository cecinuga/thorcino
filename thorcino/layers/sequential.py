from abc import ABC, abstractmethod
from pathlib import Path
from typing import override, TYPE_CHECKING
import numpy as np
from thorcino.consts import BIAS_ROLE, WEIGHTS_ROLE
from thorcino.layers.layer import Layer
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
    
    @property
    @override
    def state(self) -> dict:
        state = {}
        for layer in self.layers:
            state[layer.__repr__()] = layer.state

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

    def __build_graph(self, shape: tuple[int, ...], arch: bool = True, forward: bool = False,
                    backward: bool = False) -> None:
        """Build the graph of this model.

        Args:
            shape: real input shape (including the batch dimension) to run
                the model with, e.g. ``(batch, in_feature)`` for a plain
                feed-forward model or ``(batch, seq_len, in_feature)`` for a
                recurrent one. The graph is built from an actual forward/
                backward pass with this shape, so every node carries the
                real shape it would have in production.
            arch: include the network architecture (layer info).
            forward: include the forward computational graph (data flow).
            backward: include the backward computational graph (autograd ops).
        """
        # Imported lazily to avoid a circular import (core.graph imports layers).
        from thorcino.graph import ComputationalGraph
        self._graph = ComputationalGraph(self)
        self._graph.build(shape, arch=arch, forward=forward, backward=backward)

    def __save_graph(self, path: str | Path, shape: tuple[int, ...], arch: bool = True,
                   forward: bool = False, backward: bool = False) -> None:
        """Render the graph to a .png image at ``path``.

        Args:
            path: output .png path.
            shape: real input shape (including the batch dimension), see
                :meth:`build_graph`.
            arch: include the network architecture (layer info).
            forward: include the forward computational graph (data flow).
            backward: include the backward computational graph (autograd ops).
        """
        self.__build_graph(shape, arch, forward, backward)
        assert self._graph is not None
        self._graph.render(path)


    def save_arch(self, path: str | Path, shape: tuple[int, ...]) -> None:
        self.__save_graph(path, shape)

    def save_forward(self, path: str | Path, shape: tuple[int, ...]) -> None:
        self.__save_graph(path, shape, arch=False, forward=True)

    def save_backward(self, path: str | Path, shape: tuple[int, ...]) -> None:
        self.__save_graph(path, shape, arch=False, backward=True)

    def destroy_graph(self) -> None:
        self._graph = None
