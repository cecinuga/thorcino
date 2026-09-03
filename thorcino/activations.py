"""Activation layers.

Each wires its backward node only while `training`, so an `eval()`-mode activation
returns a tensor detached from the autograd graph."""

from typing import override
from thorcino.layers import Layer
from thorcino.tensor import Tensor
from thorcino.functions import relu, sigmoid, tanh, gelu, softmax
from thorcino.autograd import IdentityBackward, ReLUBackward, SigmoidBackward, TanhBackward, GELUBackward, SoftmaxBackward

class Identity(Layer):
    def __init__(self):
        self.training = True

    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(x.data)
        if self.training:
            res._grad_fn = IdentityBackward(x)
        return res

    @override
    def train(self) -> None:
        self.training = True

    @override
    def eval(self) -> None:
        self.training = False

class ReLU(Layer):
    def __init__(self):
        self.training = True

    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(relu(x.data))
        if self.training:
            res._grad_fn = ReLUBackward(x)
        return res

    @override
    def train(self) -> None:
        self.training = True

    @override
    def eval(self) -> None:
        self.training = False

class Sigmoid(Layer):
    def __init__(self):
        self.training = True

    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(sigmoid(x.data))
        if self.training:
            res._grad_fn = SigmoidBackward(x)
        return res

    @override
    def train(self) -> None:
        self.training = True

    @override
    def eval(self) -> None:
        self.training = False

class Tanh(Layer):
    def __init__(self):
        self.training = True

    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(tanh(x.data))
        if self.training:
            res._grad_fn = TanhBackward(x)
        return res
    
    @override
    def train(self) -> None:
        self.training = True

    @override
    def eval(self) -> None:
        self.training = False

class GELU(Layer):
    """Sigmoid-approximated GELU."""

    def __init__(self):
        self.training = True

    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(gelu(x.data))
        if self.training:
            res._grad_fn = GELUBackward(x)
        return res
    
    @override
    def train(self) -> None:
        self.training = True

    @override
    def eval(self) -> None:
        self.training = False

class Softmax(Layer):
    def __init__(self):
        self.training = True

    @override
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        """Normalize along `dim` (last axis by default)."""
        res = Tensor(softmax(x.data, dim))
        if self.training:
            res._grad_fn = SoftmaxBackward(x, dim)
        return res

    @override
    def train(self) -> None:
        self.training = True

    @override
    def eval(self) -> None:
        self.training = False
