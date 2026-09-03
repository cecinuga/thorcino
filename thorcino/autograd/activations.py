import numpy as np
from thorcino.functions import sigmoid, tanh, softmax
from thorcino.tensor import Tensor
from typing import override
from thorcino.autograd.base import Function

class ReLUBackward(Function):
    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        local_grad = (t.data > 0)

        return Tensor(grad_output.data * local_grad),

class SigmoidBackward(Function):
    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        s = sigmoid(t.data)
        local_grad = s*(1-s)

        return Tensor(grad_output.data * local_grad),

class TanhBackward(Function):
    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        local_grad = (1-tanh(t.data)**2)

        return Tensor(grad_output.data * local_grad),

class GELUBackward(Function):
    """Derivative of the sigmoid GELU approximation, matching `functions.gelu`."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        s = sigmoid(t.data * 1.702)
        local_grad = s * (1 + 1.702 * t.data * (1-s))

        return Tensor(grad_output.data * local_grad),

class SoftmaxBackward(Function):
    """Softmax Jacobian-vector product: `s * (g - sum(g * s))` along `dim`."""

    def __init__(self, x: Tensor, dim:int):
        super().__init__(x)
        self.dim:int = dim

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        s = softmax(t.data, self.dim)
        dot = np.sum(grad_output.data*s, axis=self.dim, keepdims=True)
        return Tensor(s * (grad_output.data - dot)),
