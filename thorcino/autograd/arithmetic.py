"""Backward nodes for the tensor arithmetic ops.

Each `apply` returns an empty array in place of the gradient of an operand whose
`requires_grad` is False, and unbroadcasts the rest back to the operand's shape."""

import numpy as np
from typing import override
from thorcino.tensor import Tensor
from thorcino.utils import unbroadcast
from thorcino.autograd.base import Function

class IdentityBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors
        out = np.array([])

        if a.requires_grad:
            out = grad_output.data

        return Tensor(out),

class AddBackward(Function):
    @override
    def apply(self, grad_output: Tensor)-> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data, a.shape)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output.data, b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class SubBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data, a.shape)
        if b.requires_grad:
            grad_b = unbroadcast(-grad_output.data, b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class MulBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data * b.data, a.shape)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output.data * a.data, b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class DivBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        inv = (1/b.data)
        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data * inv, a.shape)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output.data * -a.data*(inv**2), b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class MatmulBackward(Function):
    """Gradients of `a @ b`, promoting a 1-D operand to a matrix so the batched
    `matmul` rules still apply."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        g = grad_output.data[..., None, :] if grad_output.data.ndim == 1 else grad_output.data
        # Aligns shapes by transposing the partner matrix
        if a.requires_grad:
            b2 = b.data[..., None] if b.data.ndim == 1 else np.swapaxes(b.data, -2, -1)
            g_a:np.ndarray = np.matmul(g, b2)
            grad_a = unbroadcast(g_a, a.shape)

        if b.requires_grad:
            a2 = a.data[..., None] if a.data.ndim == 1 else np.swapaxes(a.data, -2, -1)
            g_b:np.ndarray = np.matmul(a2, g)
            grad_b = unbroadcast(g_b, b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class SumBackward(Function):
    """Broadcasts the incoming gradient back over the axis that was summed away."""

    def __init__(self, x: Tensor, axis:int|None = -1, keepdims:bool = True):
        super().__init__(x)
        self.axis:int|None = axis
        self.keepdims:bool = keepdims

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors

        if not t.requires_grad:
            return Tensor(np.array([])),

        grad = grad_output.data
        if not self.keepdims and self.axis is not None:
            grad = np.expand_dims(grad, self.axis)

        return Tensor(np.broadcast_to(grad, t.shape)),

class ReshapeBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors
        out = np.array([])

        if a.requires_grad:
            out = np.reshape(grad_output.data, a.shape)

        return Tensor(out),

class TransposeBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        out = np.array([])

        if t.requires_grad:
            out = np.transpose(grad_output.data)

        return Tensor(out),

class StackBackward(Function):
    """Splits the incoming gradient into one chunk per stacked input, along the
    axis they were stacked on."""

    def __init__(self, *tensors: Tensor, axis: int = 0):
        super().__init__(*tensors)
        self.axis: int = axis

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        chunks = np.split(grad_output.data, len(self.saved_tensors), axis=self.axis)
        return tuple(Tensor(np.squeeze(chunk, axis=self.axis)) for chunk in chunks)


class NegBackward(Function):
    """Gradient of `-a` (and of `other - a`): the incoming gradient, negated."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors
        out = np.array([])

        if a.requires_grad:
            out = unbroadcast(-grad_output.data, a.shape)

        return Tensor(out),

class ScaleBackward(Function):
    """Gradient of `a * k` / `a / k` for a constant `k`: the incoming gradient, scaled."""

    def __init__(self, a: Tensor, scale):
        super().__init__(a)
        self.scale = scale

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors
        out = np.array([])

        if a.requires_grad:
            out = unbroadcast(grad_output.data * self.scale, a.shape)

        return Tensor(out),

class PowBackward(Function):
    """Gradient of `a ** k` for a constant exponent: `k * a**(k-1)`."""

    def __init__(self, a: Tensor, exponent: float):
        super().__init__(a)
        self.exponent: float = exponent

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors
        out = np.array([])

        if a.requires_grad:
            out = grad_output.data * self.exponent * (a.data ** (self.exponent - 1))

        return Tensor(out),

class MeanBackward(Function):
    """Spreads the incoming gradient evenly over the elements that were averaged."""

    def __init__(self, x: Tensor, axis:int|None = None, keepdims:bool = False):
        super().__init__(x)
        self.axis:int|None = axis
        self.keepdims:bool = keepdims

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors

        if not t.requires_grad:
            return Tensor(np.array([])),

        count = t.data.size if self.axis is None else t.data.shape[self.axis]
        grad = _restore_reduced_axis(grad_output.data, t.data.ndim, self.axis, self.keepdims)

        return Tensor(np.broadcast_to(grad, t.shape) / count),

class MaxBackward(Function):
    """Routes the incoming gradient to the extreme elements only, splitting it
    evenly when several elements tie for the extremum."""

    def __init__(self, x: Tensor, axis:int|None = None, keepdims:bool = False, largest: bool = True):
        super().__init__(x)
        self.axis:int|None = axis
        self.keepdims:bool = keepdims
        self.largest:bool = largest

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors

        if not t.requires_grad:
            return Tensor(np.array([])),

        reduce = np.max if self.largest else np.min
        extremum = reduce(t.data, axis=self.axis, keepdims=True)
        mask = (t.data == extremum).astype(t.data.dtype)
        ties = np.sum(mask, axis=self.axis, keepdims=True)

        grad = _restore_reduced_axis(grad_output.data, t.data.ndim, self.axis, self.keepdims)

        return Tensor(mask * grad / ties),

class IndexBackward(Function):
    """Scatters the incoming gradient back into the positions that were selected,
    accumulating where an index is selected more than once."""

    def __init__(self, x: Tensor, idx):
        super().__init__(x)
        self.idx = idx

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors

        if not a.requires_grad:
            return Tensor(np.array([])),

        out = np.zeros_like(a.data)
        np.add.at(out, self.idx, grad_output.data)

        return Tensor(out),

def _restore_reduced_axis(grad: np.ndarray, ndim: int, axis:int|None, keepdims: bool) -> np.ndarray:
    """Give `grad` back the axis a reduction removed, so it broadcasts against the operand."""
    if keepdims:
        return grad
    if axis is None:
        return np.reshape(grad, (1,) * ndim)
    return np.expand_dims(grad, axis)
