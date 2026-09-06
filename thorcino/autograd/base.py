from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thorcino.tensor import Tensor

from thorcino.grad_mode import no_grad, is_grad_enabled

__all__ = ["Function", "enable_autograd", "no_grad", "is_grad_enabled"]

def enable_autograd(quiet:bool=False) -> None:
    """No-op kept for API compatibility: autograd is on unless a `no_grad` block
    is active - see `thorcino.grad_mode.no_grad`."""
    pass

class Function:
    """Backward node of the autograd graph: holds one operation's operands and
    turns an incoming gradient into their gradients."""

    def __init__(self, *tensors: Tensor):
        # Operands needed to compute the gradients in apply().
        self.saved_tensors:tuple[Tensor, ...] = tensors
        # Each operand's own backward node, turning these into a traversable graph.
        self.next_functions:list[Function|None] = [t._grad_fn for t in tensors]

    def apply(self, grad_output:Tensor)-> tuple[Tensor, ...]:
        """Return one gradient per entry of `saved_tensors`, in the same order."""
        raise NotImplementedError()
