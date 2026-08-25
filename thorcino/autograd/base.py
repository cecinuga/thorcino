from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thorcino.tensor import Tensor

def enable_autograd(quiet:bool=False) -> None:
    pass

class Function:
    def __init__(self, *tensors: Tensor):
        # Operands needed to compute the gradients in apply().
        self.saved_tensors:tuple[Tensor, ...] = tensors
        # Each operand's own backward node, turning these into a traversable graph.
        self.next_functions:list[Function|None] = [t._grad_fn for t in tensors]

    def apply(self, grad_output:Tensor)-> tuple[Tensor, ...]:
        """Compute gradients for the inputs."""
        raise NotImplementedError()
