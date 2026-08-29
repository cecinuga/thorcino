from __future__ import annotations
import numpy as np
from typing import override, TYPE_CHECKING

if TYPE_CHECKING:
    from thorcino.autograd import Function

class Tensor:
    """NumPy-backed array that records the operation that produced it for reverse-mode autodiff."""

    def __init__(self, data, requires_grad:bool=True, role:str|None=None):
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            # Build a batched tensor from a list of Tensors, e.g. Tensor([t1, t2, ...]).
            data = np.stack([t.data for t in data])
        self.data: np.ndarray = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.requires_grad:bool = requires_grad
        self.grad: np.ndarray|None = None
        self._grad_fn: Function|None = None
        self.role:str|None = role

    @override
    def __repr__(self) -> str:
        if self.role is not None:
            return f"Tensor=(role={self.role},shape={self.shape}, size={self.size}, dtype={self.dtype})"
        return f"Tensor=(shape={self.shape}, size={self.size}, dtype={self.dtype})"

    @property
    def dim(self) -> int:
        return len(self.data.shape)

    @override
    def __str__(self) -> str:
        return f"Tensor({self.data})"

    def numpy(self):
        return self.data

    def __getitem__(self, idx: int) -> np.ndarray:
        return Tensor(self.data[idx], requires_grad=self.requires_grad)

    def __len__(self) -> int:
        return len(self.data)

    def __neg__(self):
        return Tensor(-self.data)

    def __add__(self, other: Tensor | np.ndarray | float) -> Tensor:
        if isinstance(other, Tensor):
            from thorcino.autograd import AddBackward
            out = Tensor(self.data + other.data)
            out._grad_fn = AddBackward(self, other)
            return out
        return Tensor(self.data + other)

    def __radd__(self, other: np.ndarray | float) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Tensor | np.ndarray | float) -> Tensor:
        if isinstance(other, Tensor):
            from thorcino.autograd import SubBackward
            out = Tensor(self.data - other.data)
            out._grad_fn = SubBackward(self, other)
            return out
        return Tensor(self.data - other)

    def __rsub__(self, other: np.ndarray | float) -> Tensor:
        if isinstance(other, np.ndarray):
            return Tensor(other - self.data)
        return Tensor(-self.data + 1)

    def __rmul__(self, other: np.ndarray | float) -> Tensor:
        return self.__mul__(other)

    def __mul__(self, other: Tensor | np.ndarray | float) -> Tensor:
        if isinstance(other, Tensor):
            from thorcino.autograd import MulBackward
            out = Tensor(self.data * other.data)
            out._grad_fn = MulBackward(self, other)
            return out
        return Tensor(self.data * other)

    def __pow__(self, other: float) -> Tensor:
        return Tensor(self.data**other)

    def __matmul__(self, other: Tensor | np.ndarray) -> Tensor:
        return self.matmul(other)

    def __truediv__(self, other:Tensor | np.ndarray) -> Tensor:
        if isinstance(other, Tensor):
            from thorcino.autograd import DivBackward
            out = Tensor(self.data / other.data)
            out._grad_fn = DivBackward(self, other)
            return out
        return Tensor(self.data / other)

    def __gt__(self, other: Tensor | np.ndarray | float) -> Tensor:
        if isinstance(other, Tensor) or isinstance(other, np.ndarray) and not other.shape == self.shape:
            raise ValueError(f"cannot perform comparison, shape must be equal: {self.shape} != {other.shape}")

        if isinstance(other, Tensor):
            return Tensor(self.data > other.data)
        if isinstance(other, np.ndarray):
            return Tensor(self.data > other)
        return Tensor(self.data > other)
    
    def matmul(self, other: Tensor | np.ndarray) -> Tensor:
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"cannot perform matrix multiplication: {self.shape} @ {other.shape}\n"+
                    f"inner dimension must match: {self.shape[-1]} != {other.shape[-2]}"
                )

        out = Tensor(np.matmul(self.data, other.data))
        if isinstance(other, Tensor):
            from thorcino.autograd import MatmulBackward
            out._grad_fn = MatmulBackward(self, other)
            return out

        return out

    def reshape(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        if -1 in shape:
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"cannot reshape {self.shape} to {new_shape}\n"+
                f"[x] Element count mismatch: {self.size} elements vs {target_size} elements\n"+
                f"[x] Reshape preserves data, so total elements must stay the same\n"+
                f"[x] Use -1 to infer a dimension: reshape(-1, {new_shape[-1] if len(new_shape) > 0 else 1}) lets NumPy calculate"
            )
        from thorcino.autograd import ReshapeBackward
        reshaped_data = np.reshape(self.data, new_shape)
        out = Tensor(reshaped_data)
        out._grad_fn = ReshapeBackward(self)

        return out

    def transpose(self):
        from thorcino.autograd import TransposeBackward
        out = Tensor(np.transpose(self.data))
        out._grad_fn = TransposeBackward(self)
        return out

    def sum(self, axis:int|None=None, keepdims:bool = False) -> Tensor:
        from thorcino.autograd import SumBackward
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
        out._grad_fn = SumBackward(self, axis, keepdims)
        return out

    def mean(self, axis:int|None = None, keepdims:bool = False) -> Tensor:
       return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def max(self, axis:int|None = None, keepdims:bool = False) -> Tensor:
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims))

    def min(self, axis:int|None = None, keepdims:bool = False) -> Tensor:
        return Tensor(np.min(self.data, axis=axis, keepdims=keepdims))

    def backward(self, gradient:Tensor|None=None):
        """Compute gradients via backpropagation."""
        if not self.requires_grad:
            return

        if gradient is None:
            # Initialize gradient for scalar outputs
            if self.data.size == 1:
                gradient = Tensor(np.ones_like(self.data))
            else:
                raise ValueError("backward() requires gradient for non-scalar")

        sorted = list(reversed(self.__build_topo()))
        incoming: dict[int, Tensor] = {id(self): gradient.data}

        for node in sorted:    
            grad_in = incoming[id(node)]
            grads = node._grad_fn.apply(grad_in)

            for child, grad in zip(node._grad_fn.saved_tensors, grads):
                if id(child) in incoming:
                    incoming[id(child)] = incoming[id(child)] + grad.data
                else:
                    incoming[id(child)] = grad

                if child.grad is None:
                    child.grad = np.zeros_like(child.data)

                child.grad += grad.data


    def __build_topo(self, visited:set | None = None, topo: list | None = None) -> list[Tensor]: 
        """Reverse-postorder DFS: every node's consumers are garanteed to appear before it."""
        if visited is None:
            visited, topo = set(), []

        if id(self) not in visited:
            visited.add(id(self))

            if self._grad_fn is not None:
                for tensor in self._grad_fn.saved_tensors:
                    tensor.__build_topo(visited, topo)

                topo.append(self)

        return topo

    @staticmethod
    def stack(tensors: list["Tensor"], axis: int = 0) -> Tensor:
        """Stack a list of tensors along a new axis, keeping them wired into
        the autograd graph (unlike ``Tensor(np.stack(...))``, which builds a
        fresh leaf tensor with no ``_grad_fn``)."""
        out = Tensor(np.stack([t.data for t in tensors], axis=axis))
        if any(t.requires_grad for t in tensors):
            from thorcino.autograd import StackBackward
            out._grad_fn = StackBackward(*tensors, axis=axis)
        return out

    def zero_grad(self) -> None:
        """Reset gradients to None."""
        self.grad = None

    def destroy_graph(self) -> None:
        """Destroy the operations graph"""
        if self._grad_fn is not None:
            for t in self._grad_fn.saved_tensors:
                t.destroy_graph()
            self._grad_fn = None

