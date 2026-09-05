from __future__ import annotations
import numpy as np
from typing import override, TYPE_CHECKING

from thorcino.grad_mode import is_grad_enabled

if TYPE_CHECKING:
    from thorcino.autograd import Function

class Tensor:
    """NumPy-backed array that records the operation that produced it for reverse-mode autodiff."""

    def __init__(self, data, requires_grad:bool=True, role:str|None=None):
        """Wrap `data` as a float32 array; `role` tags the tensor for graph rendering."""
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            # Build a batched tensor from a list of Tensors, e.g. Tensor([t1, t2, ...]).
            data = np.stack([t.data for t in data])
        self.data: np.ndarray = np.array(data, dtype=np.float32)
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
    
    @property
    def shape(self): 
        return self.data.shape
    
    @property
    def size(self): 
        return self.data.size
    
    @property
    def dtype(self): 
        return self.data.dtype

    def __getitem__(self, idx) -> Tensor:
        """Index/slice the tensor, routing the gradient back to the selected positions."""
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)
        if is_grad_enabled():
            from thorcino.autograd import IndexBackward
            out._grad_fn = IndexBackward(self, idx)
        return out

    def __len__(self) -> int:
        return len(self.data)

    def __neg__(self) -> Tensor:
        out = Tensor(-self.data)
        if is_grad_enabled():
            from thorcino.autograd import NegBackward
            out._grad_fn = NegBackward(self)
        return out

    def __add__(self, other: Tensor | np.ndarray | float) -> Tensor:
        if isinstance(other, Tensor):
            from thorcino.autograd import AddBackward
            out = Tensor(self.data + other.data)
            if is_grad_enabled():
                out._grad_fn = AddBackward(self, other)
            return out
        # Adding a constant shifts the value but not the derivative, so the
        # incoming gradient passes straight through to `self`.
        out = Tensor(self.data + other)
        if is_grad_enabled():
            from thorcino.autograd import IdentityBackward
            out._grad_fn = IdentityBackward(self)
        return out

    def __radd__(self, other: np.ndarray | float) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Tensor | np.ndarray | float) -> Tensor:
        if isinstance(other, Tensor):
            from thorcino.autograd import SubBackward
            out = Tensor(self.data - other.data)
            if is_grad_enabled():
                out._grad_fn = SubBackward(self, other)
            return out
        out = Tensor(self.data - other)
        if is_grad_enabled():
            from thorcino.autograd import IdentityBackward
            out._grad_fn = IdentityBackward(self)
        return out

    def __rsub__(self, other: np.ndarray | float) -> Tensor:
        """`other - self`, so the local derivative w.r.t. `self` is -1."""
        out = Tensor(np.asarray(other, dtype=np.float32) - self.data)
        if is_grad_enabled():
            from thorcino.autograd import NegBackward
            out._grad_fn = NegBackward(self)
        return out

    def __rmul__(self, other: np.ndarray | float) -> Tensor:
        return self.__mul__(other)

    def __mul__(self, other: Tensor | np.ndarray | float) -> Tensor:
        if isinstance(other, Tensor):
            from thorcino.autograd import MulBackward
            out = Tensor(self.data * other.data)
            if is_grad_enabled():
                out._grad_fn = MulBackward(self, other)
            return out
        out = Tensor(self.data * other)
        if is_grad_enabled():
            from thorcino.autograd import ScaleBackward
            out._grad_fn = ScaleBackward(self, other)
        return out

    def __pow__(self, other: float) -> Tensor:
        out = Tensor(self.data**other)
        if is_grad_enabled():
            from thorcino.autograd import PowBackward
            out._grad_fn = PowBackward(self, other)
        return out

    def __matmul__(self, other: Tensor | np.ndarray) -> Tensor:
        return self.matmul(other)

    def __truediv__(self, other:Tensor | np.ndarray) -> Tensor:
        if isinstance(other, Tensor):
            from thorcino.autograd import DivBackward
            out = Tensor(self.data / other.data)
            if is_grad_enabled():
                out._grad_fn = DivBackward(self, other)
            return out
        out = Tensor(self.data / other)
        if is_grad_enabled():
            from thorcino.autograd import ScaleBackward
            out._grad_fn = ScaleBackward(self, 1.0 / np.asarray(other, dtype=np.float32))
        return out

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
            if is_grad_enabled():
                out._grad_fn = MatmulBackward(self, other)
            return out

        return out

    def reshape(self, *shape) -> Tensor:
        """Reshape to `shape` (a varargs or a single tuple); one `-1` entry is inferred."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        if -1 in new_shape:
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
        reshaped_data = np.reshape(self.data, new_shape)
        out = Tensor(reshaped_data)
        if is_grad_enabled():
            from thorcino.autograd import ReshapeBackward
            out._grad_fn = ReshapeBackward(self)

        return out

    def transpose(self):
        """Reverse every axis, as `np.transpose` does - not just the last two."""
        out = Tensor(np.transpose(self.data))
        if is_grad_enabled():
            from thorcino.autograd import TransposeBackward
            out._grad_fn = TransposeBackward(self)
        return out

    def sum(self, axis:int|None=None, keepdims:bool = False) -> Tensor:
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
        if is_grad_enabled():
            from thorcino.autograd import SumBackward
            out._grad_fn = SumBackward(self, axis, keepdims)
        return out

    def mean(self, axis:int|None = None, keepdims:bool = False) -> Tensor:
        """Mean over `axis`; the gradient is spread evenly over the averaged elements."""
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))
        if is_grad_enabled():
            from thorcino.autograd import MeanBackward
            out._grad_fn = MeanBackward(self, axis, keepdims)
        return out

    def max(self, axis:int|None = None, keepdims:bool = False) -> Tensor:
        """Maximum over `axis`; the gradient is routed to the maximal elements
        (split evenly between them on a tie)."""
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims))
        if is_grad_enabled():
            from thorcino.autograd import MaxBackward
            out._grad_fn = MaxBackward(self, axis, keepdims, largest=True)
        return out

    def min(self, axis:int|None = None, keepdims:bool = False) -> Tensor:
        """Minimum over `axis`; the gradient is routed to the minimal elements
        (split evenly between them on a tie)."""
        out = Tensor(np.min(self.data, axis=axis, keepdims=keepdims))
        if is_grad_enabled():
            from thorcino.autograd import MaxBackward
            out._grad_fn = MaxBackward(self, axis, keepdims, largest=False)
        return out

    def backward(self, gradient:Tensor|None=None):
        """Accumulate gradients into `.grad` across the graph; `gradient` is required
        unless this tensor is a scalar. Accumulates, so zero the grads between steps."""
        if not self.requires_grad:
            return

        if gradient is None:
            # Initialize gradient for scalar outputs
            if self.data.size == 1:
                gradient = Tensor(np.ones_like(self.data))
            else:
                raise ValueError("backward() requires gradient for non-scalar")

        sorted = list(reversed(self.__build_topo()))
        # Partial gradients are kept as raw arrays: wrapping them in Tensors and
        # adding those would build a second graph while unwinding the first.
        incoming: dict[int, np.ndarray] = {id(self): gradient.data}

        for node in sorted:
            grad_in = incoming.get(id(node))
            if grad_in is None:
                # Every consumer of this node was excluded from the graph, so
                # nothing flows through it.
                continue

            saved = node._grad_fn.saved_tensors
            grads = node._grad_fn.apply(Tensor(grad_in))
            if len(grads) != len(saved):
                raise RuntimeError(
                    f"{type(node._grad_fn).__name__}.apply() returned {len(grads)} gradients "
                    f"for {len(saved)} saved tensors; it must return exactly one per operand"
                )

            for child, grad in zip(saved, grads):
                # Operands outside the graph get an empty placeholder gradient,
                # which must not be accumulated into their (differently shaped) .grad.
                if not child.requires_grad or grad.data.size == 0:
                    continue

                previous = incoming.get(id(child))
                incoming[id(child)] = grad.data if previous is None else previous + grad.data

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
        """Stack tensors along a new axis, keeping them wired into the autograd graph
        (unlike ``Tensor(np.stack(...))``, which yields a leaf with no ``_grad_fn``)."""
        out = Tensor(np.stack([t.data for t in tensors], axis=axis))
        if is_grad_enabled() and any(t.requires_grad for t in tensors):
            from thorcino.autograd import StackBackward
            out._grad_fn = StackBackward(*tensors, axis=axis)
        return out

    def zero_grad(self) -> None:
        """Reset gradients to None."""
        self.grad = None

    def destroy_graph(self) -> None:
        """Drop the whole upstream graph, freeing the tensors it kept alive."""
        if self._grad_fn is not None:
            for t in self._grad_fn.saved_tensors:
                t.destroy_graph()
            self._grad_fn = None
