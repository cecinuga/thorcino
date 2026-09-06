import numpy as np
from thorcino.tensor import Tensor
from thorcino.functions import softmax
from thorcino.autograd.base import Function
from typing import override

class MSELossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        predictions, targets = self.saved_tensors
        err = (predictions.data - targets.data)
        local_grad = (2*err)/predictions.data.size

        # Targets are data, never differentiated: an empty placeholder keeps the
        # returned arity equal to len(saved_tensors), which backward() enforces.
        return Tensor(grad_output.data * local_grad), Tensor(np.array([]))

class CrossEntropyLossBackward(Function):
    """Gradient w.r.t. raw logits; `targets` holds class indices, not one-hot rows."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        logits, targets = self.saved_tensors
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        # dL/dlogits = (softmax(logits) - onehot(targets)) / B
        probs = softmax(logits.data, dim=-1)
        probs[np.arange(batch_size), target_indices] -= 1.0
        local_grad: np.ndarray = probs / batch_size

        return Tensor(grad_output.data * local_grad), Tensor(np.array([]))

class BCELossBackward(Function):
    """Gradient w.r.t. probabilities (not logits), clipped as the forward pass is."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        predictions, targets = self.saved_tensors
        eps = 1e-7
        p = np.clip(predictions.data, eps, 1 - eps)
        y = targets.data
        n = predictions.data.size

        # dL/dpred = (p - y) / (p * (1 - p)) / N, with the same clip as the forward
        local_grad: np.ndarray = (p - y) / (p * (1 - p)) / n

        return Tensor(grad_output.data * local_grad), Tensor(np.array([]))
