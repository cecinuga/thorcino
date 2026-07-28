import numpy as np
from core.tensor import Tensor
from core.functions import softmax
from core.autograd.base import Function
from typing import override

class MSELossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        predictions, targets = self.saved_tensors
        err = (predictions.data - targets.data)
        local_grad = (2*err)/predictions.data.size

        return Tensor(grad_output.data * local_grad),

class CrossEntropyLossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        logits, targets = self.saved_tensors
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        # dL/dlogits = (softmax(logits) - onehot(targets)) / B
        probs = softmax(logits.data, dim=-1)
        probs[np.arange(batch_size), target_indices] -= 1.0
        local_grad: np.ndarray = probs / batch_size

        return Tensor(grad_output.data * local_grad),

class BCELossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        predictions, targets = self.saved_tensors
        eps = 1e-7
        p = np.clip(predictions.data, eps, 1 - eps)
        y = targets.data
        n = predictions.data.size

        # dL/dpred = (p - y) / (p * (1 - p)) / N, with the same clip as the forward
        local_grad: np.ndarray = (p - y) / (p * (1 - p)) / n

        return Tensor(grad_output.data * local_grad),
