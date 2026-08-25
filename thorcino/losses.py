<<<<<<< HEAD:core/losses.py
import numpy as np
from core.tensor import Tensor

class MSELoss:
    def forward(self, prediction: Tensor, targets: Tensor) -> Tensor:
        # 1. Element-wise difference
        diff = prediction - targets

        # 2. Square the difference
        squared_diff = diff**2

        # 3. Mean reduction
        mse = np.mean(squared_diff)

        return Tensor(mse)

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    # 1. Find max for stability
    max_vals = np.max(x, axis=dim, keepdims=True)

    # 2. Subtract max (the shift)
    shifted = x.data - max_vals

    # 3. Compute log-sum-exp safely
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # 4. Result = input - max - log_sum_exp
    return Tensor(x.data - max_vals - log_sum_exp)
=======
from typing import override
from abc import ABC, abstractmethod

from thorcino.autograd import CrossEntropyLossBackward, MSELossBackward, BCELossBackward, Function
from thorcino.functions import mse, cross_entropy, binary_cross_entropy
from thorcino.tensor import Tensor

class Loss(ABC):
    grad_fn: type[Function] = Function

    @abstractmethod
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pass

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)


class MSELoss(Loss):
    grad_fn:type[Function] = MSELossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        out = Tensor(mse(predictions.data, targets.data))
        out._grad_fn = self.grad_fn(predictions, targets)
        return out

class CrossEntropyLoss(Loss):
    grad_fn:type[Function] = CrossEntropyLossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        out = Tensor(cross_entropy(predictions.data, targets.data))
        out._grad_fn = self.grad_fn(predictions, targets)

        return out


class BinaryCrossEntropyLoss(Loss):
    grad_fn:type[Function] = BCELossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        out = Tensor(binary_cross_entropy(predictions.data, targets.data))
        out._grad_fn = self.grad_fn(predictions, targets)

        return out
>>>>>>> 978a770889a515157595e8a1d564c844136399ab:thorcino/losses.py
