from typing import override
from abc import ABC, abstractmethod

from thorcino.autograd import CrossEntropyLossBackward, MSELossBackward, BCELossBackward, Function
from thorcino.functions import mse, cross_entropy, binary_cross_entropy
from thorcino.tensor import Tensor

class Loss(ABC):
    """Base loss: subclasses compute a scalar and attach their `grad_fn` to it,
    which is what makes `backward()` reachable from the loss."""

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
    """Softmax cross-entropy over raw logits; `targets` are class indices, not one-hot."""

    grad_fn:type[Function] = CrossEntropyLossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        out = Tensor(cross_entropy(predictions.data, targets.data))
        out._grad_fn = self.grad_fn(predictions, targets)

        return out


class BinaryCrossEntropyLoss(Loss):
    """BCE over probabilities in [0, 1], not logits."""

    grad_fn:type[Function] = BCELossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        out = Tensor(binary_cross_entropy(predictions.data, targets.data))
        out._grad_fn = self.grad_fn(predictions, targets)

        return out
