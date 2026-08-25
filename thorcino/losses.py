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
