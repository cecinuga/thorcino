from typing import override
from thorcino.layers import Layer
from thorcino.tensor import Tensor
from thorcino.functions import relu, sigmoid, tanh, gelu, softmax
from thorcino.autograd import ReLUBackward, SigmoidBackward, TanhBackward, GELUBackward, SoftmaxBackward

class ReLU(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(relu(x.data))
        if res.requires_grad:
            res._grad_fn = ReLUBackward(x)
        return res

class Sigmoid(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(sigmoid(x.data))
        if res.requires_grad:
            res._grad_fn = SigmoidBackward(x)
        return Tensor(res)

class Tanh(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(tanh(x.data))
        if res.requires_grad:
            res._grad_fn = TanhBackward(x)
        return res

class GELU(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(gelu(x.data))
        if res.requires_grad:
            res._grad_fn = GELUBackward(x)
        return res

class Softmax(Layer):
    @override
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        res = Tensor(softmax(x.data, dim))
        if res.requires_grad:
            res._grad_fn = SoftmaxBackward(x, dim)
        return res
