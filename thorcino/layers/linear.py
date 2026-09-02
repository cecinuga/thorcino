from typing import override
import numpy as np
from thorcino.consts import BIAS_ROLE, WEIGHTS_ROLE
from thorcino.layers.layer import Layer
from thorcino.tensor import Tensor


class Linear(Layer):
    def __init__(self, in_feature:int, out_feature:int, bias:bool=True):
        self.training: bool = True
        self.in_feature = in_feature
        self.out_feature = out_feature

        # Xavier initialization
        scale = np.sqrt(1.0 / in_feature)
        weights_data:np.ndarray = np.random.randn(in_feature, out_feature) * scale
        self.weights:Tensor = Tensor(weights_data, role=WEIGHTS_ROLE)

        if bias:
            self.bias = Tensor(np.zeros(out_feature), role=BIAS_ROLE)
        else:
            self.bias = None

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature}, out_feature={self.out_feature}, bias={self.bias})"

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Compute the layer output: `y = xW + b`"""
        output = x @ self.weights

        if self.bias is not None:
            output = output + self.bias

        return output

    @override
    def train(self) -> None:
        self.training = True
        self.weights.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    @override
    def eval(self) -> None:
        self.training = False
        self.weights.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    @property
    @override
    def parameters(self):
        """Return the list of trainable parameters in the layer"""
        params = [self.weights]

        if self.bias is not None:
            params.append(self.bias)

        return params

    @property
    @override
    def state(self) -> dict:
        data = { 'W': self.weights, }
        if self.bias is not None:
            data['b'] = self.bias

        return data
