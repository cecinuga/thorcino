import numpy as np
from typing import override
from thorcino.layers.sequential import Layer
from thorcino.tensor import Tensor

class Dropout(Layer):
    def __init__(self, p:float=0.5):
        self.p: float = p
        self.training: bool=True

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        # 1. Create Mask
        keep_prob = 1.0 - self.p
        mask = np.random.random(x.data.shape) < keep_prob

        # 2. Scale Factor (Inverted Dropout)
        scale = 1.0 / keep_prob

        # 3. Apply
        return x * Tensor(mask) * Tensor(scale)


    @override
    def train(self) -> None:
        self.training = True

    @override
    def eval(self) -> None:
        self.training = False

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"