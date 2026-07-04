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
