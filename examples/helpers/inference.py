import numpy as np
from thorcino.layers.sequential import Sequential
from thorcino.tensor import Tensor


def infer(model: Sequential, X: Tensor) -> np.ndarray:
    model.eval()
    preds = model(X)
    model.train()

    return preds.data