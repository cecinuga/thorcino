import numpy as np
from examples.helpers.inference import infer
from thorcino.layers.sequential import Sequential
from thorcino.tensor import Tensor


def diff_percentage(min: float, max: float) -> float:
    assert max >= min
    diff = max - min
    return diff / max * 100

def percentage(min: float, max: float) -> float:
    assert max >= min
    return min / max * 100

def accuracy(model: Sequential, X: Tensor, Y: Tensor) -> float:
    corrects = 0
    for seq, is_valid in zip(X, Y):
        seq = Tensor(seq[np.newaxis, :])
        pred = infer(model, seq)
        pred, is_valid = np.round(pred)[0, 0], is_valid[0]

        corrects += (is_valid == pred)

    return percentage(corrects, Y.shape[0])