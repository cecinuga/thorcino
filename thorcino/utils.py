import numpy as np

def unbroadcast(x: np.ndarray, shape:tuple[int, ...]):
    """Sum `x` down to `shape`, undoing NumPy broadcasting so a gradient matches its operand's shape."""
    while x.ndim > len(shape):
        x = np.sum(x, axis=0)

    for i, s in enumerate(shape):
        if s == 1 and x.shape[i] != 1:
            x = x.sum(axis=i, keepdims=True)
    return x
