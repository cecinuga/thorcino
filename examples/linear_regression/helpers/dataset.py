from typing import Any
import numpy as np
from sklearn.datasets import make_regression

from core.dataset.dataset import DataLoader, TensorDataset
from core.tensor import Tensor

def preprocess_dataloader(sample_tr: np.ndarray, sample_te: np.ndarray, batch_size:int = 1) -> tuple[DataLoader, DataLoader]:
    """Helper to create a DataLoader starting to Numpy arrays"""
    sample_tr_x, sample_tr_y = sample_tr[:, :-1], sample_tr[:, -1].reshape(-1, 1)
    sample_te_x, sample_te_y = sample_te[:, :-1], sample_te[:, -1].reshape(-1, 1)

    data_tr = TensorDataset(Tensor(sample_tr_x), Tensor(sample_tr_y))
    data_te = TensorDataset(Tensor(sample_te_x), Tensor(sample_te_y))

    return DataLoader(data_tr, batch_size), DataLoader(data_te, batch_size)

def split_dataset(X:np.ndarray, split_ratio:float = 0.9, axis:int=0) -> tuple[np.ndarray, np.ndarray]:
    """Split in train and test dataset"""

    assert axis < len(X.shape)
    split_idx = int(X.shape[axis]*split_ratio)

    slicer0 = [slice(None)] * len(X.shape)
    slicer1 = [slice(None)] * len(X.shape)
    
    slicer0[axis] = slice(None, split_idx)
    slicer1[axis] = slice(split_idx, None)

    X0, X1 = X[tuple(slicer0)], X[tuple(slicer1)]

    return X0, X1

def generate_noise_dataset(f: Any, size: int, sample_size: int, noise: float, min: float, max: float, seed: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(min, max, sample_size)
    y = f(x)
    repeats = []

    for _ in range(size):
        y_noise = y + rng.standard_normal(x.shape[0]) * noise
        sample = np.stack([x, y_noise]).T
        repeats.append(sample)

    return np.stack(repeats)

def generate_dataset(f: Any, size: int, sample_size: int, noise_upto: float, min: float, max: float, seed: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(min, max, sample_size)
    y = f(x)
    y_noise = None

    datasets: list[np.ndarray] = []

    for noise in np.linspace(0, noise_upto, size):
        y_noise = y + rng.standard_normal(x.shape[0]) * noise
        sample = np.stack([x, y_noise]).T

        datasets.append(sample)

    return np.stack(datasets)

def make_splitted_regression(bias: float, noise: float, batch_size: int, seed: float = 777) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y, true_coefs = make_regression(n_samples=100, n_features=1, n_targets=1, bias=bias, noise=noise, coef=True, random_state=seed)

    true_coefs = np.array([true_coefs])

    X, y = X, y.reshape(-1, 1)

    X_tr, X_te = split_dataset(X, axis=0)
    y_tr, y_te = split_dataset(y, axis=0)

    return true_coefs, X_tr, X_te, y_tr, y_te