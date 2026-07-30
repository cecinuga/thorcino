from typing import Any
import numpy as np

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
    
    X0, X1 = X[:, :split_idx], X[:, split_idx:]

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