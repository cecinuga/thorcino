import random
from typing import Any
import numpy as np
from sklearn.datasets import make_regression, make_low_rank_matrix

from examples.recurrent.helpers import generate_invalid_seqs, generate_valid_seqs, parse_vect, tokenize
from thorcino.dataset.dataset import DataLoader, TensorDataset
from thorcino.tensor import Tensor

def preprocess_dataloader(sample_tr: np.ndarray, sample_te: np.ndarray, batch_size:int = 1, n_targets:int = 1) -> tuple[DataLoader, DataLoader]:
    """Helper to create a DataLoader starting to Numpy arrays"""
    sample_tr_x, sample_tr_y = sample_tr[:, :-n_targets], sample_tr[:, -n_targets:]
    sample_te_x, sample_te_y = sample_te[:, :-n_targets], sample_te[:, -n_targets:]

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

def get_dataset(row: int, col: int) -> tuple[np.array, np.array]:
    X_valid = np.array(tokenize(generate_valid_seqs(row, col)))
    X_invalid = np.array(tokenize(generate_invalid_seqs(row, col*2)))
    X = np.append(X_valid, X_invalid, axis=0)
    np.random.shuffle(X)

    Y = np.array([np.array([parse_vect(seq)], dtype=np.float32) for seq in X])

    return X, Y

def split_xy_dataset(X: np.ndarray, Y: np.ndarray, split_ratio: float) -> tuple[np.array, np.array, np.array, np.array]:
    assert X.shape[0] == Y.shape[0]

    train_len = int(split_ratio*X.shape[0])

    X_train, X_test = X[:train_len], X[train_len:]
    Y_train, Y_test = Y[:train_len], Y[train_len:]

    return X_train, Y_train, X_test, Y_test

def preprocess(X: np.ndarray, Y: np.ndarray, batch_size: int, split_ratio: float) -> tuple[DataLoader, DataLoader]:
    X_train, Y_train, X_test, Y_test = split_xy_dataset(X, Y, split_ratio)
    train_dataset, test_dataset = TensorDataset(Tensor(X_train), Tensor(Y_train)), TensorDataset(Tensor(X_test), Tensor(Y_test))
    train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size, True), DataLoader(test_dataset, batch_size, True)

    return train_dataloader, test_dataloader

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

def generate_illcond_regression_repeats(size: int, n_samples:int, n_feature:int, n_targets:int, effective_rank:int, tail_strength:float, noise:float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`size` independent ill-conditioned realizations, each drawn from its own random_state so both the design matrix and the underlying true coefficients differ per trial. Returns the stacked samples alongside the per-trial true coefficients used to generate them."""
    repeats = []
    coefs = []
    for i in range(size):
        X, y, coef = make_regression(
            n_samples=n_samples,
            n_features=n_feature,
            n_targets=n_targets,
            noise=noise,
            tail_strength=tail_strength,
            effective_rank=effective_rank,
            coef=True,
            random_state=seed+i
        )
        repeats.append(np.hstack([X, y.reshape(n_samples, n_targets)]))
        coefs.append(coef.reshape(n_feature, n_targets))

    return np.stack(repeats), np.stack(coefs)

def generate_regression_repeats(size: int, sample_size: int, n_features: int, n_targets: int, noise: float, bias: float, seed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`size` independent noisy realizations of the same sklearn make_regression signal: X and the true coefficients are fixed (generated once with noise=0), only the y-noise draw differs per repeat."""
    X, y_clean, coef = make_regression(
        n_samples=sample_size, 
        n_features=n_features, 
        n_targets=n_targets, 
        n_informative=n_features, 
        bias=bias, 
        noise=0.0, 
        coef=True, 
        random_state=seed
    )
    y_clean = y_clean.reshape(sample_size, n_targets)
    coef = coef.reshape(n_features, n_targets)

    rng = np.random.default_rng(seed)
    repeats = np.stack([
        np.hstack([X, y_clean + rng.standard_normal((sample_size, n_targets)) * noise])
        for _ in range(size)
    ])

    return X, y_clean, repeats, coef

def generate_low_rank_repeats(size: int, sample_size: int, n_features: int, effective_rank: int, tail_strength: float, seed: float) -> np.ndarray:
    """`size` independent low-rank matrices (columns hold both features and target(s)), each drawn from its own random_state so trials differ, but share the same shape/rank structure."""
    repeats = np.stack([
        make_low_rank_matrix(
            n_samples=sample_size,
            n_features=n_features,
            random_state=seed + i,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
        )
        for i in range(size)
    ])

    return repeats

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