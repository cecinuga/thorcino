from typing import Any
import numpy as np

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