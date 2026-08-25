import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # 1. Clip to prevent raw overflow
    z = np.clip(x, -500, 500)

    # 2. Stable computation mask
    result = np.zeros_like(z)

    # Positive input: standard formula
    pos_mask = z >= 0
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

    # Negative input: alternative formula
    neg_mask = z < 0
    exp_z = np.exp(z[neg_mask])
    result[neg_mask] = exp_z / (1.0 + exp_z)

    return result

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def gelu(x: np.ndarray) -> np.ndarray:
    # Sigmoid approximation of GELU, not the exact erf-based formula.
    return sigmoid(x * 1.702) * x

def softmax(x: np.ndarray, dim:int=-1) -> np.ndarray:
    # 1. Shift values so max is 0
    x_max = np.max(x, axis=dim, keepdims=True)
    x_shifted = x - x_max

    # 2. Compute safe exponentials
    exp_values = np.exp(x_shifted)

    # 3. Normalize
    exp_sum = np.sum(exp_values, axis=dim, keepdims=True)
    return exp_values/exp_sum

def log_softmax(x: np.ndarray, dim:int=-1) -> np.ndarray:
    # 1. Find max for stability
    max_vals = np.max(x, axis=dim, keepdims=True)

    # 2. Subtract max (the shift)
    shifted = x - max_vals

    # 3. Compute log-sum-exp safely
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    return x - max_vals - log_sum_exp

def squared_error(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    diff = predictions - targets
    squared_diff = diff**2

    return squared_diff

def mse(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return  np.array(np.mean(squared_error(predictions, targets)))

def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # 1. Apply stable log_softmax
    log_probs = log_softmax(logits, dim=-1)

    # 2. Select probability of the correct class
    batch_size = logits.shape[0]
    target_indices = targets.astype(int)

    # Advanced Indexing: "Lookup Table" style
    selected_log_probs = log_probs[np.arange(batch_size), target_indices]

    # 3. Negative Log Likelyhood
    result = -np.mean(selected_log_probs)
    return np.array(result)

def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # 1. Clip to prevent log(0) -> NaN
    eps = 1e-7
    clamped_preds = np.clip(predictions, eps, 1 - eps)

    # Compute BCE
    term_1 = targets * np.log(clamped_preds)
    term_2 = (1 - targets) * np.log(1 - clamped_preds)

    result = -np.mean(term_1 + term_2)
    return np.array(result)
