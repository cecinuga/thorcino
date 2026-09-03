"""Minimal NumPy-backed autograd engine and neural-network toolkit."""

from thorcino.tensor import Tensor
from thorcino.autograd import Function
from thorcino.layers import Layer, Linear, Dropout, RNN, Sequential
from thorcino.layers.lstm import LSTM
from thorcino.activations import Identity, ReLU, Sigmoid, Tanh, GELU, Softmax
from thorcino.losses import Loss, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from thorcino.optimizer import Optimizer, SGD, SGD_DL2, SGDM, Adam, AdamW
from thorcino.training import Trainer, clip_grad_norm, Schedule, CosineSchedule
from thorcino.dataset import TensorDataset, ImageDataset, DataLoader, RandomHorizontalFlip, RandomCrop, Compose
from thorcino.dataset.dataset import Dataset

__all__ = [
    # tensor + autograd
    "Tensor",
    "Function",
    # layers
    "Layer",
    "Linear",
    "Dropout",
    "RNN",
    "LSTM",
    "Sequential",
    # activations
    "Identity",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "Softmax",
    # losses
    "Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "BinaryCrossEntropyLoss",
    # optimizers
    "Optimizer",
    "SGD",
    "SGD_DL2",
    "SGDM",
    "Adam",
    "AdamW",
    # training
    "Trainer",
    "clip_grad_norm",
    "Schedule",
    "CosineSchedule",
    # data loading
    "Dataset",
    "TensorDataset",
    "ImageDataset",
    "DataLoader",
    "RandomHorizontalFlip",
    "RandomCrop",
    "Compose",
]
