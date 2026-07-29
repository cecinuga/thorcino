"""Plot data helper"""

from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def generate_axs(dataset_size: np.ndarray, axis:int = 0) -> tuple[Figure, np.ndarray]:
    row = int(np.floor(np.sqrt(dataset_size)))
    col = int(np.ceil(dataset_size/row))    
    fig, axs = plt.subplots(row, col, figsize=(16, 7))

    return fig, axs

def bar_dataset(X: np.ndarray, Y: np.ndarray, fig: Figure, axs: np.ndarray, X_color:str='blue', Y_color:str='red', width:float=0.35) -> None:
    axs = axs.flatten()
    labels = ['weight', 'bias']
    x = np.arange(len(labels))

    for i, ax in enumerate(axs):
        ax.bar(x - width/2, X[i].flatten(), width, color=X_color)
        ax.bar(x + width/2, Y.flatten(), width, color=Y_color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"Model {i+1}")

def scatter_dataset(X:np.ndarray, fig: Figure, axs: np.ndarray, color:str='blue', axis:int=1) -> None:
    assert axis < len(X.shape)
    
    axs = axs.flatten()
    X = X.swapaxes(axis, 1)

    for i, ax in enumerate(axs):
        ax.scatter(X[i, :, 0], X[i, :, 1], color=color)

def plot_dataset(X:np.ndarray, fig: Figure, axs: np.ndarray, color:str='red', axis:int=1) -> None:
    assert axis < len(X.shape)

    axs = axs.flatten()
    X = X.swapaxes(axis, 1)

    for i, ax in enumerate(axs):
        if (len(X.shape) == 3):
            ax.plot(X[i, :, 0], X[i, :, 1], color=color)
        if (len(X.shape) == 2):
            ax.plot(X[:, 0], X[:, 1], color=color)

     