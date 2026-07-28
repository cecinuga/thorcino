import numpy as np
from core.dataset.dataset import DataLoader, TensorDataset
from core.layers import Linear, Sequential
from core.losses import MSELoss
from core.optimizer import SGD
from core.tensor import Tensor
from core.training.schedulers import CosineSchedule
from core.training.trainer import Trainer


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

def create_trainer(in_feature: int, out_feature:int, min_lr: float, max_lr: float, epochs: int) -> Trainer:
    """Helper to build model, loss, optimizer, scheduler and finally return the trainer"""

    model = Sequential(
        Linear(in_feature, out_feature)
    )
    loss = MSELoss()
    optimizer = SGD(model.parameters, min_lr)
    scheduler = CosineSchedule(min_lr, min_lr, epochs)
    trainer = Trainer(model, loss, optimizer, scheduler)

    return trainer

def fit_model(trainer: Trainer, X_tr: DataLoader, X_te: DataLoader, epochs: int, eval_step: int) -> None:
    """Helper to train a model over a sample"""

    for i in range(epochs):
        _ = trainer.train_epoch(X_tr, 1)

        if i % eval_step == 0:
            _ = trainer.eval(X_te)

def fit_models(trainers: list[Trainer], dataset_tr: np.ndarray, dataset_te: np.ndarray, batch_size: int = 1) -> None:
    """Helper to train a set of models over the complete dataset"""

    for trainer, sample_tr, sample_te in zip(trainers, dataset_tr, dataset_te):        
        sample_tr, sample_te = preprocess_dataloader(sample_tr, sample_te, batch_size)
        fit_model(trainer, sample_tr, sample_te)