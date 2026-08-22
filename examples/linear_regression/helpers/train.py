import numpy as np
from thorcino.dataset.dataset import DataLoader
from thorcino.layers import Linear, Sequential
from thorcino.losses import MSELoss
from thorcino.optimizer import SGD
from thorcino.training.schedulers import CosineSchedule
from thorcino.training.trainer import Trainer
from examples.linear_regression.helpers.dataset import preprocess_dataloader


def create_trainer(in_feature: int, out_feature:int, min_lr: float, max_lr: float, epochs: int, weights_decay:float = 0) -> Trainer:
    """Helper to build model, loss, optimizer, scheduler and finally return the trainer"""

    model = Sequential(
        Linear(in_feature, out_feature)
    )
    loss = MSELoss()
    optimizer = SGD(model.parameters, max_lr, weights_decay)
    scheduler = CosineSchedule(max_lr, min_lr, epochs)
    trainer = Trainer(model, loss, optimizer, scheduler)

    return trainer

def get_model_param(model: Sequential) -> np.ndarray:
    """Return the model parameters"""

    param = model.parameters
    weights, bias = param
    bias = bias.reshape(-1, 1)

    param = np.array([weights, bias], dtype=object)
    return param

def get_models_params(models: list[Sequential]) -> np.ndarray:
    """Return the models parameters"""
    
    params = []
    for model in models:
        local_param = get_model_param(model)
        params.append(local_param)

    return np.array(params)

def fit_model(trainer: Trainer, X_tr: DataLoader, X_te: DataLoader, epochs: int, eval_step: int) -> None:
    """Helper to train a model over a sample"""

    for i in range(epochs):
        _ = trainer.train_epoch(X_tr, 1)

        if i % eval_step == 0:
            _ = trainer.eval(X_te)

def fit_models(trainers: list[Trainer], dataset_tr: np.ndarray, dataset_te: np.ndarray, epochs: int, eval_step: int, batch_size: int = 1) -> None:
    """Helper to train a set of models over the complete dataset"""

    for trainer, sample_tr, sample_te in zip(trainers, dataset_tr, dataset_te):        
        sample_tr, sample_te = preprocess_dataloader(sample_tr, sample_te, batch_size)
        fit_model(trainer, sample_tr, sample_te, epochs, eval_step)

