import pickle
import numpy as np
from pathlib import Path

from thorcino.losses import Loss
from thorcino.layers import Layer
from thorcino.tensor import Tensor
from thorcino.training.schedulers import Schedule
from thorcino.dataset import DataLoader
from thorcino.optimizer import Optimizer

def clip_grad_norm(parameters: list[Tensor], max_norm: float = 1.0) -> float:
    """Scale all parameter gradients in-place so their combined L2 norm doesn't exceed max_norm; returns the pre-clip norm."""
    # 1. Compute global norm across all parameters
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            # Access raw data to avoid graph overhead
            grad_data = param.grad
            total_norm += np.sum(grad_data ** 2)
    total_norm = np.sqrt(total_norm)

    # 2. Scale uniformly if norm exceeds threshold
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                param.grad *= clip_coef

    return float(total_norm)

class Trainer:
    def __init__(self,
        model: Layer,
        loss_fn: Loss,
        optimizer: Optimizer,
        scheduler: Schedule|None = None,
        grad_clip_norm: float|None = None
    ):
        self.model: Layer = model
        self.loss_fn: Loss = loss_fn
        self.optimizer: Optimizer = optimizer
        self.scheduler: Schedule|None = scheduler
        self.grad_clip_norm: float|None = grad_clip_norm

        # State tracking
        self.step: int = 0
        self.epoch: int = 0
        self.training: bool = True
        self.history:dict[str, list[float]] = {'train_loss': [], 'eval_loss': [], 'lr': []}

    def _accumulate(self, total_loss: float, accumulated_loss: float, num_batches: int):
        if self.grad_clip_norm is not None:
            _ = clip_grad_norm(self.model.parameters, self.grad_clip_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        total_loss += accumulated_loss
        num_batches += 1

        return total_loss, accumulated_loss, num_batches

    def train_epoch(self, dataloader: DataLoader, accumulation_steps:int = 1) -> float:
        """Run one epoch, updating parameters every `accumulation_steps` batches; returns the average per-batch loss."""
        self.model.train()

        total_loss: float = 0
        num_batches:float = 0
        accumulated_loss: float = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 1. Forward pass
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets)

            # 2. Scale loss for accumulation
            # Dividing by N so the sum of N gradients equals to mean
            scaled_loss = loss.data / accumulation_steps
            accumulated_loss += float(scaled_loss)

            # 3. Backward pass (accumulates into .grad).
            # Scale the seed gradient here, not `loss` itself: Tensor.__truediv__
            # by a plain scalar returns a fresh leaf with no _grad_fn, which would
            # sever the graph and make backward() a no-op.
            loss.backward(Tensor(np.array(1.0 / accumulation_steps, dtype=np.float32)))

            # Only update every 'accumulation_steps'
            if (batch_idx + 1) % accumulation_steps == 0:
                total_loss, _, num_batches = self._accumulate(total_loss, accumulated_loss, num_batches)
                accumulated_loss = 0
                self.step += 1

        if accumulated_loss > 0:
            total_loss, _, num_batches = self._accumulate(total_loss, accumulated_loss, num_batches)

        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)

        if self.scheduler is not None:
            self.optimizer.lr = self.scheduler.get_lr(self.epoch)
            self.history['lr'].append(self.optimizer.lr)

        self.epoch += 1
        return avg_loss

    def eval(self, dataloader: DataLoader) -> tuple[float, float]:
        """Evaluate without updating parameters; returns (avg_loss, accuracy). Accuracy is 0.0 for non-classification (1-D) outputs."""
        self.model.eval()
        self.training = False

        total_loss: float = 0
        correct: int = 0
        total: int = 0

        for inputs, targets in dataloader:
            # Forward pass only
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets)

            total_loss += float(loss.data)

            # Calculate accuracy (for classification)
            if len(preds.shape) > 1: # Multi class
                predictions = np.argmax(preds.data, axis=1)
                if len(targets.shape) == 1: # Integer targets
                    correct += np.sum(predictions == targets.data)
                else:
                    correct += np.sum(predictions == np.argmax(targets.data, axis=1))
                total += len(predictions)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct / total if total > 0 else 0

        self.model.train()
        self.training = True
        self.history['eval_loss'].append(avg_loss)
        return avg_loss, accuracy

    def _get_model_state(self) -> list[Tensor]:
        return self.model.parameters

    def _get_optimizer_state(self):
        return self.optimizer.get_state()

    def _get_scheduler_state(self):
        if self.scheduler is not None:
            return self.scheduler.get_state()
        return None

    @property
    def train_loss(self):
        return self.history['train_loss']

    @property
    def eval_loss(self):
        return self.history['eval_loss']

    def save(self, path: Path|str) -> None:
        checkpoint = {
            'epoch':            self.epoch,
            'step':             self.step,
            'history':          self.history,
            'training_mode':    self.training,
            'model_state':      self._get_model_state(),
            'optimizer_state':  self._get_optimizer_state(),
            'scheduler_state':  self._get_scheduler_state(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load(self, path: Path|str) -> None:
        """Restore epoch/step/history/training-mode from a checkpoint; model, optimizer and scheduler state are not restored."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        self.training = checkpoint['training_mode']

        """
        # Restore states (simplified for educational purposes)
        if 'model_state' in checkpoint:
            self._set_model_state(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self._set_optimizer_state(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self._set_scheduler_state(checkpoint['scheduler_state'])
        """
