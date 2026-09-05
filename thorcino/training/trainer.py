import pickle
import numpy as np
from pathlib import Path

from thorcino.losses import Loss
from thorcino.layers import Layer
from thorcino.tensor import Tensor
from thorcino.training.schedulers import Schedule
from thorcino.dataset import DataLoader
from thorcino.grad_mode import no_grad
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
    """Drives the train/eval loop over a model, recording per-epoch metrics in `history`
    and exposing pickle checkpoints through `save`/`load`."""

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
        self.history:dict[str, list[float]] = {'train_loss': [], 'eval_loss': [], 'accuracy': [], 'lr': []}

    def _accumulate(self, total_loss: float, accumulated_loss: float, num_batches: int):
        """Clip, apply one optimizer step, clear the grads, and fold `accumulated_loss`
        into the running totals."""
        if self.grad_clip_norm is not None:
            _ = clip_grad_norm(self.model.parameters, self.grad_clip_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        total_loss += accumulated_loss
        num_batches += 1

        return total_loss, accumulated_loss, num_batches

    def train_epoch(self, dataloader:DataLoader, accumulation_steps:int = 1) -> float:
        """Run one epoch, updating parameters every `accumulation_steps` batches; returns the average per-batch loss."""
        self.model.train()
        self.training = True

        # The schedule is indexed by the epoch it applies to, so it has to be read
        # before that epoch's updates run. Applying it afterwards would train epoch 0
        # at the optimizer's constructor rate and leave the final epoch one step
        # short of the floor the schedule was built to reach.
        if self.scheduler is not None:
            self.optimizer.lr = self.scheduler.get_lr(self.epoch)
            self.history['lr'].append(self.optimizer.lr)

        total_loss: float = 0
        num_batches:int = 0
        accumulated_loss: float = 0
        pending_batches: int = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 1. Forward pass
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets)

            # 2. Scale loss for accumulation
            # Dividing by N so the sum of N gradients equals to mean
            scaled_loss = loss.data / accumulation_steps
            accumulated_loss += float(scaled_loss)
            pending_batches += 1

            # 3. Backward pass (accumulates into .grad).
            # The seed gradient carries the accumulation scale so `loss` itself is
            # left untouched and the graph stays intact.
            loss.backward(Tensor(np.array(1.0 / accumulation_steps, dtype=np.float32)))

            # Only update every 'accumulation_steps'
            if (batch_idx + 1) % accumulation_steps == 0:
                total_loss, _, num_batches = self._accumulate(total_loss, accumulated_loss, num_batches)
                accumulated_loss = 0
                pending_batches = 0
                self.step += 1

        # Flush a short trailing group. Testing the loss value here instead of the
        # batch count would skip the flush whenever the leftover loss happened to
        # be zero, stranding those gradients in .grad for the next epoch.
        if pending_batches > 0:
            total_loss, _, num_batches = self._accumulate(total_loss, accumulated_loss, num_batches)
            self.step += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)

        self.epoch += 1
        return avg_loss

    def eval(self, dataloader: DataLoader, record: bool = True) -> tuple[float, float]:
        """Evaluate without updating parameters; returns (avg_loss, accuracy). Accuracy is 0.0 for non-classification (1-D) outputs.

        `record` appends the result to `history`. Pass `record=False` for evaluations
        that are not part of the training curve - a sweep over several validation
        sets, say - otherwise their entries interleave with the periodic test
        evaluations in the same two lists and can no longer be told apart.
        """
        was_training = self.training
        self.model.eval()
        self.training = False

        total_loss: float = 0
        correct: int = 0
        total: int = 0

        # No parameters are updated here, so building the backward graph would only
        # allocate one node per timestep per batch and immediately drop it.
        with no_grad():
            for inputs, targets in dataloader:
                # Forward pass only
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)

                total_loss += float(loss.data)

                # Calculate accuracy (for classification)
                if len(preds.shape) > 1 and preds.shape[1] > 1: # Multi class
                    predictions = np.argmax(preds.data, axis=1)
                    if len(targets.shape) == 1: # Integer targets
                        correct += np.sum(predictions == targets.data)
                    else:
                        correct += np.sum(predictions == np.argmax(targets.data, axis=1))
                    total += len(predictions)

                elif len(preds.shape) > 1 and preds.shape[1] == 1: # Binary class
                    predictions = np.round(preds.data)
                    correct += np.sum(predictions == targets.data)
                    total += len(predictions)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct / total if total > 0 else 0

        if record:
            self.history['accuracy'].append(accuracy)
            self.history['eval_loss'].append(avg_loss)

        # Restore the mode the caller was in rather than assuming it was training.
        if was_training:
            self.model.train()
            self.training = True

        return avg_loss, accuracy

    def _get_model_state(self) -> dict:
        return self.model.state
    
    def _set_model_state(self, state: dict, training: bool) -> None:
        if training:
            self.model.train()
        else:
            self.model.eval()
        self.model.set_state(state)

    def _get_optimizer_state(self):
        return self.optimizer.state
    
    def _set_optimizer_state(self, state: dict) -> None:
        self.optimizer.set_state(state)

    def _get_scheduler_state(self):
        if self.scheduler is not None:
            return self.scheduler.state
        return None
    
    def _set_scheduler_state(self, state: dict) -> None:
        if self.scheduler is not None:
            self.scheduler.set_state(state)
        return None

    @property
    def train_loss(self):
        return self.history['train_loss']

    @property
    def eval_loss(self):
        return self.history['eval_loss']

    def save(self, path: Path|str) -> None:
        """Pickle a full checkpoint: metrics plus model, optimizer and scheduler state."""
        checkpoint = {
            'epoch':            self.epoch,
            'step':             self.step,
            'history':          self.history,
            'training_mode':    self.training,
            'model_state':      self._get_model_state(),
            'optimizer_state':  self._get_optimizer_state(),
            'scheduler_state':  self._get_scheduler_state(),
        }
        self._save_artifact(path, checkpoint)

    def save_metrics(self, path: Path|str) -> None:
        """Pickle epoch, step and history only - no model or optimizer state."""
        checkpoint = {
            'epoch':            self.epoch,
            'step':             self.step,
            'history':          self.history,
        }
        self._save_artifact(path, checkpoint)

    def _save_artifact(self, path: Path|str, artifact: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(artifact, f)

    def load(self, path: Path|str) -> None:
        """Restore a full checkpoint in place: counters, history, mode, and the model,
        optimizer and scheduler state. Requires a `save()` file, not `save_metrics()`."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        self.training = checkpoint['training_mode']

        self._set_model_state(checkpoint['model_state'], checkpoint['training_mode'])
        self._set_optimizer_state(checkpoint['optimizer_state'])
        self._set_scheduler_state(checkpoint['scheduler_state'])
        