"""Full Factorial Experiment"""
import matplotlib.pyplot as plt
from genericpath import isdir
import itertools
import math
from pathlib import Path
import random
import re
import time
import numpy as np
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from os import listdir, path

from thorcino.artifact.artifact import Artifact, load_artifact
from thorcino.dataset.dataset import DataLoader, TensorDataset
from thorcino.training.trainer import Trainer

def updates_per_epoch(n_sequence: int, split_ratio: float, batch_size: int) -> int:
    """Optimizer steps one epoch actually performs.

    Not n_sequence/BATCH_SIZE: only SPLIT_RATIO of the data is trained on, and the
    DataLoader emits a short final batch that still produces a full update. Getting
    this wrong mislabels the updates axis the whole study is plotted against.
    """
    train_size = int(split_ratio * n_sequence)
    return math.ceil(train_size / batch_size)

def epochs_for(updates: int, n_sequence: int, split_ratio: float, batch_size: int) -> int:
    """Epochs needed to reach at least `updates` optimizer steps."""
    return math.ceil(updates / updates_per_epoch(n_sequence, split_ratio, batch_size))

def log_scale(arr: list[float]) -> list[float]:
    c = arr[int(len(arr)/2)]

    log_scale = list(map(
        lambda x: ((math.log2(x) - math.log2(c))), 
        arr
    ))

    return log_scale

## E<experiment id>__<epoch>_<updates>_<n_sequence>_<sequence_length>__<age>s.pkl
ARTIFACT_RE = re.compile(r"^E(\d+)__(\d+)_(\d+)_(\d+)_(\d+)__(\d+)s\.pkl$")

@dataclass(frozen=True)
class ExperimentInfo():
    """The hyperparameters an artifact file name encodes.

    The name is the only record of the setup a checkpoint was produced with, so
    parsing it back is what lets an analysis pass group runs by factor without
    re-opening every pickle.
    """
    experiment_id: int
    epochs: int
    updates: int
    n_sequence: int
    sequence_length: int
    age: int

@dataclass(frozen=True)
class ExperimentArtifact():
    data: Artifact
    metadata: ExperimentInfo


class FullFactorialMetrics:
    def __init__(self, path: str | Path) -> None:
        self.metrics: list[ExperimentArtifact] = []
        self.load_dir(path)

    def load_dir(self, path: str | Path) -> None:
        if not isdir(path):
            raise Exception("It's not a directory, rinvieniti.")
        
        artifacts = listdir(path)
        for artifact in artifacts:
            match = ARTIFACT_RE.match(artifact)
            if match is not None:
                self.metrics.append(load_experiment_artifact(f"{path}/{artifact}"))

def parse_artifact(artifact: str) -> ExperimentInfo | None:
    """Hyperparameters of one artifact name, or None if it is not one of ours.

    Accepts either a bare file name or a path; anything that does not match the
    E<id>__<epoch>_<updates>_<n_seq>_<seq_len>__<age>s.pkl shape is rejected rather
    than half parsed, so a stray file in a checkpoint folder cannot be read as a run.
    """
    match = ARTIFACT_RE.match(path.basename(artifact))
    if match is None:
        return None

    experiment_id, epochs, updates, n_seq, seq_len, age = (int(g) for g in match.groups())

    return ExperimentInfo(experiment_id, epochs, updates, n_seq, seq_len, age)

def load_experiment_artifact(artifact: str|Path) -> ExperimentArtifact:
    info = parse_artifact(artifact)
    data = load_artifact(artifact)

    if info is None or data is None:
        return None
    
    return ExperimentArtifact(data, info)

def index_artifacts(folder: str) -> dict[int, tuple[str, int]]:
    """Map experiment id -> (file name, epoch reached) for one artifact folder."""
    found: dict[int, tuple[str, int]] = {}

    if not path.isdir(folder):
        return found

    for name in listdir(folder):
        match = ARTIFACT_RE.match(name)
        if match is None:
            continue
        found[int(match.group(1))] = (name, int(match.group(2)))

    return found

@dataclass(frozen=True)
class Experiment:
    """The static setup every run of the sweep shares, and the driver that runs it.

    `make_trainer(checkpoint_epochs)` builds a fresh trainer for one grid cell; the
    factor being studied - model, learning rate schedule - stays with the caller,
    while dataset construction, checkpointing and recovery live here.
    """
    batch_size: int
    split_ratio: float
    seed: int
    checkpoint_folder: str = "./checkpoint"

    @property
    def backup_folder(self) -> str:
        return f"{self.checkpoint_folder}/backup"

    @property
    def training_folder(self) -> str:
        return f"{self.checkpoint_folder}/metrics/training"

    @property
    def validation_folder(self) -> str:
        return f"{self.checkpoint_folder}/metrics/validation"

    def validation_set(self, n_sequence: int, sequence_lengths: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
        """One validation set per sequence length, shared across every run.

        Seeded once and kept fixed: a re-run of the notebook must score its models
        against the same data.
        """
        ## Imported here because examples.helpers.dataset imports this module, so a
        ## module level import would close the cycle.
        from examples.helpers.dataset import get_dataset

        np.random.seed(self.seed)

        val_dataset = []
        for seq_len in sequence_lengths:
            X, Y = get_dataset(n_sequence, seq_len)
            val_dataset.append((X, Y))
            print(f'validation set: SEQUENCE_LENGTH={seq_len} X={X.shape} Y={Y.shape} positives={Y.mean():.3f}')

        return val_dataset

    def evaluate(self, trainer: Trainer, val_dataset: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        """Score a checkpoint against every validation length, without touching its history.

        `record=False` matters here: Trainer.eval otherwise appends to the same
        history['eval_loss'] / history['accuracy'] lists the periodic test evaluations
        use, so these entries would interleave with the training curve and every
        later checkpoint would save a curve that no longer lines up with its epochs.
        """
        sequence_lengths, loss_history, accuracy_history = [], [], []
        for X, Y in val_dataset:
            dl = DataLoader(TensorDataset(X, Y), self.batch_size)

            loss, acc = trainer.eval(dl, record=False)

            sequence_lengths.append(X.shape[1])
            loss_history.append(loss)
            accuracy_history.append(acc)

        return {
            'sequence_lengths': sequence_lengths,
            'loss_history': loss_history,
            'accuracy_history': accuracy_history
        }

    def scan_checkpoints(self) -> dict[int, tuple[str, int]]:
        """Experiment ids that wrote all three artifacts, so they can be trusted as done.

        Intersecting the three folders is what makes recovery safe: an id that only
        reached the backup before the notebook was interrupted is left out and simply
        gets recomputed, rather than being counted as complete with metrics missing.
        """
        backups = index_artifacts(self.backup_folder)
        training = index_artifacts(self.training_folder)
        validation = index_artifacts(self.validation_folder)

        complete = backups.keys() & training.keys() & validation.keys()
        partial = (backups.keys() | training.keys() | validation.keys()) - complete

        if partial:
            print(f'ignoring {len(partial)} partially written checkpoint(s): {sorted(partial)}')

        return {i: backups[i] for i in sorted(complete)}

    def run(self,
        make_trainer: Callable[[list[int]], Trainer],
        epochs: int,
        eval_step: int,
        checkpoint_epochs: list[int],
        n_sequence: int,
        sequence_length: int,
        seed: int,
        resume_path: str | None = None,
        skip_epochs: Iterable[int] = (),
    ) -> Generator[tuple[Trainer, int], None, None]:
        """Train one factor cell, yielding (trainer, epoch) at every checkpoint epoch.

        `resume_path` reloads a checkpoint written by a previous run and continues from
        the epoch it stored; `skip_epochs` suppresses the checkpoints that run already
        produced, so an interrupted notebook picks up where it stopped instead of
        recomputing from scratch.
        """
        from examples.helpers.dataset import get_dataset, preprocess

        skip = set(skip_epochs)

        ## Seed both generators: numpy draws the sequences, random drives the
        ## DataLoader shuffle.
        random.seed(seed)
        np.random.seed(seed)

        X, Y = get_dataset(n_sequence, sequence_length)
        train_dl, test_dl = preprocess(X, Y, self.batch_size, self.split_ratio)

        trainer = make_trainer(checkpoint_epochs)

        start_epoch = 0
        if resume_path is not None:
            trainer.load(resume_path)
            start_epoch = trainer.epoch
            print(f'resumed from {resume_path}: EPOCH={start_epoch}, UPDATES={trainer.step}')

        for e in range(start_epoch, epochs):
            _ = trainer.train_epoch(train_dl)

            if e % eval_step == 0:
                _ = trainer.eval(test_dl)

            if e in checkpoint_epochs and e not in skip:
                yield trainer, e

    def plan(self, hyperparams: dict[str, list[int]], n_sequence: int) -> tuple[int, int, list[int]]:
        """(epochs, eval_step, checkpoint epochs) for a cell of `n_sequence` sequences.

        Epochs and checkpoints are derived from the updates one epoch really performs,
        so the budgets in hyperparams['updates'] are the budgets actually trained.
        """
        per_epoch = updates_per_epoch(n_sequence, self.split_ratio, self.batch_size)
        checkpoint_epochs = [
            epochs_for(u, n_sequence, self.split_ratio, self.batch_size) - 1
            for u in hyperparams['updates']
        ]
        epochs = checkpoint_epochs[-1] + 1

        ## Two budgets landing on the same epoch would yield one checkpoint where two
        ## are expected, so the cell could never be marked complete and would be redone
        ## on every re-run. Budgets must be far enough apart for this dataset size.
        assert len(set(checkpoint_epochs)) == len(checkpoint_epochs), (
            f'update budgets {hyperparams["updates"]} collide on epochs {checkpoint_epochs} '
            f'at NUMBER_OF_SEQUENCE={n_sequence} ({per_epoch} updates per epoch)'
        )

        return epochs, max(1, epochs // 10), checkpoint_epochs

    def run_full_factorial(self,
        hyperparams: dict[str, list[int]],
        make_trainer: Callable[[list[int]], Trainer],
        val_dataset: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[int, tuple[str, int]]:
        """Sweep number_of_sequence x sequence_length, harvesting the updates factor.

        The updates factor is not swept: each cell is trained to the largest budget and
        a checkpoint is harvested at every smaller budget along the way, which keeps the
        cost at the largest budget rather than the sum of all of them. Three artifacts
        are written per checkpoint - the full trainer state, the training metrics, and
        the scores against the shared validation set - and cells already on disk are
        skipped or resumed.
        """
        ## Indexing every factor with the same range requires them to be the same length.
        assert len(hyperparams['updates']) == len(hyperparams['number_of_sequence'])
        assert len(hyperparams['number_of_sequence']) == len(hyperparams['sequence_length'])

        n_checkpoints = len(hyperparams['updates'])
        grid = list(itertools.product(range(len(hyperparams['number_of_sequence'])), repeat=2))

        completed = self.scan_checkpoints()
        print(f'recovered {len(completed)} completed checkpoint(s): {sorted(completed)}')

        for i, (idx_n_seq, idx_s_len) in enumerate(grid):
            n_seq = hyperparams['number_of_sequence'][idx_n_seq]
            s_len = hyperparams['sequence_length'][idx_s_len]

            epochs, eval_step, checkpoint_epochs = self.plan(hyperparams, n_seq)
            experiment_ids = [i * n_checkpoints + j for j in range(n_checkpoints)]

            ## Resume from the longest run of checkpoints already on disk. Stopping at the
            ## first gap keeps the resumed trainer's history contiguous: restarting from a
            ## later checkpoint would leave the skipped one permanently missing.
            done = 0
            while done < n_checkpoints and experiment_ids[done] in completed:
                done += 1

            print('-----------------NEW EXPERIMENT STARTED-----------------')
            print(f'experiment hyperparameters: EPOCHS={epochs} UPDATES={hyperparams["updates"][-1]}, NUMBER_OF_SEQUENCE={n_seq}, SEQUENCE_LENGTH={s_len}')
            print(f'updates per epoch: {updates_per_epoch(n_seq, self.split_ratio, self.batch_size)}, checkpoint epochs: {checkpoint_epochs}')

            if done == n_checkpoints:
                print(f'all {n_checkpoints} checkpoints already saved, skipping')
                print('-------------------EXPERIMENT ENDED------------------\n\n')
                continue

            resume_path = None
            if done > 0:
                resume_name, _ = completed[experiment_ids[done - 1]]
                resume_path = f'{self.backup_folder}/{resume_name}'
                print(f'{done} checkpoint(s) already saved, resuming')

            j = done
            start_experiment = time.perf_counter()
            for trainer, act_epoch in self.run(
                make_trainer,
                epochs,
                eval_step,
                checkpoint_epochs,
                n_seq,
                s_len,
                seed=self.seed + i,
                resume_path=resume_path,
                skip_epochs=checkpoint_epochs[:done],
            ):
                act_epoch += 1
                ## The trainer counts the optimizer steps it actually took, so this is the
                ## real update budget of the checkpoint rather than an estimate from n_seq.
                act_updates = trainer.step
                experiment_age = int(time.perf_counter() - start_experiment)

                artifact_name = f"E{experiment_ids[j]}__{act_epoch}_{act_updates}_{n_seq}_{s_len}__{experiment_age}s.pkl"

                trainer.save(f'{self.backup_folder}/{artifact_name}')
                trainer.save_metrics(f'{self.training_folder}/{artifact_name}')
                trainer._save_artifact(f'{self.validation_folder}/{artifact_name}', self.evaluate(trainer, val_dataset))

                ## Record it immediately so an interruption after this point still resumes here.
                completed[experiment_ids[j]] = (artifact_name, act_epoch)
                j += 1
                print(f'saved checkpoint: EPOCHS={act_epoch}, UPDATES={act_updates}, NUMBER_OF_SEQUENCE={n_seq}, SEQUENCE_LENGTH={s_len}')

            ## Measured once at the end: adding up the per-checkpoint elapsed times would
            ## sum a sequence of running totals and roughly double the reported duration.
            print(f'total experiment age: {int(time.perf_counter() - start_experiment)} seconds')
            print('-------------------EXPERIMENT ENDED------------------\n\n')

        return completed


"""Plot the metrics of the full factorial.

Three factors and one response make a 4-dimensional object, while a surface can only
show two factors against one response. The sequence length is therefore spent on the
panels - one surface per level - rather than on an axis, so every panel is a genuine
2-factor slice instead of a projection that hides a factor.
"""

METRIC_LABELS = {
    'loss_history': 'validation loss',
    'accuracy_history': 'validation accuracy',
}

def metric_grid(
    metrics: list[ExperimentArtifact],
    hyperparams: dict,
    key: str = 'loss_history',
    eval_index: int | None = None,
) -> np.ndarray:
    """(number_of_sequence, updates) grid of one validation metric.

    Every artifact stores one score per validation length, so that axis has to be
    collapsed before a surface can be drawn: `eval_index` picks a single validation
    length, `None` averages over all of them. Cells with no artifact stay NaN rather
    than 0, so a hole in the sweep reads as missing instead of as a perfect score.
    """
    n_rows = len(hyperparams['number_of_sequence'])
    n_cols = len(hyperparams['updates'])
    Z = np.full((n_rows, n_cols), np.nan)

    for metric in metrics:
        i = hyperparams['number_of_sequence'].index(metric.metadata.n_sequence)
        j = hyperparams['updates'].index(metric.metadata.updates)

        values = np.asarray(metric.data.data[key], dtype=float)
        Z[i, j] = values[eval_index] if eval_index is not None else values.mean()

    return Z

def plot_metric_surfaces(
    grouped_metrics: list[dict],
    hyperparams: dict,
    key: str = 'loss_history',
    eval_index: int | None = None,
    figsize: tuple[int, int] = (16, 5),
):
    """One surface per sequence length: updates x number of sequences -> metric.

    Both factor axes are log2 spaced and relative to the middle level, because the
    levels grow geometrically: on a linear axis the largest level would dominate the
    plot and the power law would read as a hockey stick. All panels share one colour
    scale and one z range, so their heights can actually be compared.
    """
    log_updates = np.array(log_scale(hyperparams['updates']))
    log_n_seq = np.array(log_scale(hyperparams['number_of_sequence']))
    X, Y = np.meshgrid(log_updates, log_n_seq)

    grids = [
        metric_grid(group['metrics'], hyperparams, key, eval_index)
        for group in grouped_metrics
    ]

    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
    vmin, vmax = float(finite.min()), float(finite.max())

    fig, axes = plt.subplots(
        1, len(grouped_metrics), figsize=figsize, subplot_kw={'projection': '3d'}
    )
    axes = np.atleast_1d(axes)

    surf = None
    for ax, group, Z in zip(axes, grouped_metrics, grids):
        surf = ax.plot_surface(
            X, Y, Z, cmap='viridis', vmin=vmin, vmax=vmax,
            linewidth=0.3, edgecolor='k', antialiased=True, alpha=0.9,
        )
        ## The grid is 3x3, so the surface is four flat quads drawn between the real
        ## measurements. Marking the measurements keeps the interpolation between them
        ## from being read as data.
        ax.scatter(X, Y, Z, color='k', s=12, depthshade=False)

        ax.set_xticks(log_updates)
        ax.set_xticklabels(hyperparams['updates'])
        ax.set_yticks(log_n_seq)
        ax.set_yticklabels(hyperparams['number_of_sequence'])
        ax.set_zlim(vmin, vmax)

        ax.set_xlabel('updates (log2)')
        ax.set_ylabel('number of sequences (log2)')
        ax.set_zlabel(METRIC_LABELS.get(key, key))
        ax.set_title(f"sequence length = {group['sequence_length']}")

    fig.colorbar(surf, ax=axes.tolist(), shrink=0.6, aspect=24,
                 label=METRIC_LABELS.get(key, key))
    fig.suptitle(
        f"{METRIC_LABELS.get(key, key)} "
        f"({'mean over validation lengths' if eval_index is None else f'validation length index {eval_index}'})"
    )

    return fig

def plot_metric_scaling(
    grouped_metrics: list[dict],
    hyperparams: dict,
    key: str = 'loss_history',
    eval_index: int | None = None,
    figsize: tuple[int, int] = (16, 4.5),
):
    """The same data as flat log-log curves: metric vs updates, one line per dataset size.

    A slope on this plot is the exponent of the scaling law, which is the quantity the
    study is after; on the surface that exponent is a tilt the eye has to guess at. The
    surface shows the shape of the response, this shows its rate.
    """
    grids = [
        metric_grid(group['metrics'], hyperparams, key, eval_index)
        for group in grouped_metrics
    ]

    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
    pad = 0.05 * (finite.max() - finite.min() or 1.0)

    fig, axes = plt.subplots(1, len(grouped_metrics), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, group, Z in zip(axes, grouped_metrics, grids):
        for i, n_seq in enumerate(hyperparams['number_of_sequence']):
            ax.plot(hyperparams['updates'], Z[i], marker='o', label=f'n_seq={n_seq}')

        ax.set_xscale('log', base=2)
        if key == 'loss_history':
            ax.set_yscale('log')

        ax.set_xticks(hyperparams['updates'])
        ax.set_xticklabels(hyperparams['updates'])
        ax.set_ylim(finite.min() - pad, finite.max() + pad)
        ax.set_xlabel('updates')
        ax.set_title(f"sequence length = {group['sequence_length']}")
        ax.grid(True, which='both', alpha=0.3)

    axes[0].set_ylabel(METRIC_LABELS.get(key, key))
    axes[-1].legend()
    fig.suptitle(f'{METRIC_LABELS.get(key, key)} scaling')

    return fig
