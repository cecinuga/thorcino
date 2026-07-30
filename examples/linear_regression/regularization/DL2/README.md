# Regularization — Weights & Biases Decay

> **Work in progress.** Only Scenario 1 (no weight decay) is implemented so
> far; the notebook's last cell is currently empty, waiting on Scenarios 2
> and 3.

A **tiny-torch** notebook that isolates the effect of L2 decay from the
dataset itself. Unlike [`../L2/README.md`](../L2/README.md), which sweeps
both the noise level and the decay strength across freshly resampled
datasets, this notebook generates **one** noisy dataset with
`sklearn.datasets.make_regression` and reuses it for every trial — so any
change in the spread of the learned weight/bias can only come from decay,
not from a new draw of noise.

```python
SEED = 888
BIAS = 7.2
NOISE = 150

X, y, true_coefs = make_regression(
    n_samples=100, n_features=1, n_targets=1,
    bias=BIAS, noise=NOISE, coef=True, random_state=SEED,
)
```

`split_dataset` carves out a 90/10 train/test split, wrapped in the usual
`TensorDataset` + `DataLoader(batch_size=16)`.

---

## Three planned scenarios

The dataset stays fixed throughout; only the training regime changes:

1. **No weight decay** — several trials on the same dataset, to establish a
   baseline spread. ✅ implemented
2. **Weight decay** — several trials per decay coefficient, sweeping the
   decay strength upward. 🚧 not yet implemented
3. **Weight decay + bias decay** — same sweep, but decaying both parameters
   instead of just the weight. 🚧 not yet implemented

---

## SCENARIO 1 — No weight decay (baseline)

`N_TRIALS = 5` fresh `Trainer`s (`Sequential(Linear(1, 1))`, `MSELoss`, `SGD`,
`CosineSchedule`) are fit on the **same** train/test split, one after another:

```python
EPOCHS = 80
EVAL_STEP = 10
N_TRIALS = 5
MAX_LR, MIN_LR = 1e-2, 1e-4

trainers = [create_trainer(1, 1, MIN_LR, MAX_LR, EPOCHS) for _ in range(N_TRIALS)]

for trainer in trainers:
    fit_model(trainer, tr_dataloader, te_dataloader, EPOCHS, EVAL_STEP)
```

The train and eval loss curves of every trial are plotted on shared axes, and
`get_models_params` pulls out the final learned weight and bias of each
trial, shown as bar charts against the `true_coefs`/`BIAS` returned by
`make_regression`.

**Consideration:** with no decay and a fixed, high-noise (`NOISE = 150`)
dataset, repeated trials still land on visibly different weight and bias
values — the baseline variance that Scenarios 2 and 3 are meant to shrink.

---

## Run it

```bash
jupyter notebook examples/linear_regression/regularization/DL2/main.ipynb
```

As it stands, running top to bottom stops after Scenario 1 — the final cell
is an empty placeholder for the weight-decay sweep.
