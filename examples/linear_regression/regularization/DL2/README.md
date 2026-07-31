# Regularization — Weights & Biases Decay

> **Work in progress.** Only Scenario 1 (no weight decay) is implemented so
> far; the notebook's last cell is currently empty, waiting on Scenarios 0,
> 2 and 3.

A **tiny-torch** notebook that isolates the effect of L2 decay from the
noise level itself. Unlike [`../L2/README.md`](../L2/README.md), which
sweeps both the noise level and the decay strength, this notebook fixes the
noise level and draws `N_TRIALS` independent noisy realizations of the same
signal `f(x) = SLOPE·x + INTERCEPT` for each scenario — so any change in the
spread of the learned weight/bias across scenarios can only come from decay,
not from a change in noise level.

```python
SLOPE = 3.23
INTERCEPT = 7.6

def f(x: np.ndarray) -> np.ndarray:
    return x * SLOPE + INTERCEPT

SEED = 888
NOISE = 100
MIN, MAX = -7, 7
SAMPLE_SIZE = 120

repeats = generate_noise_dataset(f, N_TRIALS, SAMPLE_SIZE, NOISE, MIN, MAX, SEED)
```

`split_dataset` carves an 80/20 train/test split (`SPLIT_RATIO = 0.8`),
wrapped in the usual `TensorDataset` + `DataLoader(batch_size=16)` via
`preprocess_dataloader`.

---

## Four planned scenarios

The noise level stays fixed throughout; only the training regime changes:

0. **No variance baseline** — same dataset, very little noise. 🚧 not yet
   implemented
1. **No weight decay** — several trials on independent noisy draws, to
   establish a baseline spread. ✅ implemented
2. **Weight decay** — several trials per decay coefficient, sweeping the
   decay strength upward. 🚧 not yet implemented
3. **Weight decay + bias decay** — same sweep, but decaying both parameters
   instead of just the weight. 🚧 not yet implemented

---

## SCENARIO 1 — No weight decay (baseline)

`N_TRIALS = 4` fresh `Trainer`s (`Sequential(Linear(1, 1))`, `MSELoss`, `SGD`,
`CosineSchedule`) are fit on `N_TRIALS` independent noisy realizations of the
same signal `f(x)`, one after another:

```python
EPOCHS = 80
N_TRIALS = 4
EVAL_STEP = 10
SPLIT_RATIO = 0.8
MAX_LR, MIN_LR = 1e-2, 1e-4
IN_FEATURE, OUT_FEATURE = 1, 1

repeats = generate_noise_dataset(f, N_TRIALS, SAMPLE_SIZE, NOISE, MIN, MAX, SEED)
tr, te = split_dataset(repeats, split_ratio=SPLIT_RATIO, axis=1)

fitted_trainers: list[Trainer] = []

for sample_tr, sample_te in zip(tr, te):
    loader_tr, loader_te = preprocess_dataloader(sample_tr, sample_te, BATCH_SIZE)
    trainer = create_trainer(IN_FEATURE, OUT_FEATURE, MIN_LR, MAX_LR, EPOCHS)
    fit_model(trainer, loader_tr, loader_te, EPOCHS, EVAL_STEP)
    fitted_trainers.append(trainer)
```

The train and eval loss curves of every trial are plotted on shared axes,
each overlaid with a dashed line marking the closed-form MSE of the true
`SLOPE`/`INTERCEPT` on that trial's data, and `get_models_params` pulls out
the final learned weight and bias of each trial, shown as bar charts against
the true `SLOPE`/`INTERCEPT`.

For each trial, the notebook also plots the train/test samples, the model's
learned prediction line, and the true signal `f(x)` side by side, to
visualize how each fit compares to the ground truth.

**Consideration:** with no decay and independent noisy draws (`NOISE = 100`)
of the same signal, repeated trials still land on visibly different weight
and bias values — the baseline variance that Scenario 0's near-noise-free
control will contrast against, and that Scenarios 2 and 3 are meant to
shrink via decay.

---

## Run it

```bash
jupyter notebook examples/linear_regression/regularization/DL2/main.ipynb
```

As it stands, running top to bottom stops after Scenario 1 — the final cell
is an empty placeholder for Scenario 0 and the weight-decay sweeps.
