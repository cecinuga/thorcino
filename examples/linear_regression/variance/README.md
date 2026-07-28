# Variance in Linear Regression

A **tiny-torch** notebook that studies how the amount of noise in the
training data affects a linear regression model. Instead of fitting one
model to one dataset, it fits **nine** models to nine datasets sampled from
the same underlying line but with progressively more noise, and compares how
each one trains and what it learns.

The signal to recover is:

```
f(x) = 3.5·x + 7.1   →   slope = 3.5, intercept = 7.1
```

Every model only ever sees noisy `(x, y)` pairs — the goal is to see how far
increasing noise pushes the learned slope and intercept away from these true
values.

---

## Building the datasets

`generate_dataset(size, sample_size, noise_upto)` draws a single set of `x`
samples on `[-7, 7]` and computes the clean signal `y = f(x)` once, then
builds `size` variants of it by adding Gaussian noise of increasing standard
deviation to `y`:

```python
for noise in np.linspace(0, noise_upto, size):
    y_noise = y + rng.standard_normal(x.shape[0]) * noise
    ...
```

The noise levels are equally spaced between `0` and `noise_upto`, so the
first dataset in the stack is the clean signal itself and the last one is the
noisiest. The notebook builds `DATASETS_SIZE = 9` datasets of
`SAMPLE_SIZE = 100` points each, with `NOISE_UPTO = 100`, plus a
noise-free reference (`noise_upto=0`) used purely to draw the true line
alongside the noisy scatter:

```python
X = generate_dataset(size=DATASETS_SIZE, sample_size=SAMPLE_SIZE, noise_upto=NOISE_UPTO)
X_clean = generate_dataset(size=DATASETS_SIZE, sample_size=SAMPLE_SIZE, noise_upto=0)
```

Both datasets share the same `SEED`, so the noisy samples and the clean line
line up on the same `x` values, one subplot per noise level.

---

## Training nine models

One `Trainer` is created per noise level, all with an identical
architecture (`Linear(1, 1)`) and identical hyperparameters — only the data
they see differs:

```python
EPOCHS = 500
EVAL_STEP = 10
BATCH_SIZE = 16
MAX_LR, MIN_LR = 1e-2, 1e-4

X_tr, X_te = split_dataset(X, split_ratio=0.9, axis=1)

for i in range(DATASETS_SIZE):
    trainer = create_trainer(X.shape[2]-1, X.shape[2]-1, MIN_LR, MAX_LR, EPOCHS)
    trainers.append(trainer)

fit_models(trainers, X_tr, X_te, EPOCHS, EVAL_STEP, BATCH_SIZE)
```

`create_trainer` wires up a `Sequential(Linear(1, 1))` with `MSELoss`, `SGD`
and a `CosineSchedule` annealing from `MAX_LR` to `MIN_LR`. `fit_models`
loops over the `(trainer, dataset)` pairs, wraps each split into a
`TensorDataset` + `DataLoader` (batch size `16`), and trains every model for
`EPOCHS` full passes, evaluating every `EVAL_STEP` epochs.

---

## Loss history

For each of the nine models, the train loss (blue) and eval loss (red) are
plotted side by side, one subplot per noise level:

```python
train_loss_history.append([range(EPOCHS), trainer.train_loss])
eval_loss_history.append([range(0, EPOCHS, EVAL_STEP), trainer.eval_loss])
```

Laying the nine curves out together makes the effect of noise visible
directly: models trained on low-noise datasets converge to a small, tight
loss, while models trained on high-noise datasets settle at a much higher
loss floor and show a wider gap between train and eval loss.

---

## Weights interpretation

Finally, the notebook extracts the learned `weights` and `bias` from every
trained model:

```python
def get_model_param(model: Sequential) -> np.ndarray:
    weights, bias = model.parameters
    return np.array([weights, bias.reshape(-1, 1)])
```

and plots them as bar charts, one per noise level, blue bars for the learned
`[weight, bias]` pair against red bars for the true coefficients
`[SLOPE, INTERCEPT] = [3.5, 7.1]`:

```python
models_params = get_models_params([trainer.model for trainer in trainers])
bar_dataset(models_params, coefs, fig, axs)
```

At low noise the blue and red bars line up almost exactly. As the noise level
increases across the nine subplots, the learned weight and bias drift further
from the true coefficients — the clearest, most direct evidence that more
variance in the training data makes the fitted parameters less reliable, even
when the loss on that same noisy data looks acceptable.

---

## Why it happens

Ordinary least squares (and SGD, which approximates it) is unbiased in
expectation but not exact on any single finite sample. The variance of the
estimated slope and intercept grows with the variance of the noise in `y`
and shrinks as the number of samples grows — so for a fixed `sample_size`,
pushing `noise_upto` up spreads the `(x, y)` points further off the true
line, and gradient descent has to split the difference among noisier
observations. The loss can still look small (it's measured against noisy
targets, not the clean signal), while the parameters themselves wander
further from `SLOPE` and `INTERCEPT`. This is the same distinction explored
in [`../ill-cond/README.md`](../ill-cond/README.md): fitting the data well and
recovering the true, interpretable coefficients are two different things.

---

## Run it

Open and run the notebook top to bottom:

```bash
jupyter notebook examples/linear_regression/variance/main.ipynb
```