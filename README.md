# thorcino

A minimal, educational deep-learning framework built from scratch on top of NumPy.
`thorcino` reimplements the essential pieces of a PyTorch-style workflow — a
tensor with reverse-mode automatic differentiation, a small set of layers
(including recurrent ones), activation and loss functions, and a data-loading
pipeline — in a few hundred lines of readable Python.

The goal is not performance but clarity: every gradient is computed by hand in an
explicit backward class, so you can read exactly how backpropagation flows through
the computation graph.

## Quick start

Everything you normally need is re-exported from the top-level package, so a
whole training run fits in a handful of lines:

```python
import numpy as np
from thorcino import Tensor, Sequential, Linear, MSELoss, SGD, Trainer, TensorDataset, DataLoader

rng = np.random.default_rng(0)

# 1. A noisy dataset for y = 2x + 5
x = np.linspace(-5, 5, 200).reshape(-1, 1)
y = 2 * x + 5 + rng.normal(0, 0.5, x.shape)
loader = DataLoader(TensorDataset(Tensor(x), Tensor(y)), batch_size=32, shuffle=True)

# 2. A one-layer model, a loss and an optimizer
model = Sequential(Linear(1, 1))
trainer = Trainer(model, MSELoss(), SGD(model.parameters, lr=1e-2))

# 3. Train
for epoch in range(50):
    loss = trainer.train_epoch(loader)

# 4. Read the recovered coefficients back off the parameters
w, b = model.parameters
print(f"loss={loss:.4f}  w={w.data.item():.3f}  b={b.data.item():.3f}")
# loss=0.2211  w=1.988  b=4.997  <- recovers 2 and 5; Xavier init and the
# shuffling are unseeded, so the exact digits move a little between runs
```

To poke at the autograd engine directly, skip the layers entirely — any
`Tensor` produced by an operation carries the backward node that made it, and
`backward()` walks that graph for you:

```python
from thorcino import Tensor

a = Tensor([[1.0, 2.0]])
b = Tensor([[3.0], [4.0]])

c = (a @ b).sum()
c.backward()

print(c.data)              # 11.0
print(a.grad)              # [[3. 4.]]  -> dc/da = bᵀ
print(b.grad)              # [[1.] [2.]] -> dc/db = aᵀ
print(type(c._grad_fn))    # <class 'thorcino.autograd.arithmetic.SumBackward'>
```

## Requirements

- Python >= 3.14
- NumPy >= 2.5.0
- Graphviz >= 0.21
- scikit-learn >= 1.9.0 (dataset generators used by the examples)
- pandas >= 3.0.3

Optional (dev): `ipykernel`, `matplotlib`, `tabulate`, `myst_nb` (used by the
examples and notebooks), `flake8` and `pytest` (used by CI).

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for environment and
dependency management, with `hatchling` as the build backend.

```bash
uv sync --all-groups
```

This creates a `.venv` and installs the runtime and dev dependencies. The lock
file and build artifacts (`uv.lock`, `dist/`) are not tracked in git.

Continuous integration ([`.github/workflows/python-app.yml`](.github/workflows/python-app.yml))
runs the same `uv sync --all-groups`, then lints with `flake8` and runs
`pytest`. There is no test suite yet, so the pytest step currently tolerates
the "no tests collected" exit code.

## Architecture

The whole library lives under `thorcino/` and is organized in a small number of
single-responsibility modules:

```
thorcino/
├── __init__.py          # Flat public API: re-exports the names below
├── tensor.py            # Tensor: the numpy-backed frontend + operator overloading
├── functions.py         # Pure numpy math: activations, softmax, loss functions
├── activations.py       # Activation layers (Identity, ReLU, Sigmoid, Tanh, GELU, Softmax)
├── losses.py            # Loss objects (MSE, CrossEntropy, BinaryCrossEntropy)
├── optimizer.py         # Optimizer, SGD, SGD_DL2, SGDM, Adam, AdamW
├── graph.py             # ComputationalGraph: graphviz visualisation of a Sequential model
├── consts.py            # Parameter role tags (WEIGHTS_ROLE, BIAS_ROLE)
├── utils.py             # unbroadcast() helper for gradient reduction
├── layers/               # Network layers
│   ├── layer.py          #   Layer: abstract base (forward, train/eval, parameters)
│   ├── linear.py         #   Linear
│   ├── dropout.py        #   Dropout
│   ├── rnn.py            #   RNN (unrolled recurrence, BPTT via autograd)
│   ├── lstm.py           #   LSTM (input/forget/output gates + cell state)
│   └── sequential.py     #   Sequential + graph rendering helpers
├── autograd/            # Reverse-mode automatic differentiation
│   ├── base.py          #   Function: base class for every backward node
│   ├── arithmetic.py    #   Identity/Add/Sub/Mul/Div/Matmul/Sum/Reshape/Transpose/Stack backward
│   ├── activations.py   #   ReLU/Sigmoid/Tanh/GELU/Softmax backward
│   └── losses.py        #   MSE/CrossEntropy/BCE backward
├── dataset/              # Data loading pipeline
│   ├── dataset.py        #   Dataset, TensorDataset, ImageDataset, DataLoader
│   ├── transformation.py #   RandomHorizontalFlip, RandomCrop, Compose
│   └── utils.py          #   image loading helpers
└── training/             # Training loop orchestration
    ├── trainer.py         #   Trainer: train_epoch/eval, checkpointing, clip_grad_norm
    └── schedulers.py      #   Schedule, CosineSchedule
```

The design follows a clear **frontend / backend split**:

- `Tensor` is the *frontend*. It wraps a NumPy array, overloads the Python
  operators (`+`, `-`, `*`, `/`, `@`, …) and records the operation that produced
  it in a `_grad_fn` attribute.
- The `autograd` package is the *backend*. Every operation has a matching
  `*Backward` class (a `Function`) that knows how to turn an upstream gradient
  into the gradients of its inputs.

Layers, activations and losses are thin objects that call into `functions.py` for
the forward pass and attach the corresponding `Function` for the backward pass.

`thorcino/__init__.py` flattens all of this into a single import surface — the
tensor, every layer and activation, the losses, the optimizers, the trainer and
the data-loading pieces — so day-to-day use never needs the module paths. They
stay available if you want to read the code alongside it.

## Automatic differentiation (`thorcino/autograd/`)

`thorcino` implements **reverse-mode autodiff** by building a dynamic graph as
operations execute (define-by-run), then walking it backwards to accumulate
gradients.

### The `Function` node

Every backward node subclasses `Function` (`thorcino/autograd/base.py`):

```python
class Function:
    def __init__(self, *tensors):
        self.saved_tensors = tensors                         # inputs needed by backward
        self.next_functions = [t._grad_fn for t in tensors]  # links to parent nodes

    def apply(self, grad_output):
        """Turn the upstream gradient into gradients for each input."""
        raise NotImplementedError()
```

- `saved_tensors` holds the operands captured during the forward pass.
- `next_functions` records each operand's own `_grad_fn`, which is what turns the
  set of nodes into a traversable graph.
- `apply(grad_output)` implements the chain rule for that specific operation and
  returns one gradient per input.

### How the graph is built

When you write `c = a + b`, `Tensor.__add__` computes the numeric result with
NumPy and attaches the backward node:

```python
out = Tensor(self.data + other.data)
out._grad_fn = AddBackward(self, other)
```

So each output tensor remembers *how it was produced*. Chaining operations
produces a graph of `Function` nodes rooted at the final output.

### The backward pass

`Tensor.backward()` (`thorcino/tensor.py`) drives backpropagation recursively:

1. If no gradient is supplied, it seeds `1.0` for a scalar output (and raises for
   non-scalar outputs, matching PyTorch's behaviour).
2. It accumulates the incoming gradient into `self.grad` (gradients **add up**,
   which is what makes shared subgraphs correct).
3. It calls `self._grad_fn.apply(gradient)` to get the input gradients, then
   recurses into each input tensor that `requires_grad`.

Broadcasting is handled by `unbroadcast()` (`thorcino/utils.py`), which sums a
gradient back down to the shape of the original operand so that broadcasted
operations (e.g. adding a bias vector to a batch) produce correctly-shaped
gradients.

### Managing the graph

- `Tensor.zero_grad()` resets a tensor's accumulated gradient.
- `Tensor.destroy_graph()` walks the graph and drops every `_grad_fn`, freeing the
  saved tensors so the graph can be garbage-collected between iterations.

### Supported backward operations

| Category    | Backward classes |
|-------------|------------------|
| Arithmetic  | `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward`, `IdentityBackward` |
| Linear alg. | `MatmulBackward`, `TransposeBackward` |
| Reductions  | `SumBackward` |
| Shape       | `ReshapeBackward`, `StackBackward` |
| Activations | `ReLUBackward`, `SigmoidBackward`, `TanhBackward`, `GELUBackward`, `SoftmaxBackward` |
| Losses      | `MSELossBackward`, `CrossEntropyLossBackward`, `BCELossBackward` |

## The `Tensor` class (`thorcino/tensor.py`)

`Tensor` is a lightweight wrapper around a `np.ndarray` (always stored as
`float32`). It exposes:

- **Metadata**: `data`, `shape`, `size`, `dim`, `dtype`, `requires_grad`, `grad`,
  `_grad_fn`, `role`. The optional `role` tags a tensor as `weights` or `bias`
  (see `thorcino/consts.py`); it drives per-role weight decay in the optimizers
  and the colour coding in the graph renderer.
- **Operator overloading**: `__add__`/`__radd__`, `__sub__`/`__rsub__`,
  `__mul__`/`__rmul__`, `__truediv__`, `__matmul__`, `__pow__`, `__neg__`,
  `__gt__`. The autograd-aware operations (`+`, `-`, `*`, `/`, `@`) attach a
  `_grad_fn`; scalar/`ndarray` fast paths return plain results.
- **Tensor ops**: `matmul`, `reshape` (supports `-1` inference), `transpose`,
  `sum`, `mean`, `max`, `min`, plus `__getitem__`/`__len__` for indexing and
  iteration.
- **Stacking**: the `Tensor.stack(tensors, axis=0)` static method stacks a list
  of tensors while keeping them wired into the graph via `StackBackward` —
  unlike `Tensor(np.stack(...))`, which builds a fresh leaf with no `_grad_fn`.
  This is what lets the recurrent layers return one output per time step and
  still backpropagate through time.
- **Autograd control**: `backward()`, `zero_grad()`, `destroy_graph()`.
- **Interop**: `numpy()` returns the underlying array.

A convenience path in `__init__` lets you build a batched tensor from a list of
tensors — `Tensor([t1, t2, ...])` stacks their data automatically.

Note that `thorcino.tensor` imports the backward classes *lazily*, inside the
method bodies that need them (plus a `TYPE_CHECKING`-only import for the type
annotations), to break the circular import between the tensor frontend and the
autograd backend — the backward classes need `Tensor` at runtime.

## Layers (`thorcino/layers/`)

All layers derive from the abstract `Layer` base class, which defines
`forward()`, `train()` and `eval()`, makes instances callable, and exposes a
`parameters` property. `train()`/`eval()` flip each layer's `training` flag and
the `requires_grad` of its parameters.

| Layer        | Description |
|--------------|-------------|
| `Linear`     | Fully-connected layer `y = xW + b` with Xavier weight initialization and optional bias. |
| `Dropout`    | Inverted dropout with keep-probability scaling; a no-op when `training=False`. |
| `RNN`        | Vanilla recurrent layer: `H_t = φ_h(X_t·W_xh + H_{t−1}·W_hh + b_h)`, `O_t = φ_o(H_t·W_ho + b_o)`. Unrolls over the sequence axis and returns every step's output stacked along `axis=1`; BPTT falls out of the autograd graph. Takes explicit `activation_h`/`activation_o` layers. |
| `LSTM`       | Long short-term memory cell with input/forget/output gates and a cell state, unrolled the same way and returning the per-step hidden states. |
| `Sequential` | Chains layers and forwards through them in order; aggregates their parameters, and propagates `train()`/`eval()`. |

Both recurrent layers are re-exported from the top-level package
(`from thorcino import RNN, LSTM`). Note that `thorcino.layers` itself only
re-exports `RNN`, so importing `LSTM` from that sub-package needs its module
path (`from thorcino.layers.lstm import LSTM`).

`Sequential.save_graph(path, arch=True, forward=False, backward=False)` renders a
`.png` of the model via `thorcino/graph.py` (needs `graphviz`): a cluster per layer
for the architecture, and — if requested — the forward/backward computational
graphs built from a synthetic input, tensors colour-coded by role
(input/weights/bias/hidden). `build_graph()` builds and caches the graph without
rendering it, and `destroy_graph()` drops the cached one.

## Activation functions (`thorcino/activations.py`)

Each activation is available both as a pure NumPy function (`thorcino/functions.py`)
and as an autograd-aware `Layer`:

| Activation | Notes |
|------------|-------|
| `Identity` | `x` unchanged; useful as an explicit pass-through for the `RNN` layer's activation slots. |
| `ReLU`     | `max(0, x)` |
| `Sigmoid`  | Numerically stable (branch on the sign of the input) |
| `Tanh`     | `np.tanh` |
| `GELU`     | Sigmoid approximation `x · σ(1.702·x)` |
| `Softmax`  | Max-shifted for stability; configurable `dim` |

`functions.py` also provides a stable `log_softmax`, used internally by the
cross-entropy loss.

## Loss functions (`thorcino/losses.py`)

| Loss                     | Input | Notes |
|--------------------------|-------|-------|
| `MSELoss`                | predictions, targets | Mean squared error. |
| `CrossEntropyLoss`       | logits, integer targets | Combines a stable `log_softmax` with negative log-likelihood; the backward is the classic `softmax(logits) − onehot(targets)`. |
| `BinaryCrossEntropyLoss` | probabilities, targets | Clips predictions to `[1e-7, 1 − 1e-7]` to avoid `log(0)`. |

All three derive from the abstract `Loss` base class, which pairs a forward pass
with the matching `grad_fn`. Each loss is callable (`loss(pred, target)`) and
returns a scalar `Tensor` you can call `.backward()` on.

## Optimizers (`thorcino/optimizer.py`)

| Optimizer | Notes |
|-----------|-------|
| `SGD`     | Plain gradient descent with optional L2 weight decay, applied only to tensors tagged `weights`. |
| `SGD_DL2` | Same, with *independent* `weight_decay` and `bias_decay` terms, so weights and biases can be regularized separately (used by the `regularization/L2` example). |
| `SGDM`    | SGD with momentum. |
| `Adam`    | Adaptive moments with bias correction. |
| `AdamW`   | Adam with decoupled weight decay. |

Every optimizer takes `model.parameters` and a learning rate; `step()` updates
`param.data` in place, `zero_grad()` clears `param.grad`, and `get_state()`
returns the optimizer's hyperparameters/buffers for checkpointing.

## Training loop (`thorcino/training/`)

- **`Trainer`** (`trainer.py`) wraps a model, loss, optimizer and optional
  scheduler. `train_epoch(dataloader, accumulation_steps=1)` runs one epoch
  (with gradient accumulation and optional gradient clipping, enabled by the
  `grad_clip_norm` constructor argument) and returns the average per-batch
  loss; `eval(dataloader)` runs a pass with `requires_grad` switched off and
  returns `(avg_loss, accuracy)` — accuracy is `0.0` for non-classification
  (1-D) outputs. Both log into `trainer.history` (`train_loss`, `eval_loss`,
  `lr`), also reachable via the `train_loss`/`eval_loss` properties.
  `save()` pickles the full training state to a checkpoint file; `load()`
  currently restores only `epoch`, `step`, `history` and the training flag —
  restoring model/optimizer/scheduler state is left out on purpose, to keep
  the checkpointing code readable.
- **`clip_grad_norm(parameters, max_norm=1.0)`** rescales every parameter
  gradient in place so their combined L2 norm stays within `max_norm`, and
  returns the pre-clip norm. `Trainer` calls it for you, but it is exported
  from `thorcino.training` for standalone use.
- **`Schedule`** (`schedulers.py`) is the abstract base for learning-rate
  schedules; `CosineSchedule(max_lr, min_lr, total_epochs)` anneals the
  learning rate from `max_lr` to `min_lr` following a cosine curve, applied by
  `Trainer` at the end of every `train_epoch()` call.

## Data loading (`thorcino/dataset/`)

The module mirrors the PyTorch `Dataset` / `DataLoader` pattern.

- **`Dataset`** — abstract base defining `__len__` and `__getitem__`.
- **`TensorDataset`** — wraps in-memory tensors and validates that they share the
  same length along dimension 0.
- **`ImageDataset`** — lazily loads images from disk on access (via `load_jpeg`),
  pairing each with its label. Note that `load_jpeg` (`dataset/utils.py`) is
  still a stub, so this dataset is scaffolding rather than a working loader.
- **`DataLoader`** — iterates a `Dataset` in mini-batches, with optional
  shuffling, and collates each batch by stacking samples along a new leading
  (batch) axis.

Data augmentation transforms live in `transformation.py`:

- `RandomHorizontalFlip(p)` — flips along the width axis with probability `p`.
- `RandomCrop(height, width, padding)` — zero-pads then crops a random window.
- `Compose([...])` — chains transforms into a single callable.

## Examples

See [`examples/linear_regression/`](examples/linear_regression/) for a family
of end-to-end regression scripts and notebooks built on `Trainer` +
`DataLoader`:

| Example | Description |
|---|---|
| [`linear/`](examples/linear_regression/linear/README.md) | Recovers the slope/intercept of `2·x + 5` with a single `Linear(1, 1)` layer; compares the learned fit against the closed-form `lstsq` solution. |
| [`quadratic/`](examples/linear_regression/quadratic/README.md) | Recovers the coefficients of `x² + 2·x + 2` via feature expansion (`[x, x²]`) fed into a `Linear(2, 1)` layer. |
| [`cubic/`](examples/linear_regression/cubic/cubic.py) | Same idea one degree further: recovers `1.2·x³ − 2.3·x² + 2·x + 2` with a `Linear(3, 1)` layer over `[x, x², x³]`. |
| [`ill-cond/`](examples/linear_regression/ill-cond/README.md) | Notebook comparing closed-form OLS vs. SGD on ill-conditioned (near-collinear) features, showing OLS's coefficients blow up while SGD's stay stable. |
| [`variance/`](examples/linear_regression/variance/README.md) | Notebook fitting 20 repeated `Linear(1, 1)` models at each of nine label-noise levels, showing via boxplots how the loss and learned slope/intercept drift and spread as noise grows. |
| [`regularization/`](examples/linear_regression/regularization/README.md) | Two notebooks on weight decay as variance reduction: [`L2/`](examples/linear_regression/regularization/L2/README.md) sweeps the decay strength (with separate weights/bias decay via `SGD_DL2`) and watches the parameter spread shrink; [`DL2/`](examples/linear_regression/regularization/DL2/README.md) isolates decay from noise across repeated trials (work in progress — only the no-decay baseline is implemented). |

The [`linear_regression/README.md`](examples/linear_regression/README.md)
ties the linear/quadratic/cubic scripts together and explains why the learning
rate has to shrink as the polynomial degree grows. The notebooks share the
dataset/training/plotting helpers in
[`linear_regression/helpers/`](examples/linear_regression/helpers/).

Beyond regression, [`examples/rnn/`](examples/rnn/README.md) is a full
walkthrough of a recurrent network on a next-number-in-sequence task, in two
notebooks: `scratch.ipynb` hand-codes the forward recurrence, BPTT and the SGD
update with raw NumPy, then `main.ipynb` rebuilds the same model on
`thorcino.layers.RNN` + `Trainer` and renders its architecture and
forward/backward graphs with `model.save_graph(...)`. Its README derives the
BPTT gradients in full. [`examples/lstm/`](examples/lstm/) is reserved for the
same treatment of the `LSTM` layer and is still an empty notebook.

```bash
uv run python examples/linear_regression/linear/linear.py
```

## Roadmap

Planned work is tracked in [`TODO.md`](TODO.md), and includes:

- **Loader**: parallel loading via multithreading; prefetching of the next batch.
- **Autograd**: move backward computation entirely onto NumPy arrays (keeping
  `Tensor` as a pure frontend); cache forward intermediates for reuse in backward;
  add a debug step that reports which node a backward failure occurred on.

Known gaps in the current tree: `dataset/utils.py:load_jpeg` is a stub, the
`LSTM` layer has no example yet, and there is no test suite (the CI pytest step
is wired up but collects nothing).

## License

[MIT](LICENSE) © cecinuga
