# tiny-torch

A minimal, educational deep-learning framework built from scratch on top of NumPy.
`tiny-torch` reimplements the essential pieces of a PyTorch-style workflow — a
tensor with reverse-mode automatic differentiation, a small set of layers,
activation and loss functions, and a data-loading pipeline — in a few hundred
lines of readable Python.

The goal is not performance but clarity: every gradient is computed by hand in an
explicit backward class, so you can read exactly how backpropagation flows through
the computation graph.

## Requirements

- Python >= 3.14
- NumPy >= 2.5.0

Optional (dev): `ipykernel`, `matplotlib` (used by the examples).

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for environment and
dependency management (`uv.lock` is committed).

```bash
uv sync
```

This creates a `.venv` and installs the runtime and dev dependencies. Prebuilt
artifacts for `tiny_torch-0.1.0` are also available under `dist/`.

## Architecture

The whole library lives under `core/` and is organized in a small number of
single-responsibility modules:

```
core/
├── tensor.py            # Tensor: the numpy-backed frontend + operator overloading
├── functions.py         # Pure numpy math: activations, softmax, loss functions
├── activations.py       # Activation layers (ReLU, Sigmoid, Tanh, GELU, Softmax)
├── losses.py            # Loss objects (MSE, CrossEntropy, BinaryCrossEntropy)
├── layers.py            # Layer, Linear, Dropout, Sequential
├── optimizer.py         # Optimizer, SGD, SGDM, Adam, AdamW
├── graph.py             # ComputationalGraph: graphviz visualisation of a Sequential model
├── utils.py             # unbroadcast() helper for gradient reduction
├── autograd/            # Reverse-mode automatic differentiation
│   ├── base.py          #   Function: base class for every backward node
│   ├── arithmetic.py    #   Add/Sub/Mul/Div/Matmul/Sum/Reshape/Transpose backward
│   ├── activations.py   #   ReLU/Sigmoid/Tanh/GELU/Softmax backward
│   └── losses.py        #   MSE/CrossEntropy/BCE backward
├── dataset/              # Data loading pipeline
│   ├── dataset.py        #   Dataset, TensorDataset, ImageDataset, DataLoader
│   ├── transformation.py #   RandomHorizontalFlip, RandomCrop, Compose
│   └── utils.py          #   image loading helpers
└── training/             # Training loop orchestration
    ├── trainer.py         #   Trainer: train_epoch/eval, checkpointing, grad clipping
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

## Automatic differentiation (`core/autograd/`)

`tiny-torch` implements **reverse-mode autodiff** by building a dynamic graph as
operations execute (define-by-run), then walking it backwards to accumulate
gradients.

### The `Function` node

Every backward node subclasses `Function` (`core/autograd/base.py`):

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

`Tensor.backward()` (`core/tensor.py`) drives backpropagation recursively:

1. If no gradient is supplied, it seeds `1.0` for a scalar output (and raises for
   non-scalar outputs, matching PyTorch's behaviour).
2. It accumulates the incoming gradient into `self.grad` (gradients **add up**,
   which is what makes shared subgraphs correct).
3. It calls `self._grad_fn.apply(gradient)` to get the input gradients, then
   recurses into each input tensor that `requires_grad`.

Broadcasting is handled by `unbroadcast()` (`core/utils.py`), which sums a
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
| Arithmetic  | `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward` |
| Linear alg. | `MatmulBackward`, `TransposeBackward` |
| Reductions  | `SumBackward` |
| Shape       | `ReshapeBackward` |
| Activations | `ReLUBackward`, `SigmoidBackward`, `TanhBackward`, `GELUBackward`, `SoftmaxBackward` |
| Losses      | `MSELossBackward`, `CrossEntropyLossBackward`, `BCELossBackward` |

## The `Tensor` class (`core/tensor.py`)

`Tensor` is a lightweight wrapper around a `np.ndarray` (always stored as
`float32`). It exposes:

- **Metadata**: `data`, `shape`, `size`, `dtype`, `requires_grad`, `grad`,
  `_grad_fn`.
- **Operator overloading**: `__add__`/`__radd__`, `__sub__`/`__rsub__`,
  `__mul__`/`__rmul__`, `__truediv__`, `__matmul__`, `__pow__`, `__neg__`,
  `__gt__`. The autograd-aware operations (`+`, `-`, `*`, `/`, `@`) attach a
  `_grad_fn`; scalar/`ndarray` fast paths return plain results.
- **Tensor ops**: `matmul`, `reshape` (supports `-1` inference), `transpose`,
  `sum`, `mean`, `max`, `min`.
- **Autograd control**: `backward()`, `zero_grad()`, `destroy_graph()`.
- **Interop**: `numpy()` returns the underlying array.

A convenience path in `__init__` lets you build a batched tensor from a list of
tensors — `Tensor([t1, t2, ...])` stacks their data automatically.

Note that `core.tensor` imports the backward classes at the *bottom* of the file,
after `Tensor` is defined, to break the circular import between the tensor
frontend and the autograd backend (the backward classes need `Tensor` at runtime).

## Layers (`core/layers.py`)

All layers derive from the abstract `Layer` base class, which defines `forward()`,
makes instances callable, and exposes a `parameters` property.

| Layer        | Description |
|--------------|-------------|
| `Linear`     | Fully-connected layer `y = xW + b` with Xavier weight initialization and optional bias. |
| `Dropout`    | Inverted dropout with keep-probability scaling; a no-op when `training=False`. |
| `Sequential` | Chains layers and forwards through them in order; aggregates their parameters. |

`Sequential.save_graph(path, arch=True, forward=False, backward=False)` renders a
`.png` of the model via `core/graph.py` (needs `graphviz`): a cluster per layer
for the architecture, and — if requested — the forward/backward computational
graphs built from a synthetic input, tensors colour-coded by role
(input/weights/bias/hidden).

## Activation functions (`core/activations.py`)

Each activation is available both as a pure NumPy function (`core/functions.py`)
and as an autograd-aware `Layer`:

| Activation | Notes |
|------------|-------|
| `ReLU`     | `max(0, x)` |
| `Sigmoid`  | Numerically stable (branch on the sign of the input) |
| `Tanh`     | `np.tanh` |
| `GELU`     | Sigmoid approximation `x · σ(1.702·x)` |
| `Softmax`  | Max-shifted for stability; configurable `dim` |

`functions.py` also provides a stable `log_softmax`, used internally by the
cross-entropy loss.

## Loss functions (`core/losses.py`)

| Loss                     | Input | Notes |
|--------------------------|-------|-------|
| `MSELoss`                | predictions, targets | Mean squared error. |
| `CrossEntropyLoss`       | logits, integer targets | Combines a stable `log_softmax` with negative log-likelihood; the backward is the classic `softmax(logits) − onehot(targets)`. |
| `BinaryCrossEntropyLoss` | probabilities, targets | Clips predictions to `[1e-7, 1 − 1e-7]` to avoid `log(0)`. |

Each loss is callable (`loss(pred, target)`) and returns a scalar `Tensor` you can
call `.backward()` on.

## Optimizers (`core/optimizer.py`)

| Optimizer | Notes |
|-----------|-------|
| `SGD`     | Plain gradient descent with optional L2 weight decay. |
| `SGDM`    | SGD with momentum. |
| `Adam`    | Adaptive moments with bias correction. |
| `AdamW`   | Adam with decoupled weight decay. |

Every optimizer takes `model.parameters` and a learning rate; `step()` updates
`param.data` in place, `zero_grad()` clears `param.grad`, and `get_state()`
returns the optimizer's hyperparameters/buffers for checkpointing.

## Training loop (`core/training/`)

- **`Trainer`** (`trainer.py`) wraps a model, loss, optimizer and optional
  scheduler. `train_epoch(dataloader, accumulation_steps=1)` runs one epoch
  (with gradient accumulation and optional `clip_grad_norm` clipping) and
  `eval(dataloader)` runs a no-grad pass, both logging into `trainer.history`
  (`train_loss`, `eval_loss`, `lr`). `save()`/`load()` (de)serialize training
  state to a checkpoint file via `pickle`.
- **`Schedule`** (`schedulers.py`) is the abstract base for learning-rate
  schedules; `CosineSchedule(max_lr, min_lr, total_epochs)` anneals the
  learning rate from `max_lr` to `min_lr` following a cosine curve, applied by
  `Trainer` at the end of every `train_epoch()` call.

## Data loading (`core/dataset/`)

The module mirrors the PyTorch `Dataset` / `DataLoader` pattern.

- **`Dataset`** — abstract base defining `__len__` and `__getitem__`.
- **`TensorDataset`** — wraps in-memory tensors and validates that they share the
  same length along dimension 0.
- **`ImageDataset`** — lazily loads images from disk on access (via `load_jpeg`),
  pairing each with its label.
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
| [`variance/`](examples/linear_regression/variance/README.md) | Notebook fitting nine `Linear(1, 1)` models across increasing label noise, showing how the learned slope/intercept drift as noise grows. |

The [`linear_regression/README.md`](examples/linear_regression/README.md)
ties the linear/quadratic/cubic scripts together and explains why the learning
rate has to shrink as the polynomial degree grows.

```bash
uv run python examples/linear_regression/linear/linear.py
```

## Roadmap

Planned work is tracked in [`TODO.md`](TODO.md), and includes:

- **Loader**: parallel loading via multithreading; prefetching of the next batch.
- **Autograd**: move backward computation entirely onto NumPy arrays (keeping
  `Tensor` as a pure frontend); cache forward intermediates for reuse in backward;
  add a debug step that reports which node a backward failure occurred on.
