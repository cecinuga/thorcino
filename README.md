# thorcino

A minimal, educational deep-learning framework built from scratch on top of NumPy.
`thorcino` reimplements the essential pieces of a PyTorch-style workflow — a
tensor with reverse-mode automatic differentiation, a small set of layers
(including recurrent ones), activation and loss functions, checkpointing and a
data-loading pipeline — in a few hundred lines of readable Python.

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
# loss=0.2412  w=2.000  b=4.996  <- recovers 2 and 5; Xavier init and the
# shuffling are unseeded, so the exact digits move a little between runs
```

The run above can be resumed later: `trainer.save(path)` pickles the model,
optimizer and scheduler state alongside the metrics, and `trainer.load(path)`
puts all of it back.

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

Wrap a forward pass in `no_grad()` to skip building the graph entirely — the
same switch `Trainer.eval()` uses:

```python
from thorcino.grad_mode import no_grad   # also: from thorcino.autograd import no_grad

with no_grad():
    print((a @ b)._grad_fn)   # None -> nothing was recorded
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
├── grad_mode.py         # Global autograd switch: no_grad(), is_grad_enabled()
├── functions.py         # Pure numpy math: activations, softmax, loss functions
├── activations.py       # Activation layers (Identity, ReLU, Sigmoid, Tanh, GELU, Softmax)
├── losses.py            # Loss objects (MSE, CrossEntropy, BinaryCrossEntropy)
├── optimizer.py         # Optimizer, SGD, SGD_DL2, SGDM, Adam, AdamW
├── graph.py             # ComputationalGraph: graphviz visualisation of a Sequential model
├── consts.py            # Parameter role tags (WEIGHTS_ROLE, BIAS_ROLE)
├── utils.py             # unbroadcast() helper for gradient reduction
├── layers/               # Network layers
│   ├── layer.py          #   Layer: abstract base (forward, train/eval, parameters, state)
│   ├── linear.py         #   Linear
│   ├── dropout.py        #   Dropout
│   ├── rnn.py            #   RNN (unrolled recurrence, BPTT via autograd)
│   ├── lstm.py           #   LSTM (input/forget/output gates + cell state)
│   └── sequential.py     #   Sequential + graph rendering helpers
├── autograd/            # Reverse-mode automatic differentiation
│   ├── base.py          #   Function: base class for every backward node
│   ├── arithmetic.py    #   Add/Sub/Mul/Div/Matmul/Sum/Mean/Max/Pow/Index/… backward
│   ├── activations.py   #   ReLU/Sigmoid/Tanh/GELU/Softmax backward
│   └── losses.py        #   MSE/CrossEntropy/BCE backward
├── dataset/              # Data loading pipeline
│   ├── dataset.py        #   Dataset, TensorDataset, ImageDataset, DataLoader
│   ├── transformation.py #   RandomHorizontalFlip, RandomCrop, Compose
│   └── utils.py          #   image loading helpers
├── artifact/             # Pickled run artifacts
│   └── artifact.py       #   Artifact, load_artifact
└── training/             # Training loop orchestration
    ├── trainer.py         #   Trainer: train_epoch/eval, checkpointing, clip_grad_norm
    └── schedulers.py      #   Schedule, CosineSchedule, CosineRestartSchedule
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

`thorcino/__init__.py` flattens most of this into a single import surface — the
tensor, every layer and activation, the losses, the optimizers, the trainer and
the data-loading pieces — so day-to-day use never needs the module paths. A few
newer pieces are not re-exported yet and need theirs: `no_grad`
(`thorcino.grad_mode`), `CosineRestartSchedule`
(`thorcino.training.schedulers`), `LSTM` (`thorcino.layers.lstm`, though
`from thorcino import LSTM` does work) and the artifact helpers
(`thorcino.artifact`).

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
if is_grad_enabled():
    out._grad_fn = AddBackward(self, other)
```

So each output tensor remembers *how it was produced*. Chaining operations
produces a graph of `Function` nodes rooted at the final output.

### Turning recording off (`thorcino/grad_mode.py`)

Every operation checks `is_grad_enabled()` before attaching its `_grad_fn`, and
the `no_grad` context manager flips that global off:

```python
from thorcino.grad_mode import no_grad

with no_grad():
    preds = model(inputs)   # no backward nodes are allocated at all
```

`grad_mode` is deliberately dependency-free so `thorcino.tensor` can import it
without pulling in `thorcino.autograd` (which imports `thorcino.tensor` back).
`Trainer.eval()` runs under `no_grad()`: for a recurrent model an evaluation pass
would otherwise build — and immediately discard — one node per timestep per
batch. `enable_autograd()` survives as a no-op for API compatibility; autograd is
on unless a `no_grad` block is active.

### The backward pass

`Tensor.backward()` (`thorcino/tensor.py`) drives backpropagation **iteratively**:

1. If no gradient is supplied, it seeds `1.0` for a scalar output (and raises for
   non-scalar outputs, matching PyTorch's behaviour).
2. It topologically sorts the graph (reverse-postorder DFS), so every node is
   visited only once all of its consumers have contributed their gradient — no
   Python recursion, so a long unrolled sequence can't blow the stack.
3. It walks that order, calling `_grad_fn.apply(gradient)` on each node and
   accumulating the results both into a per-node buffer (used to keep flowing
   downstream) and into each input tensor's `.grad`.

Partial gradients travel as raw NumPy arrays rather than `Tensor`s: wrapping and
adding them would build a second graph while unwinding the first. `apply()` must
return exactly one gradient per saved tensor — a mismatch raises a `RuntimeError`
naming the offending backward class.

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
| Arithmetic  | `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward`, `NegBackward`, `ScaleBackward`, `PowBackward`, `IdentityBackward` |
| Linear alg. | `MatmulBackward`, `TransposeBackward` |
| Reductions  | `SumBackward`, `MeanBackward`, `MaxBackward` |
| Shape       | `ReshapeBackward`, `StackBackward`, `IndexBackward` |
| Activations | `ReLUBackward`, `SigmoidBackward`, `TanhBackward`, `GELUBackward`, `SoftmaxBackward` |
| Losses      | `MSELossBackward`, `CrossEntropyLossBackward`, `BCELossBackward` |

`ScaleBackward` handles the scalar/`ndarray` fast paths of `*` and `/`;
`MaxBackward` serves both `max()` and `min()` (via a `largest` flag) and splits
the gradient evenly across ties.

## The `Tensor` class (`thorcino/tensor.py`)

`Tensor` is a lightweight wrapper around a `np.ndarray` (always stored as
`float32`). It exposes:

- **Metadata**: `data`, `shape`, `size`, `dim`, `dtype`, `requires_grad`, `grad`,
  `_grad_fn`, `role`. The optional `role` tags a tensor as `weights` or `bias`
  (see `thorcino/consts.py`); it drives per-role weight decay in the optimizers
  and the colour coding in the graph renderer.
- **Operator overloading**: `__add__`/`__radd__`, `__sub__`/`__rsub__`,
  `__mul__`/`__rmul__`, `__truediv__`, `__matmul__`, `__pow__`, `__neg__`,
  `__gt__`. Every arithmetic path is autograd-aware, including the scalar and
  `ndarray` ones — a constant addend attaches `IdentityBackward`, a scalar factor
  `ScaleBackward`, and `other - self` `NegBackward`.
- **Tensor ops**: `matmul`, `reshape` (supports `-1` inference), `transpose`,
  `sum`, `mean`, `max`, `min`, plus `__len__` and an autograd-aware
  `__getitem__` that routes the gradient back to the positions it selected.
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
`forward()`, `train()` and `eval()`, makes instances callable, and exposes
`parameters`, `state` and `set_state()`. `train()`/`eval()` flip each layer's
`training` flag and the `requires_grad` of its parameters. The `state` accessors
default to empty, so a layer without weights needs no override.

| Layer        | Description |
|--------------|-------------|
| `Linear`     | Fully-connected layer `y = xW + b` with Xavier weight initialization and optional bias. |
| `Dropout`    | Inverted dropout with keep-probability scaling; a no-op when `training=False`. |
| `RNN`        | Vanilla recurrent layer: `H_t = φ_h(X_t·W_xh + H_{t−1}·W_hh + b_h)`. Unrolls over the sequence axis and returns every hidden state stacked along `axis=1`; BPTT falls out of the autograd graph. Takes an explicit `activation_h` layer. The hidden state restarts at zero on each call, so sequences are never carried over between batches. |
| `LSTM`       | Long short-term memory cell with input/forget/output gates and a cell state (forget-gate bias initialised to ones), unrolled the same way. `out_type='n_to_m'` returns every hidden state, `'n_to_1'` only the last. Inputs are normalised to `(batch, time, feature)` first, so a 1-D or 2-D input is read as one scalar per timestep and requires `in_feature == 1`. |
| `Sequential` | Chains layers and forwards through them in order; aggregates their parameters and state, and propagates `train()`/`eval()`. |

Both recurrent layers are re-exported from the top-level package
(`from thorcino import RNN, LSTM`). Note that `thorcino.layers` itself only
re-exports `RNN`, so importing `LSTM` from that sub-package needs its module
path (`from thorcino.layers.lstm import LSTM`).

### Checkpointable state

Every parameterised layer implements `state` (a dict of copied NumPy arrays under
short keys — `'W'`/`'b'` for `Linear`, `'W_xh'`/`'W_hh'`/`'b_h'` for `RNN`, the
twelve `'W_<gate>'`/`'W_h<gate>'`/`'b_<gate>'` entries for `LSTM`) and
`set_state()`, which copies them back into the existing parameters and asserts
that the shapes already match. `Sequential` keys its children by position
(`"l0"`, `"l1"`, …), so a reload only accepts a model with the same layers in
the same order. This is what `Trainer.save()`/`load()` is built on.

### Rendering the graph

`Sequential` renders `.png` pictures of the model via `thorcino/graph.py` (needs
`graphviz`), through three methods that each take a real input `shape` including
the batch dimension:

- `save_arch(path, shape)` — one node per layer, annotated with the real
  input/output shapes it produces.
- `save_forward(path, shape)` — the forward data-flow graph.
- `save_backward(path, shape)` — the same autograd graph read in reverse.

Each is built from an actual forward/backward pass with the shape you pass in
(e.g. `(batch, in_feature)`, or `(batch, seq_len, in_feature)` for a recurrent
model), so every node carries the shape it would really have. Tensors are
colour-coded by role (input/weights/bias/hidden). `destroy_graph()` drops the
cached picture.

## Activation functions (`thorcino/activations.py`)

Each activation is available both as a pure NumPy function (`thorcino/functions.py`)
and as an autograd-aware `Layer`:

| Activation | Notes |
|------------|-------|
| `Identity` | `x` unchanged; useful as an explicit pass-through for the `RNN` layer's activation slot. |
| `ReLU`     | `max(0, x)` |
| `Sigmoid`  | Numerically stable (branch on the sign of the input) |
| `Tanh`     | `np.tanh` |
| `GELU`     | Sigmoid approximation `x · σ(1.702·x)` |
| `Softmax`  | Max-shifted for stability; configurable `dim` |

Each wires its backward node only while `training`, so an `eval()`-mode
activation returns a tensor detached from the graph. `functions.py` also provides
a stable `log_softmax`, used internally by the cross-entropy loss.

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
| `SGDM`    | SGD with classic (non-Nesterov) momentum; unlike `SGD`, its `weight_decay` applies to every parameter regardless of role. |
| `Adam`    | Adaptive moments with bias correction. |
| `AdamW`   | Adam with decoupled weight decay — the decay shrinks `param.data` directly instead of entering the gradient, so it never feeds the moment estimates. |

Every optimizer takes `model.parameters` and a learning rate; `step()` updates
`param.data` in place and bumps `step_count`, and `zero_grad()` clears
`param.grad`. For checkpointing, the `state` property returns the
hyperparameters, the step counter and any per-parameter buffers (momentum, or
Adam's `m`/`v`), and `set_state()` restores them onto the same parameter list in
the same order.

## Training loop (`thorcino/training/`)

- **`Trainer`** (`trainer.py`) wraps a model, loss, optimizer and optional
  scheduler.
  - `train_epoch(dataloader, accumulation_steps=1)` runs one epoch and returns
    the average per-batch loss. It reads the schedule **before** the epoch's
    updates run (the schedule is indexed by the epoch it applies to), supports
    gradient accumulation — the accumulation scale rides on the seed gradient, so
    the loss tensor and its graph are left untouched — flushes a short trailing
    group, and optionally clips gradients via the `grad_clip_norm` constructor
    argument.
  - `eval(dataloader, record=True)` runs under `no_grad()` and returns
    `(avg_loss, accuracy)`; accuracy covers both multi-class and binary outputs
    and is `0.0` for non-classification (1-D) outputs. It restores whichever mode
    the caller was in rather than assuming training. Pass `record=False` for
    evaluations that aren't part of the training curve (a sweep over several
    validation sets, say) so they don't interleave with the periodic ones.
  - Both log into `trainer.history` (`train_loss`, `eval_loss`, `accuracy`,
    `lr`), also reachable via the `train_loss`/`eval_loss` properties.
  - `save(path)` pickles a **full** checkpoint — counters, history, mode, plus
    model, optimizer and scheduler state — and `load(path)` restores all of it in
    place. `save_metrics(path)` writes epoch, step and history only, for the
    common case of keeping the curves without the weights.
- **`clip_grad_norm(parameters, max_norm=1.0)`** rescales every parameter
  gradient in place so their combined L2 norm stays within `max_norm`, and
  returns the pre-clip norm. `Trainer` calls it for you, but it is exported
  from `thorcino.training` for standalone use.
- **`Schedule`** (`schedulers.py`) is the abstract base for learning-rate
  schedules, with the same `state`/`set_state()` contract as the optimizers:
  - `CosineSchedule(max_lr, min_lr, total_epochs)` anneals from `max_lr` to
    `min_lr` along a cosine curve, then holds at `min_lr`.
  - `CosineRestartSchedule(max_lr, min_lr, boundaries)` restarts a full
    `max_lr → min_lr` sweep at every entry of `boundaries`, so the epoch just
    before each boundary is trained at (essentially) `min_lr`. That is what makes
    an intermediate checkpoint comparable to a completed run of that budget — the
    property the scaling-law study relies on. It is not re-exported at the top
    level: `from thorcino.training.schedulers import CosineRestartSchedule`.

## Artifacts (`thorcino/artifact/`)

A thin wrapper over the pickles the training loop writes: `load_artifact(path)`
reads one file and returns a frozen `Artifact` dataclass holding its `data` dict
of NumPy arrays. The scaling-law example builds its whole analysis pass on top of
this — see `examples/recurrent/scaling_law/helpers.py`.

## Data loading (`thorcino/dataset/`)

The module mirrors the PyTorch `Dataset` / `DataLoader` pattern.

- **`Dataset`** — abstract base defining `__len__` and `__getitem__`.
- **`TensorDataset`** — wraps in-memory tensors and validates that they share the
  same length along dimension 0. Raw `np.ndarray`s are accepted too and wrapped
  on the way in.
- **`ImageDataset`** — lazily loads images from disk on access (via `load_jpeg`),
  pairing each with its label. Note that `load_jpeg` (`dataset/utils.py`) is
  still a stub, so this dataset is scaffolding rather than a working loader.
- **`DataLoader`** — iterates a `Dataset` in mini-batches, with optional
  shuffling, and collates each batch by stacking samples along a new leading
  (batch) axis. A bare list of per-field tensors is accepted as shorthand for a
  `TensorDataset`. The final batch is short when the dataset size isn't a
  multiple of `batch_size`.

Data augmentation transforms live in `transformation.py`:

- `RandomHorizontalFlip(p)` — flips along the width axis with probability `p`.
- `RandomCrop(height, width, padding)` — zero-pads then crops a random window.
- `Compose([...])` — chains transforms into a single callable.

## Examples

The examples split in two families, sharing the dataset/training/plotting
helpers in [`examples/helpers/`](examples/helpers/).

### Regression — [`examples/regression/`](examples/regression/README.md)

| Example | Description |
|---|---|
| [`linear/`](examples/regression/linear/README.md) | Recovers the slope/intercept of `2·x + 5` with a single `Linear(1, 1)` layer; compares the learned fit against the closed-form `lstsq` solution. |
| [`quadratic/`](examples/regression/quadratic/README.md) | Recovers the coefficients of `x² + 2·x + 2` via feature expansion (`[x, x²]`) fed into a `Linear(2, 1)` layer. |
| [`cubic/`](examples/regression/cubic/cubic.py) | Same idea one degree further: recovers `1.2·x³ − 2.3·x² + 2·x + 2` with a `Linear(3, 1)` layer over `[x, x², x³]`. |
| [`ill-cond/`](examples/regression/ill-cond/README.md) | Notebook comparing closed-form OLS vs. SGD on ill-conditioned (near-collinear) features, showing OLS's coefficients blow up while SGD's stay stable. |
| [`variance/`](examples/regression/variance/README.md) | Notebook fitting 20 repeated `Linear(1, 1)` models at each of nine label-noise levels, showing via boxplots how the loss and learned slope/intercept drift and spread as noise grows. |
| [`regularization/`](examples/regression/regularization/README.md) | Two notebooks on weight decay as variance reduction: [`L2/`](examples/regression/regularization/L2/README.md) sweeps the decay strength (with separate weights/bias decay via `SGD_DL2`) and watches the parameter spread shrink; [`DL2/`](examples/regression/regularization/DL2/README.md) isolates decay from noise across repeated trials (work in progress — only the no-decay baseline is implemented). |

The [`regression/README.md`](examples/regression/README.md) ties the
linear/quadratic/cubic scripts together and explains why the learning rate has to
shrink as the polynomial degree grows.

### Recurrent — [`examples/recurrent/`](examples/recurrent/)

| Example | Description |
|---|---|
| [`rnn/`](examples/recurrent/rnn/README.md) | Full walkthrough of a recurrent network on a next-number-in-sequence task, in two notebooks: `scratch.ipynb` hand-codes the forward recurrence, BPTT and the SGD update with raw NumPy, then `main.ipynb` rebuilds the same model on `thorcino.layers.RNN` + `Trainer` and renders its architecture and forward/backward graphs with `save_arch`/`save_forward`/`save_backward`. Its README derives the BPTT gradients in full. |
| [`lstm/`](examples/recurrent/lstm/README.md) | Binary classification of character sequences against a small BNF grammar (each term is a vowel followed by a consonant), which needs both memory and non-linear temporal dependencies. Includes the 3-dimensional tokenizer (is-vowel, is-consonant, alphabet position) and renders the model architecture. |
| [`scaling_law/`](examples/recurrent/scaling_law/) | A full-factorial scaling study over dataset size × sequence length × update budget, driven by the `Experiment` harness in `helpers.py`. |

The scaling-law example is the largest thing built on the framework so far, and
exercises most of the checkpointing machinery:

- Each grid cell is trained once to the **largest** update budget, harvesting a
  checkpoint at every smaller budget along the way — so the cost is that of the
  largest budget, not the sum of all of them. `CosineRestartSchedule` is what
  makes those intermediate checkpoints comparable to completed shorter runs.
- Epoch counts are derived from the updates an epoch *really* performs
  (`updates_per_epoch` accounts for the train split and the short final batch),
  so the budgets plotted are the budgets actually trained.
- Three artifacts are written per checkpoint — the full trainer state, the
  training metrics, and the scores against a shared validation set — under names
  encoding the hyperparameters
  (`E<id>__<epoch>_<updates>_<n_sequence>_<sequence_length>__<age>s.pkl`), so an
  analysis pass can group runs by factor without reopening every pickle.
- Recovery intersects the three artifact folders: a cell that only reached the
  backup before an interrupted run is treated as incomplete and recomputed,
  rather than counted as done with metrics missing.

Its README is still empty — the notebook and `helpers.py` docstrings are the
documentation for now.

```bash
uv run python examples/regression/linear/linear.py
```

## Roadmap

Planned work is tracked in [`TODO.md`](TODO.md), and includes:

- **Trainer**: a static constructor that builds a `Trainer` straight from a
  checkpoint file.
- **Loader**: parallel loading via multithreading; prefetching of the next batch.
- **Autograd**: cache forward intermediates for reuse in backward; add a debug
  step that reports which node a backward failure occurred on.

Known gaps in the current tree: `dataset/utils.py:load_jpeg` is a stub, the
scaling-law README is empty, and there is no test suite (the CI pytest step is
wired up but collects nothing).

## License

[MIT](LICENSE) © cecinuga
