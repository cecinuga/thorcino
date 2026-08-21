## Recurrent Neural Network (RNN)

The aim of this notebook is to show a full example of how to implement from scratch a RNN, first using only NumPy and then using thorcino (this framework), building the necessary components to make it a reusable and composable module of this framework.

See [scratch.ipynb](./scratch.ipynb) for the full, runnable implementation.


## Introduction

A RNN is a type of neural network that embed the concept of time **implicity** in the model. This is done by using a hidden state that carry the information from the previous time step to the next one, recursively. This allows the model to have a memory of the past inputs and use it to make predictions on the current input.

**implicity** means that *time* is **not** and explicit input dimension/axes, it's not encoded into the input data, but it's implicitly present in how model learn.

The RNN can learn through a variation of classic Back Propagation algorithm called Back Propagation Through Time (BPTT). This algorithm is used to train RNNs by unrolling the network in time and applying the standard backpropagation algorithm to the unrolled network.


## The Learning Task

The notebook trains the RNN on a simple, but general enough, task: predicting the next number in a sequence of equidistant ordered numbers. Random sequences are generated with NumPy, and for each sequence the input $X$ is the sequence itself while the target $Y$ is the same sequence shifted by one step.


## Notation

$t$ is the time step

$n$ is the number of samples (or the batch size)

$d$ is the number of input for each sample (or the number of features)

$h$ is the number of hidden units


$\mathbf{H}_t \in \mathbb{R}^{n \times h}$ is the hidden state at time step $t$.

$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ is the input at time step $t$.

$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ is the weight matrix (shared across time step!)

$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ is the hidden-state-to-hidden-state weight matrix (shared across time step!)

### The output of an RNN is:

$$
\mathbf{O}_t = \phi_o \left( \mathbf{H}_t \mathbf{W}_{ho} + \mathbf{b}_o \right)
$$

Where $\mathbf{H}_t$ equals:

$$
\mathbf{H}_t = \phi_h \left( \mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h \right)
$$


## Gradient computation (Back Propagation Through Time)

The loss over a sequence is the sum of the per-step losses:

$$
\mathcal{L} \left( \mathbf{O}, \mathbf{Y} \right)
= \sum_{t=1}^{T} \ell_t \left( \mathbf{O}_t, \mathbf{Y}_t \right)
$$

Since $\mathbf{W}_{xh}$ and $\mathbf{W}_{hh}$ are shared across all time steps, their gradients accumulate contributions from every step, unrolled back through time:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \mathbf{O}_t}
   \cdot \frac{\partial \mathbf{O}_t}{\partial \phi_o}
   \cdot \mathbf{W}_{ho}
   \sum_{k=1}^{t} \left( \mathbf{W}_{hh}^{\top} \right)^{t-k}
   \cdot \mathbf{H}_k \\[1em]
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}}
&= \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \mathbf{O}_t}
   \cdot \frac{\partial \mathbf{O}_t}{\partial \phi_o}
   \cdot \mathbf{W}_{ho}
   \sum_{k=1}^{t} \left( \mathbf{W}_{hh}^{\top} \right)^{t-k}
   \cdot \mathbf{X}_k
\end{aligned}
$$

In the notebook, this is implemented by iterating backwards over the cached time steps, accumulating the gradients of `Wxh`, `Whh`, `bh`, `Who` and `bo`, and propagating the hidden-state gradient (`dH_next`) to the previous time step.


## Optimizer: Stochastic Gradient Descent (SGD)

The network is trained with plain SGD, with optional weight decay ($\lambda$) on the weight matrices (not on the biases), and a cosine learning-rate schedule that anneals the learning rate from a max to a min value over the training epochs:

$$
\begin{equation}
\hat{g}_t^{(i)} =
\begin{cases}
\nabla_\theta \mathcal{L}(\theta_t^{(i)}) + \lambda \, \theta_t^{(i)} & \text{if } \theta^{(i)} \in \mathcal{W} \\[4pt]
\nabla_\theta \mathcal{L}(\theta_t^{(i)}) & \text{otherwise}
\end{cases}
\end{equation}

\begin{equation}
\theta_{t+1}^{(i)} = \theta_t^{(i)} - \eta \, \hat{g}_t^{(i)}
\end{equation}
$$


## Training and Inference

Each training step runs a full forward pass over the sequence, computes the loss with `thorcino.functions.mse`, backpropagates through time, and applies an SGD step. Inference reuses the same forward recurrence (without computing the loss) to roll the hidden state forward and produce a prediction at each time step.

![Training loss over 500 epochs](./loss.png)

The plot above shows the training loss (summed MSE over the sequence) across the 500 training epochs used in the notebook. The loss drops sharply in the first epochs, thanks to the higher learning rate at the start of the cosine schedule, then keeps decreasing more gradually as the learning rate anneals towards `MIN_LR`, indicating the RNN is converging on the next-number-in-sequence task.

---

Resources:

[Recurrent Neural Networks (RNNs): A gentle Introduction and Overview - Robin M. Schmid](https://arxiv.org/abs/1912.05911)

[Finding Structure in Time - Jeffrey L. Elman](https://onlinelibrary.wiley.com/doi/epdf/10.1207/s15516709cog1402_1)