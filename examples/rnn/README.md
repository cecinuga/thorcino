## Recurrent Neural Network (RNN)

The aim of this notebook is to show a full example of how to implement from scratch a RNN, first using only NumPy and then using thorcino (this frameworks), building the necessary components to make it a component of this framework.


## Introduction

A RNN is a type of neural network that embed the concept of time **implicity** in the model. This is done by using a hidden state that carry the information from the previous time step to the next one, recursively. This allows the model to have a memory of the past inputs and use it to make predictions on the current input.

**implicity** means that *time* is **not** and explicit input dimension/axes, it's not encoded into the input data, but it's implicitly present in how model learn.

The RNN can learn through a variation of classic Back Propagation algorithm called Back Propagation Through Time (BPTT). This algorithm is used to train RNNs by unrolling the network in time and applying the standard backpropagation algorithm to the unrolled network.


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
\mathbf{H}_t = \phi_h \left( \mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} + \mathbf{b}_h \right)
$$

---

Resources:

[Recurrent Neural Networks (RNNs): A gentle Introduction and Overview - Robin M. Schmid](https://arxiv.org/abs/1912.05911)

[Finding Structure in Time - Jeffrey L. Elman](https://onlinelibrary.wiley.com/doi/epdf/10.1207/s15516709cog1402_1)