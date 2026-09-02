from typing import override
import numpy as np
from thorcino.consts import BIAS_ROLE, WEIGHTS_ROLE
from thorcino.functions import relu, sigmoid, tanh, gelu
from thorcino.layers.layer import Layer
from thorcino.tensor import Tensor

class RNN(Layer):
    def __init__(
        self, 
        in_feature: int,
        hidden_units: int,
        activation_h: Layer,
        bias_hidden: bool = True,
    ):
        self.training = True
        self.in_feature = in_feature
        self.hidden_units = hidden_units
        self.activation_h = activation_h

        scale = np.sqrt(1.0 / in_feature)
        weights_data = np.random.randn(in_feature, hidden_units) * scale
        self.weights_input = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.randn(hidden_units, hidden_units) * scale
        self.weighs_hidden = Tensor(weights_data, role=WEIGHTS_ROLE)

        if bias_hidden:
            self.bias_hidden = Tensor(np.zeros(hidden_units), role=BIAS_ROLE)
        else:
            self.bias_hidden = None

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature}, hidden_units={self.hidden_units}, bias_hidden={self.bias_hidden.shape}))"

    @override
    def forward(self, X: Tensor) -> Tensor:
        """
        Compute the layer output: 

        `H_t = phi_h(Xt @ W_xh + H_{t-1} @ W_hh + b_h)`
        
        Params
        ------
        X: inputs, possible shapes: 
            (batch_size, sequence_length, in_feature), 
            (batch_size, sequence_length), 
            (sequence_length)

        Returns
        -------
        outs: output for each step stacked along first dim
        """
        if X.dim == 1:
            X = X.reshape(1, -1)

        batch_size, seq_len = X.shape[0], X.shape[1]
        Ht = Tensor(np.zeros((batch_size, self.hidden_units)))
        
        outs = []
        for t in range(seq_len):
            Xt = Tensor(X.data[:, t].reshape(-1, self.in_feature))
            
            Hprev = Ht
            Ht = Xt @ self.weights_input + Hprev @ self.weighs_hidden
            if self.bias_hidden is not None:
                Ht += self.bias_hidden
            Ht = self.activation_h(Ht)

            outs.append(Ht)

        return Tensor.stack(outs, axis=1)
    
    @property
    @override
    def parameters(self) -> list[Tensor]:
        params = [self.weights_input, self.weighs_hidden]

        if self.bias_hidden is not None:
            params.append(self.bias_hidden)
        
        return params
    
    @property
    @override
    def state(self) -> dict:
        data = { 'W_xh': self.weights_input.data,  'W_hh': self.weighs_hidden.data }
        if self.bias_hidden is not None:
            data['b_h'] = self.bias_hidden.data

        return data
    
    @override
    def set_state(self, state: dict) -> None:
        self.weights_input.data = state['W_xh']
        self.weighs_hidden.data = state['W_hh']
        if self.bias_hidden is not None:
            self.bias_hidden.data = state['b_h']

    @override
    def train(self) -> None:
        self.training = True

        self.weights_input.requires_grad = True
        self.weighs_hidden.requires_grad = True

        if self.bias_hidden is not None:
            self.bias_hidden.requires_grad = True
        

    @override
    def eval(self) -> None:
        self.training = False

        self.weights_input.requires_grad = False
        self.weighs_hidden.requires_grad = False

        if self.bias_hidden is not None:
            self.bias_hidden.requires_grad = False
