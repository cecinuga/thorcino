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
        out_feature:int,
        hidden_units:int,
        h_activation:Layer,
        o_activation:Layer,
        bias_hidden:bool = True,
        bias_out:bool = True,
    ):
        self.training = True
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_units = hidden_units
        self.h_activation = h_activation
        self.o_activation = o_activation

        scale = np.sqrt(1.0 / in_feature)
        weights_data = np.random.randn(in_feature, hidden_units) * scale
        self.weights_input = Tensor(weights_data, role=WEIGHTS_ROLE)

        scale = np.sqrt(1.0 / hidden_units)
        weights_data = np.random.randn(hidden_units, out_feature) * scale
        self.weights_output = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.randn(hidden_units, hidden_units) * scale
        self.weighs_hidden = Tensor(weights_data, role=WEIGHTS_ROLE)
        
        if bias_out:
            self.bias_out = Tensor(np.zeros(out_feature), role=BIAS_ROLE)
        else:
            self.bias_out = None

        if bias_hidden:
            self.bias_hidden = Tensor(np.zeros(hidden_units), role=BIAS_ROLE)
        else:
            self.bias_hidden = None

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature},out_feature={self.out_feature},hidden_units={self.hidden_units},bias_hidden={self.bias_hidden},bias_out={self.bias_out})"

    @override
    def forward(self, X: Tensor) -> Tensor:
        """
        Compute the layer output: 

        `H_t = phi_h(X_t @ W_xh + H_{t-1} @ W_hh + b_h)`
        
        `O_t = phi_o(H_t @ W_ho + b_o)`

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
            X_t = Tensor(X.data[:, t].reshape(-1, self.in_feature))
            
            Hprev = Ht
            Ht = X_t @ self.weights_input + Hprev @ self.weighs_hidden
            if self.bias_hidden is not None:
                Ht += self.bias_hidden
            Ht = self.h_activation(Ht)

            out = Ht @ self.weights_output
            if self.bias_out is not None:
                out += self.bias_out
            out = self.o_activation(out)

            outs.append(out)

        return Tensor.stack(outs, axis=1)
    
    @override
    def train(self) -> None:
        self.training = True
        self.weights_input.requires_grad = True
        self.weighs_hidden.requires_grad = True
        self.weights_output.requires_grad = True

        if self.bias_hidden is not None:
            self.bias_hidden.requires_grad = True
        
        if self.bias_out is not None:
            self.bias_out.requires_grad = True

    @override
    def eval(self) -> None:
        self.training = False
        self.weights_input.requires_grad = False
        self.weighs_hidden.requires_grad = False
        self.weights_output.requires_grad = False

        if self.bias_hidden is not None:
            self.bias_hidden.requires_grad = False
        
        if self.bias_out is not None:
            self.bias_out.requires_grad = False


    @property
    @override
    def parameters(self):
        params = [self.weights_input, self.weighs_hidden, self.weights_output]

        if self.bias_hidden is not None:
            params.append(self.bias_hidden)
        
        if self.bias_out is not None:
            params.append(self.bias_out)

        return params