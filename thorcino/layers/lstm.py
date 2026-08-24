from typing import override
import numpy as np
from thorcino.activations import Sigmoid, Tanh
from thorcino.consts import BIAS_ROLE, WEIGHTS_ROLE
from thorcino.layers.layer import Layer
from thorcino.tensor import Tensor

class LSTM(Layer):
    def __init__(
        self, 
        in_feature: int, 
        out_feature: int, 
        hidden_units: int, 
    ):
        self.training = True
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_units = hidden_units

        scale = np.sqrt(1.0 / in_feature)

        weights_data = np.random.randn(in_feature, hidden_units) * scale
        self.weights_input = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.randn(in_feature, hidden_units) * scale
        self.weights_forget = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.randn(in_feature, hidden_units) * scale
        self.weights_output = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.randn(in_feature, hidden_units) * scale
        self.weights_cell = Tensor(weights_data, role=WEIGHTS_ROLE)

        scale = np.sqrt(1.0 / hidden_units)

        weights_data = np.random.randn(hidden_units, hidden_units) * scale
        self.weights_h_input = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.randn(hidden_units, hidden_units) * scale
        self.weights_h_forget = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.randn(hidden_units, hidden_units) * scale
        self.weights_h_output = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.randn(hidden_units, hidden_units) * scale
        self.weights_h_cell = Tensor(weights_data, role=WEIGHTS_ROLE)

        self.bias_input = Tensor(np.zeros(1, hidden_units), role=BIAS_ROLE)
        self.bias_forget = Tensor(np.zeros(1, hidden_units), role=BIAS_ROLE)
        self.bias_output = Tensor(np.zeros(1, hidden_units), role=BIAS_ROLE)
        self.bias_cell = Tensor(np.zeros(1, hidden_units), role=BIAS_ROLE)

    
    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature}, out_feature={self.out_feature}, hidden_units={self.hidden_units})"
    
    def __compute_gate_input(self, Xt: Tensor, Ht: Tensor) -> Tensor:
        L = Xt @ self.weights_input + Ht @ self.weights_h_input + self.bias_input
        return self.sigmoid(L)

    def __compute_gate_forget(self, Xt: Tensor, Ht: Tensor) -> Tensor:
        L = Xt @ self.weights_forget + Ht @ self.weights_h_forget + self.bias_forget
        return self.sigmoid(L)
    
    def __compute_gate_output(self, Xt: Tensor, Ht: Tensor) -> Tensor:
        L = Xt @ self.weights_output + Ht @ self.weights_h_output + self.bias_output
        return self.sigmoid(L)
    
    def __compute_cell(self, Xt: Tensor, Ht: Tensor) -> Tensor:
        L = Xt @ self.weights_cell + Ht @ self.weights_h_cell + self.bias_cell
        return self.tanh(L)
    
    @override
    def forward(self, X: Tensor) -> Tensor:
        """
        Compute the layer output

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
        Mt = Tensor(np.zeros((batch_size, self.hidden_units)))

        outs = []
        for t in range(seq_len):
            Hprev, Mprev = Ht, Mt

            Xt = Tensor(X.data[:, t].reshape(-1, self.in_feature))

            It = self.__compute_gate_input(Xt, Hprev)
            Ft = self.__compute_gate_forget(Xt, Hprev)
            Ot = self.__compute_gate_output(Xt, Hprev)
            Ct = self.__compute_cell(Xt, Hprev)

            Mt = Ft * Mprev + It * Ct
            Ht = Ot * self.tanh(Mt)           

            outs.append(Ot)

        return Tensor.stack(outs, axis=1)
    

    @property
    @override
    def parameters(self) -> list[Tensor]:
        params = [
            self.weights_input, self.weights_forget, self.weights_output, self.weights_cell,
            self.weights_h_input, self.weights_h_forget, self.weights_h_output, self.weights_h_cell,
            self.bias_input, self.bias_forget, self.bias_output, self.bias_cell
        ]

        return params

    @override
    def train(self) -> None:
        self.training = True
        
        self.weights_input.requires_grad = True
        self.weights_forget.requires_grad = True
        self.weights_output.requires_grad = True
        self.weights_cell.requires_grad = True

        self.weights_h_input.requires_grad = True
        self.weights_h_forget.requires_grad = True
        self.weights_h_output.requires_grad = True
        self.weights_h_cell.requires_grad = True

        self.bias_input.requires_grad = True
        self.bias_forget.requires_grad = True
        self.bias_output.requires_grad = True
        self.bias_cell.requires_grad = True


    @override
    def eval(self) -> None:
        self.training = False
        
        self.weights_input.requires_grad = False
        self.weights_forget.requires_grad = False
        self.weights_output.requires_grad = False
        self.weights_cell.requires_grad = False

        self.weights_h_input.requires_grad = False
        self.weights_h_forget.requires_grad = False
        self.weights_h_output.requires_grad = False
        self.weights_h_cell.requires_grad = False

        self.bias_input.requires_grad = False
        self.bias_forget.requires_grad = False
        self.bias_output.requires_grad = False
        self.bias_cell.requires_grad = False
        


