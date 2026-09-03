from typing import override
import numpy as np
from thorcino.activations import Sigmoid, Tanh
from thorcino.consts import BIAS_ROLE, WEIGHTS_ROLE
from thorcino.layers.layer import Layer
from thorcino.tensor import Tensor

class LSTM(Layer):
    """LSTM cell unrolled over the time axis. `out_type` picks what `forward` returns:
    'n_to_m' every hidden state, 'n_to_1' only the last one. Hidden and cell state
    start at zero on each call, so sequences are never carried over."""

    valid_types = {"n_to_m", "n_to_1"}

    def __init__(
        self, 
        in_feature: int, 
        hidden_units: int,
        out_type: str = 'n_to_m',
    ):
        assert out_type in LSTM.valid_types

        self.out_type = out_type
        self.training = True
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        self.in_feature = in_feature
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

        self.bias_input = Tensor(np.zeros((1, hidden_units)), role=BIAS_ROLE)
        self.bias_forget = Tensor(np.zeros((1, hidden_units)), role=BIAS_ROLE)
        self.bias_output = Tensor(np.zeros((1, hidden_units)), role=BIAS_ROLE)
        self.bias_cell = Tensor(np.zeros((1, hidden_units)), role=BIAS_ROLE)

    
    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature}, hidden_units={self.hidden_units})"
    
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
        outs: with out_type='n_to_m', every hidden state stacked on the time axis
            (batch_size, sequence_length, hidden_units); with 'n_to_1', the last
            hidden state only (batch_size, hidden_units)
        """
        if X.dim == 1:
            X = X.reshape(1, -1)

        batch_size, seq_len = X.shape[0], X.shape[1]

        Ht = Tensor(np.zeros((batch_size, self.hidden_units)))
        Mt = Tensor(np.zeros((batch_size, self.hidden_units)))

        outs: list[Tensor] = []
        for t in range(seq_len):
            Hprev, Mprev = Ht, Mt

            Xt = Tensor(X.data[:, t])
            
            It = self.__compute_gate_input(Xt, Hprev)
            Ft = self.__compute_gate_forget(Xt, Hprev)
            Ot = self.__compute_gate_output(Xt, Hprev)
            Ct = self.__compute_cell(Xt, Hprev)

            Mt = Ft * Mprev + It * Ct
            Ht = Ot * self.tanh(Mt)           

            outs.append(Ht)

        if self.out_type == 'n_to_m':
            return Tensor.stack(outs, axis=1)
        else:
            return outs[-1]
    
    @property
    @override
    def parameters(self) -> list[Tensor]:
        params = [
            self.weights_input, self.weights_forget, self.weights_output, self.weights_cell,
            self.weights_h_input, self.weights_h_forget, self.weights_h_output, self.weights_h_cell,
            self.bias_input, self.bias_forget, self.bias_output, self.bias_cell
        ]

        return params
    
    @property
    @override
    def state(self) -> dict:
        """Copies of the twelve parameters, keyed 'W_<gate>' (input), 'W_h<gate>'
        (recurrent) and 'b_<gate>' for gates i/f/c/o."""
        data = {
            'W_i': self.weights_input.data.copy(), 'W_f': self.weights_forget.data.copy(), 'W_c': self.weights_cell.data.copy(), 'W_o': self.weights_output.data.copy(),
            'W_hi': self.weights_h_input.data.copy(), 'W_hf': self.weights_h_forget.data.copy(), 'W_hc': self.weights_h_cell.data.copy(), 'W_ho': self.weights_h_output.data.copy(),
            'b_i': self.bias_input.data.copy(), 'b_f': self.bias_forget.data.copy(), 'b_c': self.bias_cell.data.copy(), 'b_o': self.bias_output.data.copy()
        }

        return data
    
    @override
    def set_state(self, state: dict) -> None:
        """Copy the saved arrays into the existing parameters; shapes must already match."""
        assert self.weights_input.shape == state['W_i'].shape
        assert self.weights_forget.shape == state['W_f'].shape
        assert self.weights_cell.shape == state['W_c'].shape
        assert self.weights_output.shape == state['W_o'].shape
        assert self.weights_h_input.shape == state['W_hi'].shape
        assert self.weights_h_forget.shape == state['W_hf'].shape
        assert self.weights_h_cell.shape == state['W_hc'].shape
        assert self.weights_h_output.shape == state['W_ho'].shape
        assert self.bias_input.shape == state['b_i'].shape
        assert self.bias_forget.shape == state['b_f'].shape
        assert self.bias_cell.shape == state['b_c'].shape
        assert self.bias_output.shape == state['b_o'].shape

        self.weights_input.data = state['W_i'].copy()
        self.weights_forget.data = state['W_f'].copy()
        self.weights_cell.data = state['W_c'].copy()
        self.weights_output.data = state['W_o'].copy()
        self.weights_h_input.data = state['W_hi'].copy()
        self.weights_h_forget.data = state['W_hf'].copy()
        self.weights_h_cell.data = state['W_hc'].copy()
        self.weights_h_output.data = state['W_ho'].copy()
        self.bias_input.data = state['b_i'].copy()
        self.bias_forget.data = state['b_f'].copy()
        self.bias_cell.data = state['b_c'].copy()
        self.bias_output.data = state['b_o'].copy()

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
        


