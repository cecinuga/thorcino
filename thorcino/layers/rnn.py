import numpy as np
from thorcino.consts import BIAS_ROLE, WEIGHTS_ROLE
from thorcino.layers.sequential import Layer
from thorcino.tensor import Tensor

class RNN(Layer):
    def __init__(
        self, 
        in_feature: int,
        out_feature:int,
        hidden_units:int,
        bias_hidden:bool = True,
        bias_out:bool = True,
        hidden_activation = np.tanh,
        out_activation = np.tanh
    ):
        self.training = True
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_units = hidden_units
        self.h_activation = hidden_activation
        self.out_activation = out_activation

        scale = np.sqrt(1.0 / in_feature)
        weights_data = np.random.randn(in_feature, hidden_units) * scale
        self.weights_input = Tensor(weights_data, role=WEIGHTS_ROLE)

        scale = np.sqrt(1.0 / hidden_units)
        weights_data = np.random.randn(hidden_units, out_feature) * scale
        self.weights_output = Tensor(weights_data, role=WEIGHTS_ROLE)

        weights_data = np.random.rand(hidden_units, hidden_units) * scale
        self.weighs_hidden = Tensor(weights_data, role=WEIGHTS_ROLE)
        
        if bias_out:
            self.bias_out = Tensor(np.zeros(out_feature), role=BIAS_ROLE)
        else:
            self.bias_out = None

        if bias_hidden:
            self.bias_hidden = Tensor(np.zeros(out_feature), role=BIAS_ROLE)
        else:
            self.bias_hidden = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature},out_feature={self.out_feature},hidden_units={self.hidden_units},bias_hidden={self.bias_hidden},bias_out={self.bias_out})"

    def forward(self, X: Tensor) -> Tensor:
        """
        Compute the layer output: 

        `H_t = phi_h(X_t @ W_xh + H_{t-1} @ W_hh + b_h)`
        
        `O_t = phi_o(H_t @ W_ho + b_o)`

        Params
        ------
        X: inputs, shape=(batch_size, sequence_length, in_feature), last ax optional

        Returns
        -------
        cache: if training mode is on for each `t` it contains `Hprev`, `Ht`, `out`, otherwise only `out`
        """
        batch_size, seq_len = X.shape[0], X.shape[1]
        Ht = np.zeros((batch_size, self.hidden_units))
        
        cache = []
        for t in range(seq_len):
            X_t = X[:, t].reshape(-1, self.in_feature)
            
            Hprev = Ht
            Ht = self.h_activation( X_t @ self.weights_input + Hprev @ self.weighs_hidden + self.bias_hidden ) 
            out = self.out_activation( Ht @ self.weights_output + self.bias_out )

            if self.training:
                history = {'Hprev':Hprev, 'Ht': Ht, 'out': out}
            else:
                history = {'out': out}

            self.cache.append(history)

        return cache
