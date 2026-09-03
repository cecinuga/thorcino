from typing import override, Any
import numpy as np
from thorcino.consts import BIAS_ROLE, WEIGHTS_ROLE
from thorcino.tensor import Tensor

class Optimizer:
    def __init__(self, params: list[Tensor], lr: float):
        self.params: list[Tensor] = params
        self.step_count: int = 0
        self.lr: float = lr

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        raise NotImplementedError()

    @property
    def state(self) -> dict:
        raise NotImplementedError()
    
    @property
    def set_state(self, state: dict) -> None:
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float=0.01, weight_decay:float = 0.0):
        super().__init__(params, lr)
        self.lr:float = lr
        self.weight_decay:float = weight_decay

    @override
    def step(self) -> None:
        self.step_count += 1
        for param in self.params:
            if param.grad is None: continue

            grad_data = param.grad
            if param.role == WEIGHTS_ROLE and self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            param.data -= self.lr * grad_data

    @override
    @property
    def state(self):
        return { 'lr': self.lr, 'weight_decay': self.weight_decay }
    
    @override
    def set_state(self, state: dict) -> None:
        self.lr = state['lr']
        self.weight_decay = state['weight_decay']
    
class SGD_DL2(Optimizer):
    def __init__(self, params: list[Tensor], lr: float=0.01, weight_decay:float = 0.0, bias_decay:float = 0.0):
        super().__init__(params, lr)
        self.lr:float = lr
        self.bias_decay:float = bias_decay
        self.weight_decay:float = weight_decay

    @override
    def step(self) -> None:
        self.step_count += 1
        for param in self.params:
            if param.grad is None: continue

            grad_data = param.grad

            if param.role == WEIGHTS_ROLE and self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data
                
            if param.role == BIAS_ROLE and self.bias_decay != 0:
                grad_data = grad_data + self.bias_decay * param.data

            param.data -= self.lr * grad_data

    @override
    @property
    def state(self):
        return { 'lr': self.lr, 'weight_decay': self.weight_decay }

    @override
    def set_state(self, state: dict) -> None:
        self.lr = state['lr']
        self.weight_decay = state['weight_decay']

class SGDM(Optimizer):
    def __init__(self, params: list[Tensor], lr:float = 0.01, momentum:float = 0.0, weight_decay:float = 0.0):
        super().__init__(params, lr)
        self.momentum_buffer:list[np.ndarray|None] = [None for _ in params]
        self.weight_decay:float = weight_decay
        self.momentum:float = momentum
        self.lr:float = lr
        self.step_count:int = 0

    @override
    def step(self) -> None:
        self.step_count += 1
        for i, param in enumerate(self.params):
            if param.grad is None: continue
            grad_data = param.grad

            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data
            if self.momentum != 0:
                if self.momentum_buffer[i] is None:
                    self.momentum_buffer[i] = np.zeros_like(param.data)

                # Update velocity: v = momentum * v_prev + grad
                self.momentum_buffer[i] = (self.momentum * self.momentum_buffer[i]) + grad_data    
                grad_data = self.momentum_buffer[i]

            param.data -= (self.lr * grad_data)    

    def has_momentum(self) -> bool:
        return self.momentum > 0

    @override
    @property
    def state(self):
        return {'lr': self.lr, 'weight_decay': self.weight_decay, 'momentum': self.momentum, 'momentum_buffer': self.momentum_buffer}

    @override
    def set_state(self, state: dict) -> None:
        self.lr = state['lr']
        self.weight_decay = state['weight_decay']
        self.momentum = state['momentum'].copy()
        self.momentum_buffer = state['momentum_buffer'].copy()


class Adam(Optimizer):
    def __init__(self, params: list[Tensor], lr: float=0.001, betas:tuple[float, float]=(0.9, 0.999), eps: float=1e-8):
        super().__init__(params, lr)
        self.lr:float = lr
        self.eps:float = eps
        self.beta1, self.beta2 = betas
        self.m_buffers:list[np.ndarray|None] = [None for _ in params]
        self.v_buffers:list[np.ndarray|None] = [None for _ in params]

    @override
    def step(self) -> None:
        self.step_count += 1
        for i, param in enumerate(self.params):
            if param.grad is None: continue
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.grad)
            if self.v_buffers[i] is None:
                self.v_buffers[i] = np.zeros_like(param.grad)

            # 1. Update biased moments
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * param.grad    
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (param.grad**2)    

            # Compute bias correction
            bias_correction1:float = 1 - self.beta1 ** self.step_count
            bias_correction2:float = 1 - self.beta2 ** self.step_count

            # 2. Compute bias-corrected moments
            m_hat = self.m_buffers[i] / bias_correction1  
            v_hat = self.v_buffers[i] / bias_correction2  

            # 3. Update parameter (Adaptive step)
            param.data -= (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)

    @override
    @property
    def state(self):
        return {
            'lr': self.lr,
            'eps': self.eps,
            'step_count': self.step_count,
            'betas': (self.beta1, self.beta2),
            'm_buffers': self.m_buffers,
            'v_buffers': self.v_buffers
        }
    
    @override
    def set_state(self, state: dict) -> None:
        self.lr = state['lr']
        self.eps = state['eps']
        self.step_count = state['step_count']
        self.beta1, self.beta2 = state['betas']
        self.m_buffers = state['m_buffers'].copy()
        self.v_buffers = state['v_buffers'].copy()


class AdamW(Optimizer):
    def __init__(self, params: list[Tensor], lr: float=0.001, betas:tuple[float, float]=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.0):
        super().__init__(params, lr)
        self.lr:float = lr
        self.eps:float = eps
        self.beta1, self.beta2 = betas
        self.weight_decay:float = weight_decay
        self.m_buffers:list[np.ndarray|None] = [None for _ in params]
        self.v_buffers:list[np.ndarray|None] = [None for _ in params]

    @override
    def step(self) -> None:
        self.step_count += 1
        for i, param in enumerate(self.params):
            if param.grad is None: continue
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.grad)
            if self.v_buffers[i] is None:
                self.v_buffers[i] = np.zeros_like(param.grad)

            grad_data = param.grad

            # Update moments using pure gradients (NO weight decay mixed in)
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data    
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)    

            # Compute bias correction and bias-corrected moments
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            m_hat = self.m_buffers[i] / bias_correction1   
            v_hat = self.v_buffers[i] / bias_correction2   

            # Apply decoupled weight decay (separate from gradient update)
            if self.weight_decay != 0:
                param.data *= (1 - self.lr * self.weight_decay)

            # Apply gradient-based update
            param.data -= (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)

    @override
    @property
    def state(self):
        return {
            'lr': self.lr,
            'eps': self.eps,
            'step_count': self.step_count,
            'betas': (self.beta1, self.beta2),
            'weight_decay': self.weight_decay,
            'm_buffers': self.m_buffers,
            'v_buffers': self.v_buffers
        }
    
    @override
    def set_state(self, state: dict) -> None:
        self.lr = state['lr']
        self.eps = state['eps']
        self.step_count = state['step_count']
        self.beta1, self.beta2 = state['betas']
        self.m_buffers = state['m_buffers'].copy()
        self.v_buffers = state['v_buffers'].copy()
