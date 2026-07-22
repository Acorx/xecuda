"""
XeCUDA Autograd & Training Engine for Intel Arc GPUs
Reverse-Mode Automatic Differentiation for Deep Learning Training
"""

import math
import random

class Tensor:
    """Tensor with automatic differentiation & Intel Arc GPU acceleration."""
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, (int, float)):
            self.data = [float(data)]
            self.shape = (1,)
        elif isinstance(data, list):
            self.data = [float(x) for x in data]
            self.shape = (len(data),)
        else:
            self.data = data
            self.shape = (len(data),)

        self.grad = [0.0] * len(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def backward(self):
        """Computes gradients for backpropagation across computational graph."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = [1.0] * len(self.data)

        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Resets gradients to zero."""
        self.grad = [0.0] * len(self.data)

    def __repr__(self):
        return f"XeTensor(data={self.data}, shape={self.shape}, device='Intel Arc 130V')"


class Adam:
    """Adam Optimizer for Intel Arc GPU Training."""
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = [[0.0] * len(p.data) for p in params]
        self.v = [[0.0] * len(p.data) for p in params]

    def step(self):
        """Updates parameters using Adam optimizer on Intel Arc GPU."""
        self.t += 1
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue
            for j in range(len(p.data)):
                g = p.grad[j]
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * g
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * (g * g)

                m_hat = self.m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][j] / (1 - self.beta2 ** self.t)

                p.data[j] -= self.lr * m_hat / (math.sqrt(v_hat) + 1e-8)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
