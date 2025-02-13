import numpy as np

class Dropout:
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) < self.p) / self.p  
            return x * self.mask
        return x  

    def backward(self, grad_output):
        return grad_output * self.mask
