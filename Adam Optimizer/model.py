import torch

class SimpleLinearModel:
    def __init__(self):
        self.W = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    def __call__(self, x):
        return x * self.W + self.b

    def parameters(self):
        return [self.W, self.b]
