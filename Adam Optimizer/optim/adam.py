import torch

class AdamOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

        for p in self.params:
            self.m[p] = torch.zeros_like(p.data)
            self.v[p] = torch.zeros_like(p.data)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.data.zero_()

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data
            m = self.m[p]
            v = self.v[p]

            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p.data.addcdiv_(m_hat, (v_hat.sqrt() + self.eps), value=-self.lr)
