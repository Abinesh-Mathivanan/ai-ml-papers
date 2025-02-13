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

class SimpleNeuralNet:
    def __init__(self, input_dim, hidden_dim, output_dim, p=0.5):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        self.dropout = Dropout(p)
    
    def forward(self, X, training=True):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.A1 = self.dropout.forward(self.A1, training)
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2

    def backward(self, X, dZ2, learning_rate=0.01):
        dA1 = dZ2 @ self.W2.T
        dA1[self.Z1 <= 0] = 0
        dA1 = self.dropout.backward(dA1)

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dW1 = X.T @ dA1
        db1 = np.sum(dA1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
