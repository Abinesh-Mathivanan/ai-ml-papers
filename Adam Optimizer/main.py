import torch
import sys
sys.path.append('optim')

from model import SimpleLinearModel
from optim.adam import AdamOptimizer

def generate_data(num_samples=100):
    X = torch.linspace(0, 5, num_samples)
    Y = 2 * X + 1 + 0.5 * torch.randn(num_samples)
    return X, Y

model = SimpleLinearModel()
optimizer = AdamOptimizer(model.parameters(), lr=0.001)

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

X, Y = generate_data()

num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X)
    loss = mse_loss(predictions, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

print("\nLearned Parameters:")
print(f"W = {model.W.item():.4f} | True W = 2")
print(f"b = {model.b.item():.4f} | True b = 1")
