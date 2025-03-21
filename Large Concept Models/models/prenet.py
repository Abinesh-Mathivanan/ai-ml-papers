import torch.nn as nn

# pre-net normalizes the sonar embeddings to the model's hidden dimensions 
class PreNet(nn.Module):
    def __init__(self, input_dim, model_dim, scaler):
        super().__init__()
        self.linear = nn.Linear(input_dim, model_dim)
        self.scaler = scaler

    def forward(self, x):
        x_normalized = self.scaler.transform(x)
        return self.linear(x_normalized)