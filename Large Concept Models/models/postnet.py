import torch.nn as nn

# post-net denormalizes the layer data to the output 
class PostNet(nn.Module):
    def __init__(self, model_dim, output_dim, scaler):
        super().__init__()
        self.linear = nn.Linear(model_dim, output_dim)
        self.scaler = scaler

    def forward(self, x):
        x_denormalized = self.linear(x)
        return self.scaler.inverse_transform(x_denormalized)