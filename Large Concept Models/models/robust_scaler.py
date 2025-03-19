import torch

class RobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, data):
        self.median = torch.median(data)
        q1 = torch.quantile(data, 0.25)
        q3 = torch.quantile(data, 0.75)
        self.iqr = q3 - q1
        if self.iqr == 0:
            self.iqr = torch.tensor(1e-9)

    def transform(self, data):
        if self.median is None or self.iqr is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (data - self.median) / self.iqr

    def inverse_transform(self, data):
        if self.median is None or self.iqr is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (data * self.iqr) + self.median