import torch
import torchvision.models as models
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, model="inception_v3"):
        super(Inception, self).__init__()
        if model == "inception_v3":
            self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
            self.weights = models.Inception_V3_Weights.DEFAULT
        else:
            raise ValueError("Model not supported")
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.categories = self.weights.meta["categories"]

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

    def transform(self, x):
        return self.weights.transforms(antialias=True)(x)


if __name__ == "__main__":
    model = Inception()
    print(model(torch.randn(1, 3, 299, 299)))
