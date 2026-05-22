import torch
import torchvision.models as models
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, model="vgg16"):
        super(VGG, self).__init__()
        if model == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.weights = models.VGG16_Weights.DEFAULT
        elif model == "vgg19":
            self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            self.weights = models.VGG19_Weights.DEFAULT
        elif model == "vgg11":
            self.model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
            self.weights = models.VGG11_Weights.DEFAULT
        elif model == "vgg13":
            self.model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
            self.weights = models.VGG13_Weights.DEFAULT
        else:
            raise ValueError("Model not supported")
        self.categories = self.weights.meta["categories"]

    def forward(self, x):
        return self.model(x)

    def transform(self, x):
        return self.weights.transforms(antialias=True)(x)


if __name__ == "__main__":
    model = VGG()
    print(model(torch.randn(1, 3, 224, 224)))
