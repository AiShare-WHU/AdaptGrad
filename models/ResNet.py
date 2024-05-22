import torch
import torchvision.models as models
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, model="resnet50"):
        super(ResNet, self).__init__()
        if model == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.weights = models.ResNet50_Weights.DEFAULT
        elif model == "resnet101":
            self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.weights = models.ResNet101_Weights.DEFAULT
        elif model == "resnet152":
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            self.weights = models.ResNet152_Weights.DEFAULT
        else:
            raise ValueError("Model not supported")
        self.softmax = nn.Softmax(dim=1)
        self.categories = self.weights.meta["categories"]

    def forward(self, x):
        x = self.model(x)
        # x = self.softmax(x)
        return x

    def transform(self, x):
        return self.weights.transforms(antialias=True)(x)


if __name__ == "__main__":
    model = ResNet()
    prediction = model(model.transform(torch.randn(1, 3, 224, 224)))[0]
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = model.categories[class_id]
    print(f"Predicted class: {category_name} with score: {score}")
