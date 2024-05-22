import torch
import torchvision.models as models
import torch.nn as nn
import os


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200), nn.ReLU(), nn.Linear(200, 10), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def train(self, train_loader, epochs, device="cuda"):
        self.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        for epoch in range(epochs):
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = self(imgs.view(imgs.shape[0], -1))
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(epoch, loss.item())

    def save(self, file="./saved_models/mlp.pth"):
        if not os.path.exists("/".join(file.split("/")[:-1])):
            os.makedirs("/".join(file.split("/")[:-1]))
        torch.save(self.state_dict(), file)

    def load(self, file="./saved_models/mlp.pth", device="cuda"):
        self.load_state_dict(torch.load(file, map_location=device))


if __name__ == "__main__":
    import torchvision
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[-1] * 1, std=[1] * 1)]
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_data = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    model = MLP().to(device)
    model.train(train_loader, 20, device=device)
    model.save("./saved_models/bias_mlp.pth")
    model.load("./saved_models/bias_mlp.pth")
    model = model.cuda()
    accuracy = 0
    for imgs, labels in test_loader:
        imgs = imgs.cuda()
        labels = labels.cuda()
        outputs = model(imgs.view(imgs.shape[0], -1))
        accuracy += (torch.argmax(outputs, dim=1) == labels).sum().item()
    print(accuracy / len(test_data))
