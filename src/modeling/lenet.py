import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, in_channels: int = 1, input_size: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            n_flat = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)
