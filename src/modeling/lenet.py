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


def embed(model: LeNet5, x: torch.Tensor, layer: str = "feature_maps") -> torch.Tensor:
    """Flattened activations used as input to a quantum encoder."""
    fm = model.features(x)
    if layer == "feature_maps":
        return fm.flatten(1)
    if layer == "penultimate":
        return model.classifier[:-1](fm)
    raise ValueError(layer)


class LeNetBottleneck(nn.Module):
    """LeNet5 features -> k-unit bottleneck -> small head (same k as the QNN qubits)."""

    def __init__(
        self, in_channels: int = 1, input_size: int = 64, k: int = 8, hidden: int = 4
    ):
        super().__init__()
        self.features = LeNet5(in_channels, input_size).features
        with torch.no_grad():
            n_flat = self.features(
                torch.zeros(1, in_channels, input_size, input_size)
            ).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, k),
            nn.Tanh(),
            nn.Linear(k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)
