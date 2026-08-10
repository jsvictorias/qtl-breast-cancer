from __future__ import annotations

import torch
from torch import nn


class LeNet(nn.Module):
    """LeNet-5 adaptada para quantidade dinâmica de canais e classes."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        conv1_channels: int = 6,
        conv2_channels: int = 16,
        fc1_units: int = 120,
        fc2_units: int = 84,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels not in {1, 3}:
            raise ValueError("in_channels deve ser 1 ou 3.")
        if num_classes < 2:
            raise ValueError("num_classes deve ser pelo menos 2.")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channels, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_channels * 5 * 5, fc1_units),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(fc1_units, fc2_units),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(fc2_units, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
