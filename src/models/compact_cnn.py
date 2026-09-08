"""Configurable provisional compact CNN for temporal I/Q input."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ProvisionalCompactRFNet(nn.Module):
    """A provisional Conv1D reconstruction baseline, not recovered original code."""

    def __init__(
        self,
        num_classes: int,
        input_length: int,
        conv_channels: Sequence[int] = (64, 128, 256),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        pooling: Sequence[int] = (2, 2, 2),
        dropout: float = 0.3,
        classifier_size: int = 384,
    ) -> None:
        super().__init__()
        if num_classes < 2 or input_length < 1:
            raise ValueError("num_classes must be >= 2 and input_length must be positive.")
        if not (len(conv_channels) == len(kernel_sizes) == len(pooling)) or not conv_channels:
            raise ValueError("conv_channels, kernel_sizes, and pooling must have equal nonzero lengths.")
        if any(value <= 0 for value in (*conv_channels, *kernel_sizes, *pooling)):
            raise ValueError("Convolution and pooling settings must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = []
        in_channels = 2
        for out_channels, kernel_size, pool_size in zip(conv_channels, kernel_sizes, pooling):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(pool_size),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, classifier_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_size, num_classes),
        )
        self.input_length = input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected input [N, 2, T], got {tuple(x.shape)}.")
        if x.shape[2] != self.input_length:
            raise ValueError(f"Expected input length {self.input_length}, got {x.shape[2]}.")
        return self.classifier(self.temporal_pool(self.features(x)))


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that participate in optimization."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# Backward-compatible import name. The class itself remains explicitly provisional.
CompactRFNet = ProvisionalCompactRFNet
