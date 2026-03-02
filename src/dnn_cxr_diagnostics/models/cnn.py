"""CNN model architectures for chest X-ray classification.

Models are built on top of torchvision pre-trained backbones so they can be
fine-tuned with a relatively small labelled dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class CXRClassifier(nn.Module):
    """ResNet-50 based binary / multi-label classifier for chest X-rays.

    Args:
        num_classes: Number of output classes / labels.
        pretrained: Whether to initialise the backbone with ImageNet weights.
        dropout: Dropout probability applied before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Replace the final fully-connected layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
