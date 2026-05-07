import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights)


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, grayscale: bool = True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)

        if grayscale:
            old_conv = model.conv1
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            model.conv1 = new_conv

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 2048, H/32, W/32]

    def set_trainable_layers(self, trainable_layers: list[str] = []):
        for name, module in self.named_children():
            trainable = name in trainable_layers

            for p in module.parameters():
                p.requires_grad = trainable

            if not trainable:
                module.eval()
