import torch
import torch.nn as nn
from torchvision.models import(
    resnet18,
    resnet50,
    resnet101,
    resnet152,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
)


def get_nested_attr(obj, path):
    for p in path:
        obj = obj[p] if isinstance(p, int) else getattr(obj, p)
    return obj


def set_nested_attr(obj, path, value):
    parent = get_nested_attr(obj, path[:-1])
    last = path[-1]
    if isinstance(last, int):
        parent[last] = value
    else:
        setattr(parent, last, value)


class TorchvisionBackbone(nn.Module):
    CONFIGS = {
        "resnet18":  (resnet18,  ResNet18_Weights.DEFAULT,  ["conv1"], None),
        "resnet50":  (resnet50,  ResNet50_Weights.DEFAULT,  ["conv1"], None),
        "resnet101": (resnet101, ResNet101_Weights.DEFAULT, ["conv1"], None),
        "resnet152": (resnet152, ResNet152_Weights.DEFAULT, ["conv1"], None),

        "densenet121": (densenet121, DenseNet121_Weights.DEFAULT, ["features", "conv0"], ["features"]),
        "densenet161": (densenet161, DenseNet161_Weights.DEFAULT, ["features", "conv0"], ["features"]),
        "densenet169": (densenet169, DenseNet169_Weights.DEFAULT, ["features", "conv0"], ["features"]),
        "densenet201": (densenet201, DenseNet201_Weights.DEFAULT, ["features", "conv0"], ["features"]),

        "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b5": (efficientnet_b5, EfficientNet_B5_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b6": (efficientnet_b6, EfficientNet_B6_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b7": (efficientnet_b7, EfficientNet_B7_Weights.DEFAULT, ["features", 0, 0], ["features"]),
    }

    def __init__(self, name: str, pretrained: bool = True, grayscale: bool = True):
        super().__init__()

        if name not in self.CONFIGS:
            raise ValueError(f"Unsupported backbone: {name}. Choose from {list(self.CONFIGS)}")

        builder, default_weights, first_conv_path, features_path = self.CONFIGS[name]

        weights = default_weights if pretrained else None
        model = builder(weights=weights)

        if grayscale:
            self._convert_first_conv_to_grayscale(model, first_conv_path)

        if name.startswith("resnet"):
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
        else:
            self.features = get_nested_attr(model, features_path)

    def _convert_first_conv_to_grayscale(self, model, conv_path):
        old_conv = get_nested_attr(model, conv_path)

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        set_nested_attr(model, conv_path, new_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def set_trainable_layers(self, trainable_layers: list[str] | None = None):
        """
        trainable_layers: list of substrings of parameter names to unfreeze.
        Everything else is frozen.
        """

        if not trainable_layers:
            # freeze everything
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
            return

        for name, param in self.named_parameters():
            trainable = any(layer in name for layer in trainable_layers)
            param.requires_grad = trainable

        # set eval/train modes correctly
        for name, module in self.named_modules():
            trainable = any(layer in name for layer in trainable_layers)
            if not trainable:
                module.eval()
            else:
                module.train()