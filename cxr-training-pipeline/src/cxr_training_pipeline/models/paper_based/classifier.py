import torch
import torch.nn as nn
import torch.nn.functional as F
from cxr_training_pipeline.models.lse_poling import LSEPool2d
from cxr_training_pipeline.models.resnet_backbone import ResNet50Backbone


class CXRClassifier(nn.Module):
    """
    Classifier implementation based on: "https://arxiv.org/pdf/1705.02315"
    """

    def __init__(
        self,
        num_classes: int = 14,
        transition_dim: int = 2048,
        pooling: str = "lse",  # "lse", "avg", "max"
        lse_r: float = 10.0,
        backbone: ResNet50Backbone = ResNet50Backbone(),
        backbone_trainable_layers: list[str] = [],
    ):
        super().__init__()

        self.num_classes = num_classes
        self.transition_dim = transition_dim
        self.pooling = pooling

        self.backbone = backbone
        self.backbone.set_trainable_layers(backbone_trainable_layers)

        # transition layer
        self.transition = nn.Sequential(
            nn.Conv2d(2048, transition_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(transition_dim),
            nn.ReLU(inplace=True),
        )

        # global pooling
        if pooling == "lse":
            self.global_pool = LSEPool2d(r=lse_r)
        elif pooling == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == "max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        # prediction layer
        self.prediction = nn.Linear(transition_dim, num_classes)

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "lse":
            return self.global_pool(x)
        return torch.flatten(self.global_pool(x), 1)

    @staticmethod
    def _normalize_map(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + eps)
        return cam

    def forward(self, image: torch.Tensor, retain_transition_grad: bool = False):
        # backbone activation maps
        conv_maps = self.backbone(image)  # [B, 2048, h, w]

        # transition maps
        transition_maps = self.transition(conv_maps)  # [B, D, h, w]

        if retain_transition_grad:
            transition_maps.retain_grad()
        pooled_features = self._pool_features(transition_maps)  # [B, D]

        logits = self.prediction(pooled_features)  # [B, C]

        return {
            "logits": logits,
            "transition_maps": transition_maps,
            "pooled_features": pooled_features,
        }

    @torch.no_grad()
    def cam(self, image: torch.Tensor, class_idx: int):
        """
        CAM mentiond in the paper.
        """
        self.eval()
        out = self.forward(image, retain_transition_grad=False)

        transition_maps = out["transition_maps"]  # [B, D, h, w]
        class_weights = self.prediction.weight[class_idx]  # [D]

        cam = torch.einsum("d,bdhw->bhw", class_weights, transition_maps)
        cam = F.relu(cam)
        cam = self._normalize_map(cam)

        return cam, out
