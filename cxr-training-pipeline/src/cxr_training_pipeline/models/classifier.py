import torch
import torch.nn as nn
import torch.nn.functional as F
from cxr_training_pipeline.models.lse_poling import LSEPool2d
from cxr_training_pipeline.models.backbone import TorchvisionBackbone


class ChestXRayClassifier(nn.Module):
    """
    Chest X-Ray multi-label classifier with configurable backbone and pooling.
    """

    def __init__(
        self,
        num_classes: int = 14,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        grayscale: bool = True,
        backbone_trainable_layers: list[str] | None = None,
        in_features: int | None = None,
        transition_dim: int = 2048,
        use_transition: bool = True,
        pooling: str = "lse",
        lse_r: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if backbone_trainable_layers is None:
            backbone_trainable_layers = []

        self.num_classes = num_classes
        self.pooling = pooling
        self.use_transition = use_transition

        self.backbone = TorchvisionBackbone(
            name=backbone_name,
            pretrained=pretrained,
            grayscale=grayscale,
        )

        self.backbone.set_trainable_layers(backbone_trainable_layers)

        if in_features is None:
            in_features = self._infer_backbone_channels(grayscale)

        if use_transition:
            self.transition = nn.Sequential(
                nn.Conv2d(in_features, transition_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(transition_dim),
                nn.ReLU(inplace=True),
            )
            classifier_dim = transition_dim
        else:
            self.transition = nn.Identity()
            classifier_dim = in_features

        if pooling == "lse":
            self.global_pool = LSEPool2d(r=lse_r)
        elif pooling == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == "max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.prediction = nn.Linear(classifier_dim, num_classes)

    def _infer_backbone_channels(self, grayscale: bool) -> int:
        device = next(self.backbone.parameters()).device
        channels = 1 if grayscale else 3

        with torch.no_grad():
            dummy = torch.zeros(1, channels, 224, 224, device=device)
            out = self.backbone(dummy)

        return out.shape[1]

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_pool(x)

        if x.ndim == 4:
            x = torch.flatten(x, 1)

        return x

    @staticmethod
    def _normalize_map(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + eps)
        return cam

    def forward(self, image: torch.Tensor, retain_transition_grad: bool = False):
        conv_maps = self.backbone(image)
        transition_maps = self.transition(conv_maps)

        if retain_transition_grad:
            transition_maps.retain_grad()

        pooled_features = self._pool_features(transition_maps)
        pooled_features = self.dropout(pooled_features)

        logits = self.prediction(pooled_features)

        return {
            "logits": logits,
            "transition_maps": transition_maps,
            "pooled_features": pooled_features,
        }

    @torch.no_grad()
    def cam(self, image: torch.Tensor, class_idx: int):
        self.eval()

        out = self.forward(image, retain_transition_grad=False)

        transition_maps = out["transition_maps"]
        class_weights = self.prediction.weight[class_idx]

        cam = torch.einsum("d,bdhw->bhw", class_weights, transition_maps)
        cam = F.relu(cam)
        cam = self._normalize_map(cam)

        return cam, out