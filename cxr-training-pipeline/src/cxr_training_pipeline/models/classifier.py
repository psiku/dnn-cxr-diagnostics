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
        use_mask_channel: bool = False,
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
        self.use_mask_channel = use_mask_channel
        self.image_channels = 1 if grayscale else 3

        self.backbone = TorchvisionBackbone(
            name=backbone_name,
            pretrained=pretrained,
            grayscale=grayscale,
            use_mask_channel=use_mask_channel,
        )

        self.backbone.set_trainable_layers(backbone_trainable_layers)

        if in_features is None:
            in_features = self._infer_backbone_channels()

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

    @property
    def input_channels(self) -> int:
        return self.image_channels + (1 if self.use_mask_channel else 0)

    def _infer_backbone_channels(self) -> int:
        device = next(self.backbone.parameters()).device

        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, 224, 224, device=device)
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

    def forward(
        self,
        image: torch.Tensor,
        retain_transition_grad: bool = False,
    ):
        if image.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels "
                f"(grayscale={self.image_channels == 1}, use_mask_channel={self.use_mask_channel}), "
                f"got {image.shape[1]}."
            )

        conv_maps = self.backbone(image)
        transition_maps = self.transition(conv_maps)

        if retain_transition_grad:
            if not torch.is_grad_enabled():
                raise RuntimeError(
                    "Grad-CAM requires gradient computation. "
                    "Do not call the method within torch.no_grad() or "
                    "torch.inference_mode()."
                )

            if not transition_maps.requires_grad:
                transition_maps = transition_maps.detach().requires_grad_(True)

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

    def grad_cam(self, image: torch.Tensor, class_idx: int):
        self.eval()

        if not 0 <= class_idx < self.num_classes:
            raise ValueError(
                f"class_idx musi należeć do zakresu "
                f"[0, {self.num_classes - 1}], otrzymano {class_idx}"
            )

        with torch.enable_grad():
            out = self.forward(
                image,
                retain_transition_grad=True,
            )

            transition_maps = out["transition_maps"]
            logits = out["logits"]

            class_score = logits[:, class_idx].sum()

            gradients = torch.autograd.grad(
                outputs=class_score,
                inputs=transition_maps,
                retain_graph=False,
                create_graph=False,
            )[0]

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            grad_cam = (weights * transition_maps).sum(dim=1)

            grad_cam = F.relu(grad_cam)
            grad_cam = self._normalize_map(grad_cam)

        detached_out = {
            key: value.detach()
            for key, value in out.items()
        }

        return grad_cam.detach(), detached_out

