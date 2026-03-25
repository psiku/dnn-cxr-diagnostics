import torch
import torch.nn as nn


class LSEPool2d(nn.Module):
    """
    Log-Sum-Exp pooling.
    Returns tensor [B, C].
    """

    def __init__(self, r: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.r = r
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x_max = x.amax(dim=(2, 3), keepdim=True)
        pooled = x_max + (1.0 / self.r) * torch.log(
            torch.mean(torch.exp(self.r * (x - x_max)), dim=(2, 3), keepdim=True) + self.eps
        )
        return pooled.flatten(1)  # [B, C]
