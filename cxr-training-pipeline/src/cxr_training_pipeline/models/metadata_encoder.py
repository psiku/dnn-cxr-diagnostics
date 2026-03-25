import torch.nn as nn
import torch


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)
