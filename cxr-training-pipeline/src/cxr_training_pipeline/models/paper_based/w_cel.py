import torch.nn as nn
import torch
import torch.nn.functional as F


class BatchBalancedBCEWithLogitsLoss(nn.Module):
    """
    Loss function based on: "https://arxiv.org/pdf/1705.02315"
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        pos_mask = targets == 1
        neg_mask = targets == 0

        p = pos_mask.sum().clamp(min=1).float()
        n = neg_mask.sum().clamp(min=1).float()

        beta_p = (p + n) / p
        beta_n = (p + n) / n

        pos_loss = -(F.logsigmoid(logits)[pos_mask]).sum()
        neg_loss = -(F.logsigmoid(-logits)[neg_mask]).sum()

        loss = beta_p * pos_loss + beta_n * neg_loss
        loss = loss / (p + n)

        return loss
