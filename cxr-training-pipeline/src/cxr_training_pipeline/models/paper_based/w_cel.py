import torch.nn as nn
import torch
import torch.nn.functional as F


class BatchBalancedBCEWithLogitsLoss(nn.Module):
    """
    Loss function based on: "https://arxiv.org/pdf/1705.02315"
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()

        p = pos_mask.sum(dim=0).clamp(min=1.0)
        n = neg_mask.sum(dim=0).clamp(min=1.0)

        beta_p = (p + n) / p
        beta_n = (p + n) / n

        log_pos = F.logsigmoid(logits)
        log_neg = F.logsigmoid(-logits)

        pos_loss = -beta_p * pos_mask * log_pos
        neg_loss = -beta_n * neg_mask * log_neg

        loss = pos_loss + neg_loss

        return loss.mean()
