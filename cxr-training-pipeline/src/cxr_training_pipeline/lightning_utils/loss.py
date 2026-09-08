from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from cxr_training_pipeline.models.paper_based.w_cel import (
    BatchBalancedBCEWithLogitsLoss,
)

SUPPORTED_LOSSES = ("batch_balanced_bce", "bce_with_logits")


def compute_class_pos_weights(
    train_df: pd.DataFrame,
    label_cols: list[str],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-class pos_weight for BCEWithLogitsLoss (neg_count / pos_count)."""
    pos_counts = train_df[label_cols].sum(axis=0)
    neg_counts = len(train_df) - pos_counts
    weights = neg_counts / (pos_counts + eps)
    return torch.tensor(weights.to_numpy(dtype="float32"))


def resolve_pos_weights(
    loss_cfg: dict[str, Any] | None,
    train_df: pd.DataFrame | None = None,
    label_cols: list[str] | None = None,
) -> torch.Tensor | None:
    """Resolve pos_weight tensor from config and optional training data."""
    loss_cfg = loss_cfg or {}
    if loss_cfg.get("name", "batch_balanced_bce") != "bce_with_logits":
        return None

    pos_weight_cfg = loss_cfg.get("pos_weight") or {}
    manual_values = pos_weight_cfg.get("values")
    if manual_values is not None:
        return torch.tensor(manual_values, dtype=torch.float32)

    if not pos_weight_cfg.get("compute_from_train", False):
        return None

    if train_df is None or not label_cols:
        raise ValueError(
            "pos_weight.compute_from_train requires train_df and label_cols."
        )

    eps = float(pos_weight_cfg.get("eps", 1e-8))
    return compute_class_pos_weights(train_df, label_cols, eps=eps)


def build_loss_criterion(
    loss_cfg: dict[str, Any] | None,
    pos_weight: torch.Tensor | None = None,
) -> nn.Module:
    """Build a loss module from pipeline configuration."""
    loss_cfg = loss_cfg or {}
    name = loss_cfg.get("name", "batch_balanced_bce")

    if name == "batch_balanced_bce":
        eps = float(loss_cfg.get("eps", 1e-8))
        return BatchBalancedBCEWithLogitsLoss(eps=eps)

    if name == "bce_with_logits":
        if pos_weight is None:
            return nn.BCEWithLogitsLoss()
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    supported = ", ".join(SUPPORTED_LOSSES)
    raise ValueError(f"Unknown loss '{name}'. Supported values: {supported}.")
