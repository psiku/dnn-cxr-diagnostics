"""Visualization and metrics utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    save_path: str | None = None,
) -> None:
    """Plot loss and accuracy curves from a training run.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        val_accs: Per-epoch validation accuracies.
        save_path: If provided, save the figure to this path instead of
            showing it interactively.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.legend()

    ax2.plot(epochs, val_accs, label="Val accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_roc_curve(
    y_true: list[int] | np.ndarray,
    y_scores: list[float] | np.ndarray,
    save_path: str | None = None,
) -> float:
    """Plot the ROC curve and return the AUC score.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Predicted probabilities for the positive class.
        save_path: Optional path to save the figure.

    Returns:
        Area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)

    return auc


def plot_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot the confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Display names for each class.
        save_path: Optional path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
