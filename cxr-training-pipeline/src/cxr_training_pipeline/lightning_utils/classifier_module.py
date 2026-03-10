from src.cxr_training_pipeline.models.paper_based.w_cel import BatchBalancedBCEWithLogitsLoss
from torch import nn
import torch
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

class ClassifierModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 14,
        lr: float = 1e-4,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.criterion = BatchBalancedBCEWithLogitsLoss()

        default_thresholds = torch.full((num_classes,), float(threshold), dtype=torch.float32)
        self.register_buffer("thresholds", default_thresholds)

        # Validation
        self.val_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.val_ap = MultilabelAveragePrecision(num_labels=num_classes, average="macro")

        self.val_precision = MultilabelPrecision(num_labels=num_classes, threshold=float(threshold), average="macro")
        self.val_recall = MultilabelRecall(num_labels=num_classes, threshold=float(threshold), average="macro")
        self.val_f1 = MultilabelF1Score(num_labels=num_classes, threshold=float(threshold), average="macro")

        # Test
        self.test_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.test_ap = MultilabelAveragePrecision(num_labels=num_classes, average="macro")

        self.test_precision = MultilabelPrecision(num_labels=num_classes, threshold=float(threshold), average="macro")
        self.test_recall = MultilabelRecall(num_labels=num_classes, threshold=float(threshold), average="macro")
        self.test_f1 = MultilabelF1Score(num_labels=num_classes, threshold=float(threshold), average="macro")

    def forward(self, image, retain_transition_grad: bool = False):
        return self.model(image, retain_transition_grad=retain_transition_grad)

    def set_thresholds(self, thresholds: torch.Tensor):
        thresholds = torch.as_tensor(thresholds, dtype=torch.float32, device=self.device)

        if thresholds.ndim != 1:
            raise ValueError(f"thresholds must be 1D, got shape={thresholds.shape}")

        if thresholds.numel() != self.hparams.num_classes:
            raise ValueError(
                f"thresholds must have {self.hparams.num_classes} elements, got {thresholds.numel()}"
            )

        self.thresholds.copy_(thresholds)

    def _get_preds(self, probs: torch.Tensor) -> torch.Tensor:
        thresholds = self.thresholds.view(1, -1).to(probs.device)
        return (probs > thresholds).int()

    def _shared_step(self, batch):
        image = batch["image"]
        target = batch["target"].float()

        out = self(image)
        logits = out["logits"]
        probs = torch.sigmoid(logits)
        loss = self.criterion(logits, target)

        return loss, probs, target, image.size(0)

    def training_step(self, batch, batch_idx):
        loss, _, _, batch_size = self._shared_step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, target, batch_size = self._shared_step(batch)
        target_int = target.int()
        preds = self._get_preds(probs)

        self.val_auroc.update(probs, target_int)
        self.val_ap.update(probs, target_int)

        self.val_precision.update(preds, target_int)
        self.val_recall.update(preds, target_int)
        self.val_f1.update(preds, target_int)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss, probs, target, batch_size = self._shared_step(batch)
        target_int = target.int()
        preds = self._get_preds(probs)

        self.test_auroc.update(probs, target_int)
        self.test_ap.update(probs, target_int)

        self.test_precision.update(preds, target_int)
        self.test_recall.update(preds, target_int)
        self.test_f1.update(preds, target_int)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/ap", self.val_ap.compute(), prog_bar=True)
        self.log("val/precision", self.val_precision.compute(), prog_bar=True)
        self.log("val/recall", self.val_recall.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)

        self.val_auroc.reset()
        self.val_ap.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        self.log("test/auroc", self.test_auroc.compute(), prog_bar=True)
        self.log("test/ap", self.test_ap.compute(), prog_bar=True)
        self.log("test/precision", self.test_precision.compute(), prog_bar=True)
        self.log("test/recall", self.test_recall.compute(), prog_bar=True)
        self.log("test/f1", self.test_f1.compute(), prog_bar=True)

        self.test_auroc.reset()
        self.test_ap.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/ap",
            },
        }