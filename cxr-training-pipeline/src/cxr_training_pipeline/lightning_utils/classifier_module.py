from cxr_training_pipeline.models.paper_based.w_cel import BatchBalancedBCEWithLogitsLoss
from torch import nn
from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)


class BaseClassifier(pl.LightningModule, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, image, retain_transition_grad: bool = False):
        pass

    @abstractmethod
    def set_thresholds(self, thresholds: torch.Tensor):
        pass

    @abstractmethod
    def _get_preds(self, probs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _shared_step(self, batch):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass


class ClassifierModule(BaseClassifier):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 14,
        lr: float = 1e-4,
        threshold: float = 0.5,
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        optimizer_kwargs: dict = None,
        scheduler_kwargs: dict = None,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class

        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.criterion = BatchBalancedBCEWithLogitsLoss()

        default_thresholds = torch.full((num_classes,), float(threshold), dtype=torch.float32)
        self.register_buffer("thresholds", default_thresholds)

        # probs collection
        prob_metrics_macro = MetricCollection(
            {
                "auroc_macro": MultilabelAUROC(num_labels=num_classes, average="macro"),
                "ap_macro": MultilabelAveragePrecision(num_labels=num_classes, average="macro"),
            }
        )

        prob_metrics_micro = MetricCollection(
            {
                "auroc_micro": MultilabelAUROC(num_labels=num_classes, average="micro"),
                "ap_micro": MultilabelAveragePrecision(num_labels=num_classes, average="micro"),
            }
        )

        self.val_prob_metrics_macro = prob_metrics_macro.clone(prefix="val/")
        self.test_prob_metrics_macro = prob_metrics_macro.clone(prefix="test/")
        self.val_prob_metrics_micro = prob_metrics_micro.clone(prefix="val/")
        self.test_prob_metrics_micro = prob_metrics_micro.clone(prefix="test/")


    def forward(self, image, retain_transition_grad: bool = False):
        return self.model(image, retain_transition_grad=retain_transition_grad)

    def set_thresholds(self, thresholds: torch.Tensor):
        thresholds = torch.as_tensor(thresholds, dtype=torch.float32, device=self.device)

        if thresholds.ndim != 1:
            raise ValueError(f"thresholds must be 1D, got shape={thresholds.shape}")

        if thresholds.numel() != self.hparams.num_classes:
            raise ValueError(f"thresholds must have {self.hparams.num_classes} elements, got {thresholds.numel()}")

        self.thresholds.copy_(thresholds)

    def _shared_inference(self, batch):
        image = batch["image"]
        target = batch["target"].float()

        out = self(image)
        logits = out["logits"]
        probs = torch.sigmoid(logits)

        return logits, probs, target

    def _get_preds(self, probs: torch.Tensor) -> torch.Tensor:
        thresholds = self.thresholds.view(1, -1).to(probs.device)
        return (probs > thresholds).int()

    def _shared_step(self, batch):
        logits, probs, target = self._shared_inference(batch)
        loss = self.criterion(logits, target)
        return loss, probs, target, target.size(0)

    def training_step(self, batch, batch_idx):
        loss, _, _, batch_size = self._shared_step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, target, batch_size = self._shared_step(batch)
        target_int = target.int()

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(self.val_prob_metrics_macro(probs, target_int), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(self.val_prob_metrics_micro(probs, target_int), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return {"loss": loss.detach(), "probs": probs.detach(), "targets": target.detach()}

    def test_step(self, batch, batch_idx):
        loss, probs, target, batch_size = self._shared_step(batch)
        target_int = target.int()

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(self.test_prob_metrics_macro(probs, target_int), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(self.test_prob_metrics_micro(probs, target_int), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return {"loss": loss.detach(), "probs": probs.detach(), "targets": target.detach()}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, probs, target = self._shared_inference(batch)
        preds = self._get_preds(probs)
        return {
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(),
            "targets": target.detach().cpu(),
        }

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            **self.optimizer_kwargs,
        )

        if self.scheduler_class is None:
            return optimizer

        scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/ap_macro",
                },
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
