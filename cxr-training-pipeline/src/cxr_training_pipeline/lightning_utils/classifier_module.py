from cxr_training_pipeline.lightning_utils.loss import build_loss_criterion
from torch import nn
from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl
from collections.abc import Mapping
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
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
        loss: dict | None = None,
        pos_weight: torch.Tensor | list[float] | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_cfg = loss or {"name": "batch_balanced_bce"}
        pos_weight_tensor = (
            torch.as_tensor(pos_weight, dtype=torch.float32)
            if pos_weight is not None
            else None
        )
        self.save_hyperparameters(ignore=["model"])
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class

        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.criterion = build_loss_criterion(self.loss_cfg, pos_weight_tensor)

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
        # Some backbones expose retain_transition_grad (e.g. ChestXRayClassifier),
        # while simpler models accept only the image tensor.
        try:
            return self.model(image, retain_transition_grad=retain_transition_grad)
        except TypeError as exc:
            if "retain_transition_grad" not in str(exc):
                raise
            return self.model(image)

    @staticmethod
    def _extract_logits(model_output):
        if isinstance(model_output, torch.Tensor):
            return model_output

        if isinstance(model_output, Mapping):
            logits = model_output.get("logits")
            if isinstance(logits, torch.Tensor):
                return logits
            raise TypeError("Model output mapping must contain tensor under key 'logits'.")

        raise TypeError(
            "Model output must be either a logits tensor or a mapping containing 'logits'."
        )

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
        logits = self._extract_logits(out)
        probs = torch.sigmoid(logits)

        return logits, probs, target

    def _get_preds(self, probs: torch.Tensor) -> torch.Tensor:
        thresholds = self.thresholds.view(1, -1).to(probs.device)
        return (probs >= thresholds).int()

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

        self.val_prob_metrics_macro.update(probs, target_int)
        self.val_prob_metrics_micro.update(probs, target_int)

        self.log_dict(self.val_prob_metrics_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_prob_metrics_micro, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss.detach(), "probs": probs.detach(), "targets": target.detach()}

    def test_step(self, batch, batch_idx):
        loss, probs, target, batch_size = self._shared_step(batch)
        target_int = target.int()

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        self.test_prob_metrics_macro.update(probs, target_int)
        self.test_prob_metrics_micro.update(probs, target_int)

        self.log_dict(self.test_prob_metrics_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_prob_metrics_micro, on_step=False, on_epoch=True, prog_bar=False)

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
