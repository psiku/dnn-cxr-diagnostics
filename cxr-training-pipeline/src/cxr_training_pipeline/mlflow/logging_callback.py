import mlflow
from lightning.pytorch.callbacks import Callback
import torch
import pytorch_lightning as pl


class MlflowMetricLoggingCallback(Callback):
    """Logs epoch metrics and learning rates to the active MLflow run."""

    @staticmethod
    def _log_callback_metrics(trainer: pl.Trainer) -> None:
        active_run = mlflow.active_run()
        if active_run is None:
            return

        step = int(trainer.current_epoch)
        for metric_name, metric_value in trainer.callback_metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach().cpu().item()
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, float(metric_value), step=step)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        active_run = mlflow.active_run()
        if active_run is not None:
            step = int(trainer.current_epoch)
            for opt_idx, optimizer in enumerate(trainer.optimizers):
                for group_idx, param_group in enumerate(optimizer.param_groups):
                    lr = param_group.get("lr")
                    if lr is not None:
                        mlflow.log_metric(f"lr/opt{opt_idx}_group{group_idx}", float(lr), step=step)
        self._log_callback_metrics(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_callback_metrics(trainer)