import numpy as np
from typing import Any, Dict, List
from pathlib import Path
from torchvision import transforms
import pytorch_lightning as pl
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import pandas as pd
from cxr_training_pipeline.lightning_utils.factory import DataModuleFactory, ClassifierFactory, LightningModuleFactory
import torch
import mlflow


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


# DATA PREPARATION NODE
def make_train_val_indices(train_val_df, params: dict[str, float | int]) -> tuple[np.ndarray, np.ndarray]:
    """Splits the dataframe into train and validation indices."""
    train_val_df = train_val_df.reset_index(drop=True)

    n_samples = len(train_val_df)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed=params["random_state"])
    rng.shuffle(indices)

    split_idx = int(n_samples * (1 - params["val_size"]))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)

    return train_idx, val_idx


def build_transforms_node(transform_params: Dict[str, Any]) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Creates PyTorch transforms for training and evaluation.
    Only training gets the augmentations.
    """
    train_tfms = transforms.Compose([
        transforms.RandomAffine(
            degrees=transform_params["random_rotation"],
            translate=tuple(transform_params["random_translate"]),
            scale=tuple(transform_params["random_scale"])
        ),
        # transforms.ToTensor(), # just in case if we switch to tensor-based transforms in the future
    ])

    eval_tfms = transforms.Compose([
        # transforms.ToTensor(),
    ])

    return train_tfms, eval_tfms


# DATAMODULE NODE
def prepare_datamodule_config(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_tfms: transforms.Compose,
    eval_tfms: transforms.Compose,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merges catalog data and YAML parameters into a single configuration dictionary.
    """

    config = parameters.get("kwargs", {})

    config.update(
        {
            "train_val_df": train_val_df,
            "test_df": test_df,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_tfms": train_tfms,
            "eval_tfms": eval_tfms,
        }
    )

    return config


def build_datamodule_node(datamodule_name: str, datamodule_config: dict):
    """
    datamodule_config will collect all named inputs into a dict.
    """
    return DataModuleFactory.create(module_name=datamodule_name, **datamodule_config)


# CLASSIFIER NODE
def build_classifier_node(classifier_params: Dict[str, Any]):
    """Creates the backbone model using the Factory."""
    return ClassifierFactory.create(classifier_name=classifier_params["name"], **classifier_params["kwargs"])


def build_lightning_module_node(model: torch.nn.Module, lit_params: Dict[str, Any]):
    """Wraps the backbone model in the PyTorch Lightning module."""
    return LightningModuleFactory.create(module_name=lit_params["name"], model=model, **lit_params["kwargs"])


# TRAINING NODE
def _setup_callbacks(trainer_params: Dict[str, Any]) -> List[Callback]:
    """Configures training callbacks."""
    checkpoint_dir = Path(trainer_params.get("checkpoint_dir", "checkpoints")).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{val_ap_macro:.4f}",
        monitor="val/ap_macro",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/ap_macro",
        mode="max",
        patience=trainer_params.get("patience", 5),
    )

    callbacks: List[Callback] = [checkpoint_callback, early_stopping_callback]
    callbacks.append(MlflowMetricLoggingCallback())
    return callbacks


def _setup_trainer(
    trainer_params: Dict[str, Any],
    callbacks: List[Callback],
) -> pl.Trainer:
    """Initializes the PyTorch Lightning Trainer with the specified parameters and callbacks."""
    return pl.Trainer(
        max_epochs=trainer_params["max_epochs"],
        default_root_dir=str(Path.cwd().resolve()),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=trainer_params.get("precision", 32),
        logger=False,
        callbacks=callbacks,
        log_every_n_steps=trainer_params.get("log_every_n_steps", 10),
        accumulate_grad_batches=trainer_params.get("accumulate_grad_batches", 1),
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )


def _log_best_checkpoint(checkpoint_callback: ModelCheckpoint) -> None:
    if checkpoint_callback.best_model_path:
        mlflow.log_param("best_model_path", checkpoint_callback.best_model_path)

    if checkpoint_callback.best_model_score is not None:
        mlflow.log_metric(
            "best_model_score",
            float(checkpoint_callback.best_model_score.cpu().item()),
        )



def _create_inference_trainer(trainer_params: Dict[str, Any]) -> pl.Trainer:
    """Creates a lightweight Trainer without logging or checkpoints for inference."""
    return pl.Trainer(
        accelerator=trainer_params.get("accelerator", "auto"),
        devices=trainer_params.get("devices", "auto"),
        precision=trainer_params.get("precision", "32-true"),
        logger=False,
        enable_checkpointing=False,
    )


def _load_model_from_checkpoint(
    lit_model: pl.LightningModule, best_checkpoint_path: str
) -> pl.LightningModule:
    """Loads a Lightning module from a checkpoint path using the original model class."""
    model_cls = lit_model.__class__
    return model_cls.load_from_checkpoint(
        best_checkpoint_path,
        model=lit_model.model,
    )


def train_model_node(
    datamodule: pl.LightningDataModule,
    lit_model: pl.LightningModule,
    trainer_params: Dict[str, Any],
) -> pl.LightningModule:
    """Handles callbacks, loggers, fitting, testing, and metrics tracking."""
    torch.set_float32_matmul_precision("high")

    callbacks = _setup_callbacks(trainer_params)
    trainer = _setup_trainer(trainer_params, callbacks)

    mlflow.log_params(trainer_params)

    datamodule.setup(stage="fit")
    trainer.fit(model=lit_model, datamodule=datamodule)

    checkpoint_callback = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    _log_best_checkpoint(checkpoint_callback)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_model_path}")

    datamodule.setup(stage="test")
    trainer.test(model=lit_model, datamodule=datamodule, ckpt_path="best")

    return best_model_path


# PREDICTIONS NODES
def predict_validation_node(
    datamodule: pl.LightningDataModule,
    lit_model: pl.LightningModule,
    best_checkpoint_path: str,
    trainer_params: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the best checkpoint and create validation probabilities + targets."""
    trainer = _create_inference_trainer(trainer_params)
    model = _load_model_from_checkpoint(lit_model, best_checkpoint_path)

    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()

    predictions = trainer.predict(model=model, dataloaders=val_loader)

    y_proba = np.concatenate([batch["probs"].cpu().numpy() for batch in predictions], axis=0)
    y_true = np.concatenate([batch["targets"].cpu().numpy() for batch in predictions], axis=0)

    return y_proba, y_true


def predict_test_node(
    datamodule: pl.LightningDataModule,
    lit_model: pl.LightningModule,
    best_checkpoint_path: str,
    trainer_params: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the best checkpoint and create test probabilities + targets."""
    trainer = _create_inference_trainer(trainer_params)
    model = _load_model_from_checkpoint(lit_model, best_checkpoint_path)

    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    predictions = trainer.predict(model=model, dataloaders=test_loader)

    test_pred_proba = np.concatenate([batch["probs"].cpu().numpy() for batch in predictions], axis=0)
    test_targets = np.concatenate([batch["targets"].cpu().numpy() for batch in predictions], axis=0)

    return test_pred_proba, test_targets
