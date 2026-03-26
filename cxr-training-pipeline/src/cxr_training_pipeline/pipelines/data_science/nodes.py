import numpy as np
from typing import Any, Dict, List
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
import pandas as pd
from cxr_training_pipeline.lightning_utils.factory import DataModuleFactory, ClassifierFactory, LightningModuleFactory
from cxr_training_pipeline.lightning_utils.classifier_module import ClassifierModule, BaseClassifier
import torch
import mlflow


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
def _setup_logger(mlflow_params: Dict[str, Any]) -> MLFlowLogger:
    """Initializes the MLFlow logger."""
    return MLFlowLogger(
        experiment_name=mlflow_params["experiment_name"],
        tracking_uri=mlflow_params["tracking_uri"],
        run_name=mlflow_params["run_name"],
        log_model="all",
    )


def _setup_callbacks(trainer_params: Dict[str, Any]) -> List[Callback]:
    """Configures training callbacks."""
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
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

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [checkpoint_callback, early_stopping_callback, lr_monitor]


def _setup_trainer(trainer_params: Dict[str, Any], logger: MLFlowLogger, callbacks: List[Callback]) -> pl.Trainer:
    """Initializes the PyTorch Lightning trainer."""
    return pl.Trainer(
        max_epochs=trainer_params["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=trainer_params.get("precision", 32),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=trainer_params.get("log_every_n_steps", 10),
        accumulate_grad_batches=trainer_params.get("accumulate_grad_batches", 1),
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )


def _log_best_checkpoint(logger: MLFlowLogger, checkpoint_callback: ModelCheckpoint) -> None:
    """Logs the best model path and score to MLFlow."""
    if checkpoint_callback.best_model_path:
        logger.experiment.log_param(logger.run_id, "best_model_path", checkpoint_callback.best_model_path)
    if checkpoint_callback.best_model_score is not None:
        logger.experiment.log_metric(
            logger.run_id, "best_model_score", float(checkpoint_callback.best_model_score.cpu().item())
        )


def _log_test_metrics(logger: MLFlowLogger, test_results: Any) -> None:
    """Logs final test metrics to MLFlow."""
    if test_results and len(test_results) > 0:
        for metric_name, metric_value in test_results[0].items():
            logger.experiment.log_metric(logger.run_id, f"final_{metric_name}", float(metric_value))


def train_model_node(
    datamodule: pl.LightningDataModule,
    lit_model: pl.LightningModule,
    trainer_params: Dict[str, Any],
    mlflow_params: Dict[str, Any],
) -> pl.LightningModule:
    """Handles callbacks, loggers, fitting, testing, and metrics tracking."""
    torch.set_float32_matmul_precision("high")

    logger = _setup_logger(mlflow_params)
    callbacks = _setup_callbacks(trainer_params)

    trainer = _setup_trainer(trainer_params, logger, callbacks)

    datamodule.setup(stage="fit")
    trainer.fit(model=lit_model, datamodule=datamodule)

    checkpoint_callback = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    _log_best_checkpoint(logger, checkpoint_callback)

    datamodule.setup(stage="test")
    test_results = trainer.test(
        model=lit_model,
        datamodule=datamodule,
        ckpt_path="best",
    )

    _log_test_metrics(logger, test_results)

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Test results: {test_results}")

    return lit_model
