import numpy as np
from typing import Any, Dict, List
from pathlib import Path
from torchvision import transforms
import pytorch_lightning as pl
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
import pandas as pd
from cxr_training_pipeline.lightning_utils.factory import DataModuleFactory, ClassifierFactory, LightningModuleFactory
from cxr_training_pipeline.lightning_utils.loss import resolve_pos_weights
import torch
import mlflow
from cxr_training_pipeline.mlflow.logging_callback import MlflowMetricLoggingCallback


# DATA PREPARATION NODE
def _rebalance_train_indices(
    train_val_df: pd.DataFrame,
    train_idx: np.ndarray,
    rebalance_cfg: dict[str, Any] | None,
    default_random_state: int = 42,
) -> np.ndarray:
    """Apply optional under/over sampling for selected training subsets."""
    if not rebalance_cfg:
        return train_idx
    if not bool(rebalance_cfg.get("enabled", False)):
        return train_idx

    rng = np.random.default_rng(
        seed=int(rebalance_cfg.get("random_state", default_random_state))
    )
    train_df = train_val_df.iloc[train_idx].copy()
    selected_indices = list(train_idx.astype(np.int64))

    def _selector_pool(selector: str) -> np.ndarray:
        if selector == "no_finding":
            if "finding_labels" not in train_df.columns:
                raise ValueError(
                    "Cannot use selector 'no_finding' because column 'finding_labels' is missing."
                )
            mask = train_df["finding_labels"].fillna("").astype(str).str.strip() == "No Finding"
            return train_df.index[mask].to_numpy(dtype=np.int64)

        if selector not in train_df.columns:
            raise ValueError(f"Selector '{selector}' not found in train dataframe columns.")

        return train_df.index[train_df[selector].astype(int) == 1].to_numpy(dtype=np.int64)

    rules = rebalance_cfg.get("rules")
    if isinstance(rules, list) and rules:
        normalized_rules = rules

    for rule in normalized_rules:
        selector = str(rule.get("selector", "")).strip()
        mode = str(rule.get("mode", "")).strip().lower()
        target = int(rule.get("target_count", -1))

        if not selector:
            raise ValueError("Each class_rebalance rule must include a non-empty 'selector'.")
        if mode not in {"under", "over"}:
            raise ValueError(
                f"class_rebalance rule for '{selector}' has invalid mode '{mode}'."
            )
        if target < 0:
            raise ValueError(
                f"class_rebalance rule for '{selector}' has invalid target_count={target}."
            )

        pool = _selector_pool(selector)
        current_count = len(pool)
        if current_count == 0:
            continue

        if mode == "over" and target > current_count:
            extra = rng.choice(pool, size=target - current_count, replace=True)
            selected_indices.extend(extra.tolist())
        elif mode == "under" and target < current_count:
            keep_pool = set(rng.choice(pool, size=target, replace=False).tolist())
            pool_set = set(pool.tolist())
            selected_indices = [
                idx for idx in selected_indices if idx not in pool_set or idx in keep_pool
            ]

    return np.asarray(selected_indices, dtype=np.int64)


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

    train_idx = _rebalance_train_indices(
        train_val_df=train_val_df,
        train_idx=train_idx,
        rebalance_cfg=params.get("class_rebalance"),
        default_random_state=int(params["random_state"]),
    )

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)

    return train_idx, val_idx


def build_transforms_node(transform_params: Dict[str, Any]) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Creates PyTorch transforms for training and evaluation.
    Only training gets the augmentations.
    """
    normalize_mean = transform_params.get("normalize_mean")
    normalize_std = transform_params.get("normalize_std")
    apply_normalization = bool(transform_params.get("apply_normalization", True))

    train_ops: list[Any] = [
        transforms.RandomAffine(
            degrees=transform_params["random_rotation"],
            translate=tuple(transform_params["random_translate"]),
            scale=tuple(transform_params["random_scale"]),
        ),
        transforms.Normalize(mean=normalize_mean, std=normalize_std) if apply_normalization else None,
    ]
    eval_ops: list[Any] = [transforms.Normalize(mean=normalize_mean, std=normalize_std) if apply_normalization else None]

    train_tfms = transforms.Compose(train_ops)
    eval_tfms = transforms.Compose(eval_ops)

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
    """
    Creates the backbone model using the Factory.
    Example:
        classifier_params = {
            "name": "chest_xray_classifier",
            "kwargs": {
                "num_classes": 14,
                "backbone_name": "resnet50",
                "pretrained": True,
                "pooling": "lse",
                "lse_r": 10.0,
            }
        }
    """
    classifier_name = classifier_params.get("name")
    if not classifier_name:
        raise ValueError(
            "classifier_params must contain 'name' key. "
            "Supported values: 'cxr_classifier', 'chest_xray_classifier'"
        )

    kwargs = classifier_params.get("kwargs", {})
    return ClassifierFactory.create(classifier_name=classifier_name, **kwargs)


def compute_class_pos_weights_node(
    train_val_df: pd.DataFrame,
    train_idx: np.ndarray,
    datamodule_params: Dict[str, Any],
    lit_module_params: Dict[str, Any],
) -> list[float]:
    """Compute per-class pos_weight from the training split when configured.

    Returns an empty list when pos_weight is not requested. We avoid returning
    None because Kedro forbids saving None to any dataset.
    """
    loss_cfg = lit_module_params.get("kwargs", {}).get("loss")
    label_cols = datamodule_params.get("kwargs", {}).get("label_cols")
    train_df = train_val_df.iloc[train_idx]

    pos_weight = resolve_pos_weights(
        loss_cfg=loss_cfg,
        train_df=train_df,
        label_cols=label_cols,
    )
    if pos_weight is None:
        return []

    weights = pos_weight.detach().cpu().tolist()
    loss_name = (loss_cfg or {}).get("name", "batch_balanced_bce")
    print(f"Computed pos_weight for loss '{loss_name}':")
    for class_name, weight in zip(label_cols, weights):
        print(f"  {class_name}: {weight:.4f}")
    return weights


def build_lightning_module_node(
    model: torch.nn.Module,
    lit_params: Dict[str, Any],
    pos_weights: list[float] | None = None,
):
    """
    Wraps the backbone model in the PyTorch Lightning module.
    Example:
        lit_params = {
            "name": "default_classifier",
            "kwargs": {
                "num_classes": 14,
                "lr": 1e-4,
                "threshold": 0.5,
                "loss": {
                    "name": "bce_with_logits",
                    "pos_weight": {"compute_from_train": true},
                },
            }
        }
    """
    module_name = lit_params.get("name", "default_classifier")
    kwargs = dict(lit_params.get("kwargs", {}))
    if pos_weights:
        kwargs["pos_weight"] = pos_weights
    return LightningModuleFactory.create(module_name=module_name, model=model, **kwargs)


# TRAINING NODE
def _setup_callbacks(trainer_params: Dict[str, Any]) -> List[Callback]:
    """Configures training callbacks."""
    experiment_name = trainer_params.get("mlflow_experiment_name", "default_experiment")
    checkpoint_dir = Path("data/06_models") / experiment_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}",
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
