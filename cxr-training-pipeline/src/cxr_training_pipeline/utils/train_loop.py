import mlflow
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np

from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

from cxr_training_pipeline.lightning_utils.classifier_module import ClassifierModule
from cxr_training_pipeline.lightning_utils.data_module import ImageOnlyDataModule
from cxr_training_pipeline.models.paper_based.classifier import CXRClassifier


def make_train_val_indices(train_val_df, val_size=0.25, random_state=42):
    train_val_df = train_val_df.reset_index(drop=True)

    n_samples = len(train_val_df)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(indices)

    split_idx = int(n_samples * (1 - val_size))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)

    return train_idx, val_idx


def main():
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    EXPERIMENT_NAME = "cxr-classification-paper-resnet50"
    RUN_NAME = "paper_resnet50_512_lse"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    train_tfms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ]
    )
    eval_tfms = None

    TRAIN_CSV = "C:\\Users\\barte\\OneDrive\\Pulpit\\dnn-cxr-diagnostics\\cxr-training-pipeline\\data\\02_intermediate\\dataset_splits\\train_val.csv"
    TEST_CSV = "C:\\Users\\barte\\OneDrive\\Pulpit\\dnn-cxr-diagnostics\\cxr-training-pipeline\\data\\02_intermediate\\dataset_splits\\test.csv"

    TRAIN_VAL_IMAGES = "C:\\Users\\barte\\OneDrive\\Pulpit\\dnn-cxr-diagnostics\\cxr-training-pipeline\\data\\02_intermediate\\precomputed_images\\train_val_images.npy"
    TEST_IMAGES = "C:\\Users\\barte\\OneDrive\\Pulpit\\dnn-cxr-diagnostics\\cxr-training-pipeline\\data\\02_intermediate\\precomputed_images\\test_images.npy"

    Y_TRAIN = np.load(
        "C:\\Users\\barte\\OneDrive\\Pulpit\\dnn-cxr-diagnostics\\cxr-training-pipeline\\data\\02_intermediate\\precomputed_metadata\\y_train_val.npy"
    )
    Y_TEST = np.load(
        "C:\\Users\\barte\\OneDrive\\Pulpit\\dnn-cxr-diagnostics\\cxr-training-pipeline\\data\\02_intermediate\\precomputed_metadata\\y_test.npy"
    )

    train_val_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    images_train_val = np.memmap(
        TRAIN_VAL_IMAGES,
        dtype="uint8",
        mode="r",
        shape=(len(train_val_df), 512, 512),
    )

    images_test = np.memmap(
        TEST_IMAGES,
        dtype="uint8",
        mode="r",
        shape=(len(test_df), 512, 512),
    )

    train_idx, val_idx = make_train_val_indices(train_val_df, val_size=0.25, random_state=42)

    datamodule = ImageOnlyDataModule(
        train_val_images=images_train_val,
        test_images=images_test,
        y_train_val=Y_TRAIN,
        y_test=Y_TEST,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=8,
        num_workers=4,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
    )

    model = CXRClassifier(
        num_classes=14,
        transition_dim=2048,
        pooling="lse",
        lse_r=10.0,
        pretrained=True,
        grayscale=True,
        freeze_backbone=False,
    )

    lit_model = ClassifierModule(
        model=model,
        num_classes=14,
        lr=1e-4,
        threshold=0.5,
    )

    logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=RUN_NAME,
        log_model="all",  # loguje checkpointy jako artefakty
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_ap:.4f}",
        monitor="val/ap",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/ap",
        mode="max",
        patience=5,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
        ],
        log_every_n_steps=10,
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    # add parameters to mlflow
    logger.experiment.log_params(
        run_id=logger.run_id,
        params={
            "seed": 42,
            "batch_size": 8,
            "num_workers": 4,
            "max_epochs": 20,
            "lr": 1e-4,
            "precision": "16-mixed" if torch.cuda.is_available() else 32,
            "accumulate_grad_batches": 1,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_df)),
            "num_classes": 14,
            "image_size": 512,
            "model_name": "CXRClassifier",
            "backbone": "resnet50",
            "pooling": "lse",
            "lse_r": 10.0,
            "pretrained": True,
            "grayscale": True,
            "freeze_backbone": False,
            "threshold": 0.5,
        },
    )

    # Train
    datamodule.setup(stage="fit")
    trainer.fit(model=lit_model, datamodule=datamodule)

    if checkpoint_callback.best_model_path:
        logger.experiment.log_param(
            logger.run_id,
            "best_model_path",
            checkpoint_callback.best_model_path,
        )

    if checkpoint_callback.best_model_score is not None:
        logger.experiment.log_metric(
            logger.run_id,
            "best_model_score",
            float(checkpoint_callback.best_model_score.cpu().item()),
        )

    datamodule.setup(stage="test")
    test_results = trainer.test(
        model=lit_model,
        datamodule=datamodule,
        ckpt_path="best",
    )

    if len(test_results) > 0:
        for metric_name, metric_value in test_results[0].items():
            logger.experiment.log_metric(
                logger.run_id,
                f"final_{metric_name}",
                float(metric_value),
            )

    print("Best checkpoint:", checkpoint_callback.best_model_path)
    print("Test results:", test_results)


if __name__ == "__main__":
    main()
