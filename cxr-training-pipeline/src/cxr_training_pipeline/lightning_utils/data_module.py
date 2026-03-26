import pytorch_lightning as pl
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchvision import transforms
from cxr_training_pipeline.datasets.cxr_dataset import ImageOnlyDataset


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def setup(self, stage=None):
        pass


class ImageOnlyDataModule(BaseDataModule):
    def __init__(
        self,
        train_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_val_images_path: str,
        test_images_path: str,
        y_train_val_path: str,
        y_test_path: str,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        batch_size: int = 8,
        num_workers: int = 4,
        train_tfms: transforms.Compose = transforms.Compose([]),
        eval_tfms: transforms.Compose = transforms.Compose([]),
    ):
        super().__init__()
        self.train_val_df = train_val_df
        self.test_df = test_df
        self.train_val_images_path = train_val_images_path
        self.test_images_path = test_images_path
        self.y_train_val_path = y_train_val_path
        self.y_test_path = y_test_path
        self.train_idx = np.asarray(train_idx)
        self.val_idx = np.asarray(val_idx)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_tfms = train_tfms
        self.eval_tfms = eval_tfms

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Load memmaps late to avoid saving massive arrays in external workflows
        train_val_images = np.load(self.train_val_images_path, mmap_mode="r")
        y_train_val = np.load(self.y_train_val_path, mmap_mode="r")
        test_images = np.load(self.test_images_path, mmap_mode="r")
        y_test = np.load(self.y_test_path, mmap_mode="r")

        if stage == "fit" or stage is None:
            self.train_dataset = ImageOnlyDataset(
                df=self.train_val_df,
                images_array=train_val_images,
                labels_array=y_train_val,
                indices=self.train_idx,
                transform=self.train_tfms,
            )

            self.val_dataset = ImageOnlyDataset(
                df=self.train_val_df,
                images_array=train_val_images,
                labels_array=y_train_val,
                indices=self.val_idx,
                transform=self.eval_tfms,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageOnlyDataset(
                df=self.test_df,
                images_array=test_images,
                labels_array=y_test,
                indices=None,
                transform=self.eval_tfms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
