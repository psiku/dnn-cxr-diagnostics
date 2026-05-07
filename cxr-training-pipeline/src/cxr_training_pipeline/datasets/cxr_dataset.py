"""Old version of dataset class, using precomputed tensors. Kept for reference and potential future use, but not currently used in the pipeline."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ImageOnlyDataset(Dataset):
    """
    Custom PyTorch Dataset for chest X-ray images and labels. Each sample is a dictionary containing:
     - image_index - index of the image in the original dataframe
     - target - binary vector of pathologies
     - image - the chest X-ray image itself as a tensor.

     Args:
        df (pd.DataFrame): DataFrame containing the metadata and labels for the dataset.
        images_array (np.ndarray): Array of chest X-ray images.
        labels_array (np.ndarray): Array of labels corresponding to each image.
        indices (np.ndarray | list[int] | None, optional): Indices of the samples to be included in the dataset. If None, all samples are included. Defaults to None.
        transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_array: np.ndarray,
        labels_array: np.ndarray,
        indices: np.ndarray | list[int] | None = None,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.images_array = images_array
        self.labels_array = labels_array
        self.transform = transform

        if indices is None:
            self.indices = np.arange(len(self.df))
        else:
            self.indices = np.asarray(indices)

        assert len(self.df) == len(self.images_array) == len(self.labels_array), (
            "df, images_array, metadata_array and labels_array must have the same length"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]

        # [H, W], uint8 -> [1, H, W], float32 in [0, 1]
        img = self.images_array[idx]
        img = torch.from_numpy(np.array(img, copy=True)).unsqueeze(0).float() / 255.0

        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(self.labels_array[idx], dtype=torch.float32)
        image_index = self.df.iloc[idx]["image_index"]

        return {
            "image": img,
            "target": target,
            "image_index": image_index,
        }
