import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TensorCXRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        label_cols: list[str],
        mask_path_col: str = None,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.mask_path_col = mask_path_col
        self.label_cols = label_cols
        self.transform = transform

        assert self.path_col in self.df.columns, f"Path column '{self.path_col}' missing."
        if self.mask_path_col:
            assert self.mask_path_col in self.df.columns, f"Mask Path column '{self.mask_path_col}' missing."

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_index = row["image_index"]

        # Load Image
        tensor_path = row[self.path_col]
        img_tensor = torch.load(tensor_path, weights_only=False) # Shape: [C, H, W]

        # Load Mask and combine
        if self.mask_path_col and pd.notna(row[self.mask_path_col]):
            mask_path = row[self.mask_path_col]
            mask_tensor = torch.load(mask_path, weights_only=False) # Assumed shape: [1, H, W]

            # Stack along channel dimension -> Shape: [C+1, H, W]
            img_tensor = torch.cat([img_tensor, mask_tensor], dim=0)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        target = torch.tensor(row[self.label_cols].values.astype(np.float32), dtype=torch.float32)

        return {
            "image": img_tensor,
            "target": target,
            "image_index": image_index,
        }