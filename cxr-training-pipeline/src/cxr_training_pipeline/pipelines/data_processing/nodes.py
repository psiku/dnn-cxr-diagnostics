from typing import Any
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm




# Creating the dataframes with only the necessary columns and one-hot encoding the labels
def _clean_dataframe(df: pd.DataFrame, columns_to_keep: list[str]) -> pd.DataFrame:
    """ "Helper function to clean up the dataframe by keeping only the necessary columns"""
    return df[columns_to_keep]


def _create_ohe_encoding(df: pd.DataFrame, column: str, pathology_list: list[str]) -> pd.DataFrame:
    """Helper function to create one-hot encoded columns"""
    labels = df[column].str.get_dummies(sep="|")
    df = df.join(labels[pathology_list])
    return df


def _load_split_list(split_file: str) -> list[str]:
    """Helper function to clean up the raw text into a list"""
    return [line.strip() for line in split_file.splitlines() if line.strip()]


def _process_split(
    df: pd.DataFrame, split_list: list[str], columns_to_keep: list[str], column: str, pathology_list: list[str]
) -> pd.DataFrame:
    """Helper function to process a split by filtering, cleaning, and one-hot encoding"""
    split_df = df[df["image_index"].isin(split_list)].copy()
    split_df = _clean_dataframe(split_df, columns_to_keep)
    split_df = _create_ohe_encoding(split_df, column, pathology_list)
    return split_df


def create_train_val_test_dfs(
    df: pd.DataFrame,
    columns_to_keep: list[str],
    train_val_split_file: str,
    test_split_file: str,
    pathology_list: list[str],
    column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates the train_val and test dataframes with the necessary columns and one-hot encoded labels"""

    # Process train_val split
    train_val_list = _load_split_list(train_val_split_file)
    train_val_df = _process_split(df, train_val_list, columns_to_keep, column, pathology_list)

    # Process test split
    test_list = _load_split_list(test_split_file)
    test_df = _process_split(df, test_list, columns_to_keep, column, pathology_list)

    return train_val_df, test_df


# Precomputing images to npy files for faster loading during training
def _build_transform_from_config(config: dict[str, Any]) -> transforms.Compose:
    steps = []

    if config.get("grayscale", True):
        steps.append(transforms.Grayscale(num_output_channels=1))

    if "resize" in config and config["resize"] is not None:
        resize_value = config["resize"]
        if isinstance(resize_value, (list, tuple)):
            steps.append(transforms.Resize(tuple(resize_value)))
        else:
            steps.append(transforms.Resize(resize_value))

    if "center_crop" in config and config["center_crop"] is not None:
        steps.append(transforms.CenterCrop(config["center_crop"]))

    steps.append(transforms.ToTensor())  # [1, H, W], float in [0,1]

    if "normalize" in config and config["normalize"] is not None:
        norm_cfg = config["normalize"]
        steps.append(
            transforms.Normalize(
                mean=norm_cfg["mean"],
                std=norm_cfg["std"],
            )
        )

    return transforms.Compose(steps)


def _precompute_images_to_npy(
    df: pd.DataFrame,
    images_dir: str,
    output_path: str,
    image_size: int | None = None,
    image_col: str = "image_index",
    transform_config: dict[str, Any] | None = None,
) -> None:
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform_config = transform_config or {}
    transform = _build_transform_from_config(transform_config)

    n = len(df)

    use_float32 = (
        "normalize" in transform_config and transform_config["normalize"] is not None
    )
    dtype = np.float32 if use_float32 else np.uint8

    arr = np.lib.format.open_memmap(
        output_path,
        dtype=dtype,
        mode="w+",
        shape=(n, image_size, image_size),
    )

    for i, image_name in enumerate(tqdm(df[image_col].tolist(), total=n)):
        img_path = images_dir / image_name

        with Image.open(img_path) as img:
            img = transform(img)           # [1, H, W]
            img = img.squeeze(0).numpy()   # [H, W]

            if img.shape != (image_size, image_size):
                raise ValueError(
                    f"Transformed image {image_name} has shape {img.shape}, "
                    f"expected ({image_size}, {image_size})."
                )

            if dtype == np.uint8:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.astype(np.float32)

            arr[i] = img

    arr.flush()
    print(f"Saved memmap to: {output_path}")


def precompute_images_to_npy(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: str,
    train_val_output_path: str,
    test_output_path: str,
    image_transform_config: dict[str, Any],
    image_size: int | None = None,
    image_col: str = "image_index",
) -> None:
    """Precompute train/val and test images to .npy memmap files."""
    if Path(train_val_output_path).exists() and Path(test_output_path).exists():
        print("Precomputed image files already exist. Skipping precomputation.")
        return

    if not Path(train_val_output_path).exists():
        _precompute_images_to_npy(
            df=train_val_df,
            images_dir=images_dir,
            output_path=train_val_output_path,
            image_size=image_size,
            image_col=image_col,
            transform_config=image_transform_config,
        )

    if not Path(test_output_path).exists():
        _precompute_images_to_npy(
            df=test_df,
            images_dir=images_dir,
            output_path=test_output_path,
            image_size=image_size,
            image_col=image_col,
            transform_config=image_transform_config,
        )


# Precomputing labels to npy files for faster loading during training
def _precompute_labels_to_npy(
    df: pd.DataFrame,
    pathology_list: list[str],
) -> np.ndarray:

    y_labels = df[pathology_list].values.astype(np.float32)
    return y_labels


def precompute_labels(
    train_val_df: pd.DataFrame, test_df: pd.DataFrame, pathology_list: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Precomputes the labels to npy files for faster loading during training"""
    train_val_labels = _precompute_labels_to_npy(train_val_df, pathology_list)
    test_labels = _precompute_labels_to_npy(test_df, pathology_list)

    return train_val_labels, test_labels
