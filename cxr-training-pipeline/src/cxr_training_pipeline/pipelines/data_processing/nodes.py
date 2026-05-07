from typing import Any
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os
import torch
from torchvision.transforms import v2
from cxr_training_pipeline.utils.segmentation_utils import (
    get_combined_mask_by_image_index,
    get_bbox_coordinates,
    expand_bbox,
)

# const transform
to_tensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

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


def _get_cropped_arrays(image_path: str, image_index: str, masks_df: pd.DataFrame):
    """
    Helper function to load an image, get the combined mask, compute the bounding box, and return the cropped image and mask arrays.
    """
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    height, width = img_np.shape[:2]

    combined_mask = get_combined_mask_by_image_index(
        image_index=image_index,
        segmentation_masks_df=masks_df,
        height=height,
        width=width
    )

    bbox = get_bbox_coordinates(combined_mask)

    if bbox is not None:
        bbox = expand_bbox(bbox, img_shape=(height, width), margin_ratio=0.01)
        x_min, y_min, x_max, y_max = bbox

        cropped_img_np = img_np[y_min:y_max, x_min:x_max]
        cropped_mask_np = combined_mask[y_min:y_max, x_min:x_max]
    else:
        cropped_img_np = img_np
        cropped_mask_np = combined_mask

    return cropped_img_np, cropped_mask_np


def _create_cropped_image_tensor(image_path: str, image_index: str, masks_df: pd.DataFrame, image_size: int = 224) -> torch.Tensor:
    """
    Helper to create a tensor for the cropped image. It loads the image, gets the combined mask, computes the bounding box, crops the image, resizes it, and converts it to a tensor.
    """
    cropped_img_np, _ = _get_cropped_arrays(image_path, image_index, masks_df)


    img_resized = Image.fromarray(cropped_img_np).resize((image_size, image_size), Image.BILINEAR)
    img_tensor = to_tensor(img_resized) # Shape: [3, 224, 224]

    return img_tensor


def _create_cropped_mask_tensor(image_path: str, image_index: str, masks_df: pd.DataFrame, image_size: int = 224) -> torch.Tensor:
    """
    Helper to create a tensor for the cropped mask. It gets the combined mask, computes the bounding box, crops the mask, resizes it, and converts it to a tensor.
    """
    _, cropped_mask_np = _get_cropped_arrays(image_path, image_index, masks_df)


    mask_resized = Image.fromarray(cropped_mask_np * 255).resize((image_size, image_size), Image.NEAREST)
    mask_tensor = to_tensor(mask_resized) # Shape: [1, 224, 224]

    return mask_tensor

def _convert_to_tensor_and_resize(image_path: str, image_size: int) -> torch.Tensor:
    """Helper function to load an image, convert it to a tensor, and resize it"""
    image = Image.open(image_path).convert("RGB")
    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToTensor()
        ]
    )
    return transform(image)

def _precompute_and_save_tensor(images_dir: str, output_dir: str, df: pd.DataFrame, image_column: str, image_size: int) -> str:
    """Helper function to precompute tensors for a given dataframe and save them to disk"""

    for image_index in tqdm(df[image_column], desc=f"Precomputing images"):
        src = os.path.join(images_dir, image_index)
        image_tensor = _convert_to_tensor_and_resize(src, image_size)
        dst = os.path.join(output_dir, image_index.replace(".png", ".pt"))
        torch.save(image_tensor, dst)

    return output_dir

def _precompute_and_save_tensor_cropped(images_dir: str, output_dir: str, df: pd.DataFrame, image_column: str, masks_df: pd.DataFrame, image_size: int) -> str:
    """Helper function to precompute tensors for cropped images and save them to disk"""

    for image_index in tqdm(df[image_column], desc=f"Precomputing cropped images"):
        src = os.path.join(images_dir, image_index)
        cropped_image_tensor = _create_cropped_image_tensor(src, image_index, masks_df, image_size)
        dst = os.path.join(output_dir, image_index.replace(".png", "_cropped.pt"))
        torch.save(cropped_image_tensor, dst)

    return output_dir

def _precompute_and_save_mask_cropped(images_dir: str, output_dir: str, df: pd.DataFrame, image_column: str, masks_df: pd.DataFrame, image_size: int) -> str:
    """Helper function to precompute tensors for cropped masks and save them to disk"""

    for image_index in tqdm(df[image_column], desc=f"Precomputing cropped masks"):
        src = os.path.join(images_dir, image_index)
        cropped_mask_tensor = _create_cropped_mask_tensor(src, image_index, masks_df, image_size)
        dst = os.path.join(output_dir, image_index.replace(".png", "_cropped_mask.pt"))
        torch.save(cropped_mask_tensor, dst)

    return output_dir

def _precompute_tensors_for_dataframe(
    images_dir: str,
    output_dir_full: str,
    output_dir_crop: str,
    output_dir_mask: str,
    df: pd.DataFrame,
    image_column: str,
    masks_df: pd.DataFrame,
    image_size: int,
    option: str = "full",
    compute: bool = False
) -> None:
    """Precomputes tensors for both full and cropped images and saves them to disk"""
    if not compute:
        print(f"Skipping precomputation for {option} images as compute_tensors is false.")
        return
    if option == "full":
        os.makedirs(output_dir_full, exist_ok=True)
        _precompute_and_save_tensor(images_dir, output_dir_full, df, image_column, image_size)
    elif option == "cropped":
        os.makedirs(output_dir_crop, exist_ok=True)
        _precompute_and_save_tensor_cropped(images_dir, output_dir_crop, df, image_column, masks_df, image_size)
    elif option == "cropped_with_mask":
        os.makedirs(output_dir_crop, exist_ok=True)
        os.makedirs(output_dir_mask, exist_ok=True)
        _precompute_and_save_tensor_cropped(images_dir, output_dir_crop, df, image_column, masks_df, image_size)
        _precompute_and_save_mask_cropped(images_dir, output_dir_mask, df, image_column, masks_df, image_size)
    else:
        raise ValueError(f"Invalid option: {option}. Must be one of 'full', 'cropped', or 'cropped_with_mask'.")


def build_tensor_dataset(
    df: pd.DataFrame,
    masks_df: pd.DataFrame,
    images_dir: str,
    out_dir_full: str,
    out_dir_crop: str,
    out_dir_mask: str,
    image_size: int = 224,
    image_type: str = "full",
    compute_tensors: bool = False
) -> pd.DataFrame:
    """Builds a dataset by loading precomputed tensors for images and masks based on the provided dataframe"""


    _precompute_tensors_for_dataframe(
        images_dir=images_dir,
        output_dir_full=out_dir_full,
        output_dir_crop=out_dir_crop,
        output_dir_mask=out_dir_mask,
        df=df,
        image_column="image_index",
        masks_df=masks_df,
        image_size=image_size,
        option=image_type,
        compute=compute_tensors
    )


    tensor_df = df.copy()

    if image_type == "full":
        tensor_df["image_tensor_path"] = tensor_df["image_index"].apply(
            lambda x: os.path.join(out_dir_full, f"{x.replace('.png', '.pt')}")
        )
    elif image_type == "cropped":
        tensor_df["image_tensor_path"] = tensor_df["image_index"].apply(
            lambda x: os.path.join(out_dir_crop, f"{x.replace('.png', '_cropped.pt')}")
        )
    elif image_type == "cropped_with_mask":
        tensor_df["image_tensor_path"] = tensor_df["image_index"].apply(
            lambda x: os.path.join(out_dir_crop, f"{x.replace('.png', '_cropped.pt')}")
        )
        tensor_df["mask_tensor_path"] = tensor_df["image_index"].apply(
            lambda x: os.path.join(out_dir_mask, f"{x.replace('.png', '_cropped_mask.pt')}")
        )

    return tensor_df
