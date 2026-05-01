import numpy as np
import pandas as pd
from PIL import Image


def rle_to_mask(rle_str: str, height: int, width: int) -> np.ndarray:
    """Convert RLE string to binary mask.
    Args:
        rle_str: RLE string
        height: Height of the image
        width: Width of the image
    Returns:
        Binary mask as numpy array
    """
    if not isinstance(rle_str, str) or not rle_str.strip():
        return np.zeros((height, width), dtype=np.uint8)

    tokens = [int(x) for x in rle_str.split()]
    starts = np.array(tokens[0::2]) - 1  # RLE is 1-based
    lengths = np.array(tokens[1::2])
    ends = starts + lengths

    flat = np.zeros(height * width, dtype=np.uint8)
    for start, end in zip(starts, ends):
        flat[start:end] = 1

    return flat.reshape((height, width))


def get_combined_mask_by_image_index(
    image_index: str,
    segmentation_masks_df: pd.DataFrame,
    height: int,
    width: int,
    mask_columns: list[str] = ["Left Lung", "Right Lung", "Heart"],
) -> np.ndarray:
    """Get combined binary mask for all organs for a given image index from the segmentation masks DataFrame."""
    rows = segmentation_masks_df.loc[segmentation_masks_df["Image Index"] == image_index]

    if rows.empty:
        return np.zeros((height, width), dtype=np.uint8)

    row = rows.iloc[0]
    organ_masks = [rle_to_mask(row.get(col, ""), height, width) for col in mask_columns]
    return np.maximum.reduce(organ_masks)


def get_bbox_coordinates(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Calculate bounding box coordinates from binary mask.
    Args:
        mask: Binary mask as numpy array
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) coordinates. Returns None if mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1


def crop_to_bbox(img: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """Crop image to bounding box coordinates.

    Args:
        img: Input image as numpy array
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)

    Returns:
        Cropped image. If bbox is None, returns the original image unchanged.
    """
    if bbox is None:
        return img

    x_min, y_min, x_max, y_max = bbox
    return img[y_min:y_max, x_min:x_max]


def expand_bbox(
    bbox: tuple[int, int, int, int],
    img_shape: tuple[int, int],
    margin_ratio: float = 0.08,
    min_bbox_size: int = 64,
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    height, width = img_shape

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    margin_x = int(round(bbox_w * margin_ratio))
    margin_y = int(round(bbox_h * margin_ratio))

    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(width, x_max + margin_x)
    y_max = min(height, y_max + margin_y)

    if (x_max - x_min) < min_bbox_size:
        to_add = min_bbox_size - (x_max - x_min)
        left = to_add // 2
        right = to_add - left
        x_min = max(0, x_min - left)
        x_max = min(width, x_max + right)

    if (y_max - y_min) < min_bbox_size:
        to_add = min_bbox_size - (y_max - y_min)
        top = to_add // 2
        bottom = to_add - top
        y_min = max(0, y_min - top)
        y_max = min(height, y_max + bottom)

    return x_min, y_min, x_max, y_max


def crop_resize_with_mask(
    img_np: np.ndarray,
    image_index: str,
    segmentation_masks_df: pd.DataFrame,
    output_size: int,
    margin_ratio: float = 0.08,
    min_bbox_size: int = 64,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Crop and resize image based on segmentation mask.

    Args:
        img_np: Input image as numpy array
        image_index: Index to find the segmentation mask
        segmentation_masks_df: DataFrame containing segmentation masks
        output_size: Size of output image (output_size x output_size)
        margin_ratio: Ratio to expand bbox as fraction of bbox size
        min_bbox_size: Minimum size for bounding box

    Returns:
        Tuple of (resized_image, bbox_coordinates) where bbox_coordinates
        is None if no mask found for the image
    """
    combined_mask = get_combined_mask_by_image_index(
        image_index=image_index,
        segmentation_masks_df=segmentation_masks_df,
        height=img_np.shape[0],
        width=img_np.shape[1],
    )
    bbox = get_bbox_coordinates(combined_mask)

    if bbox is not None:
        bbox = expand_bbox(
            bbox=bbox,
            img_shape=img_np.shape,
            margin_ratio=margin_ratio,
            min_bbox_size=min_bbox_size,
        )
        x_min, y_min, x_max, y_max = bbox
        img_np = img_np[y_min:y_max, x_min:x_max]

    resized = Image.fromarray(img_np).resize((output_size, output_size), Image.BILINEAR)
    return np.array(resized, dtype=np.uint8), bbox

