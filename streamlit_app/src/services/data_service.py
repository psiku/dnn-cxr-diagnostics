from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import TEST_CSV, IMAGES_DIR, TRAIN_VAL_CSV


@st.cache_data
def load_data():
    return {
        "test": pd.read_csv(TEST_CSV),
        "train_val": pd.read_csv(TRAIN_VAL_CSV)
    }


def get_row_by_image_index(df, image_index):
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    matched = df.loc[df["image_index"] == image_index]
    if matched.empty:
        raise ValueError(f"No row found for image_index={image_index}")

    return matched.iloc[0]  # always return a Series, not DataFrame


def get_image_path(image_index: str) -> Path:
    return IMAGES_DIR / image_index


def load_pil_image(image_index: str):
    img_path = get_image_path(image_index)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {img_path}")
    return Image.open(img_path).convert("RGB") # fun fact: this will show an image in grayscale