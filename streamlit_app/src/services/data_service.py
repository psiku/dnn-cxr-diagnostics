import pandas as pd
import streamlit as st

from src.config import TEST_CSV, TRAIN_VAL_CSV


@st.cache_data
def load_data():
    test = pd.read_csv(TEST_CSV)
    test["split"] = "test"

    train_val = pd.read_csv(TRAIN_VAL_CSV)
    train_val["split"] = "train_val"

    return pd.concat([test, train_val])


def get_row_by_image_index(df, image_index):
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    matched = df.loc[df["image_index"] == image_index]
    if matched.empty:
        raise ValueError(f"No row found for image_index={image_index}")

    return matched.iloc[0]
