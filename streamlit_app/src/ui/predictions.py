import streamlit as st
from src.constants import DISEASES


def render_predictions_table(preds_df):
    st.subheader("Model Predictions")
    st.dataframe(
        preds_df.style.highlight_max(axis=0, subset=["Probability"], color="lightgreen")
    )


def select_target_class(preds_df):
    default_class = preds_df.iloc[0]["Disease"]
    return st.selectbox(
        "Select Class to Visualize with Grad-CAM:",
        DISEASES,
        index=DISEASES.index(default_class),
    )