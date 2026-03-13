import streamlit as st

from src.services.data_service import load_data, get_row_by_image_index, load_pil_image
from src.services.model_service import load_model
from src.services.inference_service import (
    get_real_targets,
    predict,
    build_predictions_df,
    generate_cam,
)
from src.utils.preprocessing import prepare_image_tensor
from src.utils.visualization import generate_heatmap_overlay
from src.ui.sidebar import select_image, render_sidebar_metadata, select_split
from src.ui.predictions import render_predictions_table, select_target_class
from src.ui.image_panel import render_image_comparison, render_ground_truth
from src.constants import DISEASES


def main():
    st.set_page_config(page_title="CXR Model Focus Analysis", layout="wide")
    st.title("DNN CXR Diagnostics - Model Analysis")

    # Load data
    try:
        df_dict = load_data()
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        st.stop()

    if df_dict["train_val"].empty and df_dict["test"].empty:
        st.error("Loaded dataframes are empty.")
        st.stop()


    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Sidebar
    st.sidebar.header("Select Image")

    try:
        split = select_split()
        if split not in df_dict:
            raise KeyError(f"Unknown split: {split}")

        df = df_dict[split]
        if df.empty:
            raise ValueError(f"Split '{split}' is empty.")

        image_index = select_image(df)
        row = get_row_by_image_index(df, image_index)
        render_sidebar_metadata(row)
    except Exception as e:
        st.error(f"Error selecting image: {e}")
        st.stop()

    # Load image
    try:
        img_pil = load_pil_image(row["image_index"])
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Preprocess
    try:
        image_tensor = prepare_image_tensor(img_pil)
    except Exception as e:
        st.error(f"Error preparing image tensor: {e}")
        st.stop()

    # Inference
    try:
        real_targets = get_real_targets(row)
        probs = predict(model, image_tensor)
        preds_df = build_predictions_df(probs, real_targets)
    except Exception as e:
        st.error(f"Error during inference: {e}")
        st.stop()

    # Predictions table
    render_predictions_table(preds_df)

    # Target class selection
    try:
        target_class = select_target_class(preds_df)
        class_idx = DISEASES.index(target_class)
    except Exception as e:
        st.error(f"Error selecting target class: {e}")
        st.stop()

    # CAM
    try:
        with st.spinner("Generating CAM..."):
            cam_map = generate_cam(model, image_tensor, class_idx)
    except Exception as e:
        st.error(f"Error generating CAM: {e}")
        st.stop()

    # Overlay
    try:
        img_resized, overlay = generate_heatmap_overlay(img_pil, cam_map)
    except Exception as e:
        st.error(f"Error generating heatmap overlay: {e}")
        st.stop()

    # Render results
    render_image_comparison(img_resized, overlay, target_class)
    render_ground_truth(real_targets)


if __name__ == "__main__":
    main()