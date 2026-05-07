from io import BytesIO

import streamlit as st
from PIL import Image

from src.constants import DISEASES
from src.services.data_service import get_row_by_image_index, load_data
from src.services.inference_service import (
    build_predictions_df,
    generate_weighted_cam,
    get_real_targets,
    predict,
)
from src.services.model_service import (
    _load_thresholds,
    find_thresholds_for_checkpoint,
    list_saved_models,
    load_model,
)
from src.utils.preprocessing import prepare_image_tensor
from src.utils.visualization import generate_heatmap_overlay


st.set_page_config(page_title="Triage", layout="wide")
st.title("Triage")


available_models = list_saved_models()
if not available_models:
    st.error("No saved models found in `cxr-training-pipeline/data/06_models`.")
    st.stop()


# Settings: model + threshold source

st.header("Triage Settings")

selected_model = st.selectbox(
    "Model checkpoint",
    options=available_models,
    format_func=lambda p: f"{p.parent.parent.parent.name}-{p.parent.parent.name}-{p.stem}",
)

thresholds_path = find_thresholds_for_checkpoint(selected_model)
threshold_values = _load_thresholds(thresholds_path)

with st.expander("Loaded thresholds", expanded=False):
    if thresholds_path is not None:
        st.caption(f"Source: `{thresholds_path}`")
    else:
        st.caption("No matching `best_thresholds.json` found - using 0.5 for every disease.")
    st.json({d: threshold_values.get(d, 0.5) for d in DISEASES})


# Disease selection (filters the predictions table)

if "all_diseases_selected" not in st.session_state:
    st.session_state.all_diseases_selected = False

if "selected_diseases" not in st.session_state:
    st.session_state.selected_diseases = []


def toggle_all_diseases():
    st.session_state.all_diseases_selected = not st.session_state.all_diseases_selected

    if st.session_state.all_diseases_selected:
        st.session_state.selected_diseases = DISEASES[:]
    else:
        st.session_state.selected_diseases = []


st.button("Select all / Unselect all", on_click=toggle_all_diseases)

selected_diseases = st.pills(
    "Diseases to classify",
    options=DISEASES,
    selection_mode="multi",
    key="selected_diseases",
)

if not selected_diseases:
    st.warning("Select at least one disease.")
    st.stop()



# Image upload + inference
uploaded = st.file_uploader("Drop chest X-ray image", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload an image to run triage.")
    st.stop()

uploaded_bytes = uploaded.getvalue()
img_pil = Image.open(BytesIO(uploaded_bytes)).convert("RGB")

st.session_state["labeling_image_bytes"] = uploaded_bytes
st.session_state["labeling_image_name"] = uploaded.name

df = load_data()
image_names = set(df["image_index"].astype(str))
uploaded_name = str(uploaded.name)
has_ground_truth = uploaded_name in image_names

if has_ground_truth:
    true_row = get_row_by_image_index(df, uploaded_name)
    real_targets = get_real_targets(true_row)
else:
    real_targets = []

try:
    model = load_model(selected_model)
    channels = 1 if getattr(model, "grayscale", False) else 3
    image_tensor = prepare_image_tensor(img_pil, channels=channels)
    probs = predict(model, image_tensor)
    preds_df = build_predictions_df(probs, real_targets, threshold=threshold_values)
except Exception as e:
    st.error(f"Inference failed: {e}")
    st.stop()

preds_filtered = (
    preds_df[preds_df["Disease"].isin(selected_diseases)]
    .sort_values("Probability", ascending=False)
    .reset_index(drop=True)
)


if st.button("Add Bounding Box"):
    st.switch_page("pages/labeling.py")


# Original image + probability-weighted Grad-CAM
col_original, col_gradcam = st.columns(2)
with col_original:
    st.subheader("Original image")
    st.image(img_pil, width=800)
with col_gradcam:
    st.subheader("Grad-CAM")
    with st.spinner("Generating probability-weighted Grad-CAM..."):
        combined_cam = generate_weighted_cam(model, image_tensor, probs)
        _, overlay = generate_heatmap_overlay(
            img_pil,
            combined_cam,
            target_size=(800, 800),
        )
    st.image(
        overlay,
        caption="Grad-CAM (probability-weighted across all diseases)",
        width=800,
    )


# Summary table
if has_ground_truth:
    gt_label = ", ".join(real_targets) if real_targets else "(no positive labels)"
    st.markdown("**Ground truth labels:** " + gt_label)

st.subheader("Predicted probabilities")
table_cols = ["Disease", "Probability", "Predicted Target"]
if has_ground_truth:
    table_cols.append("Is Target")
st.dataframe(preds_filtered[table_cols], use_container_width=True)
