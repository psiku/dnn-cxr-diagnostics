import streamlit as st
from PIL import Image
import tempfile
from io import BytesIO
from streamlit_image_annotation import detection
from src.constants import DISEASES
import json

st.title("Labeling")

if "bboxes" not in st.session_state:
    st.session_state.bboxes = []

if "labels" not in st.session_state:
    st.session_state.labels = []

image = None
image_name = None

if "labeling_image_bytes" in st.session_state:
    image = Image.open(BytesIO(st.session_state["labeling_image_bytes"])).convert("RGB")
    image_name = st.session_state.get("labeling_image_name", "image.png")

if image is None:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name

if image is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    result = detection(
        image_path=image_path,
        label_list=DISEASES,
        bboxes=st.session_state.bboxes,
        labels=st.session_state.labels,
        height=800,
        width=800,
        key="detection",
    )

    result = result or []

    st.session_state.bboxes = [item["bbox"] for item in result]
    st.session_state.labels = [item["label_id"] for item in result]

    annotations_data = {
        "image_name": image_name,
        "annotations": result
    }

    st.write(annotations_data)

    if result:
        st.download_button(
            "Download annotations",
            data=json.dumps(annotations_data, indent=2),
            file_name="annotations.json",
            mime="application/json",
        )

    # optional: download image too
    if "labeling_image_bytes" in st.session_state:
        st.download_button(
            "Download image",
            data=st.session_state["labeling_image_bytes"],
            file_name=image_name,
            mime="image/png",
        )