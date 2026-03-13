import streamlit as st

def select_split():
    return st.sidebar.radio("Split", ["train_val", "test"], horizontal=True)

def select_image(df):
    if df is None or df.empty:
        raise ValueError("Selected split is empty.")

    if "image_index" not in df.columns:
        raise KeyError("Missing required column: image_index")

    options = df["image_index"].dropna().tolist()
    if len(options) == 0:
        raise ValueError("No image_index values found in selected split.")

    return st.sidebar.selectbox("Image index", options=options)

def render_sidebar_metadata(row):
    st.sidebar.markdown(f"**Selected:** `{row['image_index']}`")
    st.sidebar.markdown(
        f"**Patient Details:** {row['patient_age']}yo, {row['patient_sex']}, View: {row['view_position']}"
    )