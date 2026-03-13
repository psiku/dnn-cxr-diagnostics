import streamlit as st


def render_image_comparison(original_img, overlay_img, target_class):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original X-Ray")
        st.image(original_img, use_container_width=True)

    with col2:
        st.markdown(f"### Model CAM Focus ({target_class})")
        st.image(overlay_img, use_container_width=True)


def render_ground_truth(real_targets):
    st.markdown("### Ground Truth")
    if real_targets:
        for target in real_targets:
            st.success(target)
    else:
        st.info("No Findings")