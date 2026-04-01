import streamlit as st
import pandas as pd

from src.config import REPORTING_DIR
from src.services.metrics_service import _calculate_metrics, _style_best_micro_macro

st.set_page_config(page_title="Model Statistics", layout="wide")
st.title("Model Statistics")


def main():
    reporting = REPORTING_DIR
    val_proba_path = reporting / "val_pred_proba.npy"
    val_targets_path = reporting / "val_targets.npy"

    test_proba_path = reporting / "test_pred_proba.npy"
    test_targets_path = reporting / "test_targets.npy"

    threshold_path = reporting / "best_thresholds.json"

    val_summary_metrics, val_per_class_metrics = _calculate_metrics(
        val_proba_path, val_targets_path, threshold_path
    )

    val_summary_metrics_default, val_per_class_metrics_default = _calculate_metrics(
        val_proba_path, val_targets_path, threshold_path=None
    )

    test_summary_metrics, test_per_class_metrics = _calculate_metrics(
        test_proba_path, test_targets_path, threshold_path
    )

    test_summary_metrics_default, test_per_class_metrics_default = _calculate_metrics(
        test_proba_path, test_targets_path, threshold_path=None
    )

    st.subheader("Validation metrics (default + tuned)")
    combined_val_summary = pd.concat(
        [
            val_summary_metrics_default.rename(index=lambda x: f"{x}_default"),
            val_summary_metrics.rename(index=lambda x: f"{x}_tuned"),
        ],
        axis=0,
    )
    st.dataframe(_style_best_micro_macro(combined_val_summary), use_container_width=True)

    default_col, tuned_col = st.columns(2)
    with default_col:
        st.subheader("Validation per class metrics (default)")
        st.dataframe(val_per_class_metrics_default, use_container_width=True)
    with tuned_col:
        st.subheader("Validation per class metrics (tuned)")
        st.dataframe(val_per_class_metrics, use_container_width=True)

    st.subheader("Test metrics (default + tuned)")
    combined_test_summary = pd.concat(
        [
            test_summary_metrics_default.rename(index=lambda x: f"{x}_default"),
            test_summary_metrics.rename(index=lambda x: f"{x}_tuned"),
        ],
        axis=0,
    )
    st.dataframe(_style_best_micro_macro(combined_test_summary), use_container_width=True)

    default_col, tuned_col = st.columns(2)
    with default_col:
        st.subheader("Test per class metrics (default)")
        st.dataframe(test_per_class_metrics_default, use_container_width=True)
    with tuned_col:
        st.subheader("Test per class metrics (tuned)")
        st.dataframe(test_per_class_metrics, use_container_width=True)


if __name__ == "__main__":
    main()
