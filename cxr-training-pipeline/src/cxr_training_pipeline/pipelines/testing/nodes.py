# testing/nodes.py
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import pandas as pd


def _apply_thresholds(
    y_proba: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    return (y_proba > thresholds.reshape(1, -1)).astype(int)


def _compute_global_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
    }

    # These may fail if a class has only one label in y_true; guard them.
    try:
        metrics["auroc_macro"] = roc_auc_score(y_true, y_proba, average="macro")
    except ValueError:
        metrics["auroc_macro"] = np.nan

    try:
        metrics["ap_macro"] = average_precision_score(y_true, y_proba, average="macro")
    except ValueError:
        metrics["ap_macro"] = np.nan

    return metrics


def _compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    pathology_list: list[str],
) -> pd.DataFrame:
    rows = []

    for i, class_name in enumerate(pathology_list):
        yt = y_true[:, i].astype(int)
        yp = y_pred[:, i].astype(int)
        yp_proba = y_proba[:, i]

        row = {
            "class_name": class_name,
            "f1": f1_score(yt, yp, zero_division=0),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
        }

        try:
            row["auroc"] = roc_auc_score(yt, yp_proba)
        except ValueError:
            row["auroc"] = np.nan

        try:
            row["ap"] = average_precision_score(yt, yp_proba)
        except ValueError:
            row["ap"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_default_thresholds_node(
    test_targets: np.ndarray,
    test_pred_proba: np.ndarray,
    pathology_list: list[str],
    default_threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    default_thresholds = np.full(len(pathology_list), default_threshold, dtype=np.float32)
    y_pred_default = _apply_thresholds(test_pred_proba, default_thresholds)

    test_metrics_default = _compute_global_metrics(
        test_targets, y_pred_default, test_pred_proba
    )
    per_class_default = _compute_per_class_metrics(
        test_targets, y_pred_default, test_pred_proba, pathology_list
    ).rename(
        columns={
            "f1": "f1_default_0_5",
            "precision": "precision_default_0_5",
            "recall": "recall_default_0_5",
            "auroc": "auroc_default_0_5",
            "ap": "ap_default_0_5",
        }
    )

    return test_metrics_default, per_class_default


def evaluate_tuned_thresholds_node(
    test_targets: np.ndarray,
    test_pred_proba: np.ndarray,
    best_thresholds: dict[str, float],
    pathology_list: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    tuned_thresholds = np.array(
        [best_thresholds[class_name] for class_name in pathology_list],
        dtype=np.float32,
    )
    y_pred_tuned = _apply_thresholds(test_pred_proba, tuned_thresholds)

    test_metrics_tuned = _compute_global_metrics(
        test_targets, y_pred_tuned, test_pred_proba
    )
    per_class_tuned = _compute_per_class_metrics(
        test_targets, y_pred_tuned, test_pred_proba, pathology_list
    ).rename(
        columns={
            "f1": "f1_tuned",
            "precision": "precision_tuned",
            "recall": "recall_tuned",
            "auroc": "auroc_tuned",
            "ap": "ap_tuned",
        }
    )

    per_class_tuned["threshold_tuned"] = tuned_thresholds

    return test_metrics_tuned, per_class_tuned


def compare_metrics_node(
    test_metrics_default: dict[str, float],
    test_metrics_tuned: dict[str, float],
    per_class_default: pd.DataFrame,
    per_class_tuned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    comparison_df = pd.DataFrame(
        {
            "metric": list(test_metrics_default.keys()),
            "default_0_5": list(test_metrics_default.values()),
            "tuned": [test_metrics_tuned[k] for k in test_metrics_default.keys()],
        }
    )
    comparison_df["delta"] = comparison_df["tuned"] - comparison_df["default_0_5"]

    per_class_comparison = per_class_default.merge(
        per_class_tuned, on="class_name", how="inner"
    )

    per_class_comparison["f1_delta"] = (
        per_class_comparison["f1_tuned"] - per_class_comparison["f1_default_0_5"]
    )
    per_class_comparison["precision_delta"] = (
        per_class_comparison["precision_tuned"]
        - per_class_comparison["precision_default_0_5"]
    )
    per_class_comparison["recall_delta"] = (
        per_class_comparison["recall_tuned"]
        - per_class_comparison["recall_default_0_5"]
    )

    return comparison_df, per_class_comparison
