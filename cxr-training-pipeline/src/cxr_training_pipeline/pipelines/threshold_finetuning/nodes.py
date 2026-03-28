from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0.0, average="macro")
    recall = recall_score(y_true, y_pred, zero_division=0.0, average="macro")
    f1 = f1_score(y_true, y_pred, zero_division=0.0, average="macro")
    return {"precision": precision, "recall": recall, "f1": f1}


def tune_thresholds(
    val_targets: np.ndarray,
    val_pred_proba: np.ndarray,
    pathology_list: list[str],
    threshold_params: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    start = threshold_params.get("start", 0.05)
    stop = threshold_params.get("stop", 0.95)
    step = threshold_params.get("step", 0.05)
    thresholds = np.arange(start, stop + 1e-9, step)

    best_thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for class_idx, class_name in tqdm(
        enumerate(pathology_list),
        total=len(pathology_list),
        desc="Tuning thresholds per class",
    ):
        y_true = val_targets[:, class_idx].astype(int)
        y_proba = val_pred_proba[:, class_idx]

        best_f1 = -1.0
        best_thr = 0.5

        for thr in tqdm(thresholds, desc=f"{class_name}", leave=False):
            y_pred = (y_proba > thr).astype(int)
            metrics = _binary_metrics(y_true, y_pred)

            row = {
                "class_name": class_name,
                "threshold": float(thr),
                **metrics,
            }
            rows.append(row)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_thr = float(thr)

        best_thresholds[class_name] = best_thr

    results_df = pd.DataFrame(rows)
    return best_thresholds, results_df
