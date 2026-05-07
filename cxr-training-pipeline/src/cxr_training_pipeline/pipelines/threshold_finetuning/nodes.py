from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

_DEFAULT_OBJECTIVE: dict[str, Any] = {"name": "fbeta", "beta": 1.0}
_VALID_OBJECTIVES = {"fbeta", "youden_j", "recall_at_precision", "precision_at_recall"}


def _fbeta_curve(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.where(denom > 0, (1 + b2) * precision * recall / np.maximum(denom, 1e-12), 0.0)


def _youden_j_curve(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.zeros_like(thresholds, dtype=float)
    # For each threshold t, TP = #(positives with proba >= t), FP = #(negatives with proba >= t).
    pos_scores = np.sort(y_proba[y_true == 1])
    neg_scores = np.sort(y_proba[y_true == 0])
    tp = len(pos_scores) - np.searchsorted(pos_scores, thresholds, side="left")
    fp = len(neg_scores) - np.searchsorted(neg_scores, thresholds, side="left")
    return tp / n_pos - fp / n_neg


def _resolve_objective(
    base_objective: dict[str, Any],
    overrides: dict[str, dict[str, Any]],
    class_name: str,
) -> dict[str, Any]:
    obj = dict(base_objective)
    if class_name in overrides:
        obj.update(overrides[class_name])
    name = obj.get("name", "fbeta")
    if name not in _VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown threshold objective '{name}' for class '{class_name}'. "
            f"Valid options: {sorted(_VALID_OBJECTIVES)}"
        )
    return obj


def _select_best(
    thresholds: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    fbeta_score: np.ndarray,
    youden_j: np.ndarray,
    objective: dict[str, Any],
    default_threshold: float,
) -> tuple[float, float, str]:
    """Pick the threshold that satisfies the objective.

    Returns (best_threshold, score_at_best, status) where status is one of
    {"selected", "fallback_no_candidate"}.
    """
    name = objective["name"]

    if name == "fbeta":
        idx = int(np.argmax(fbeta_score))
        return float(thresholds[idx]), float(fbeta_score[idx]), "selected"

    if name == "youden_j":
        idx = int(np.argmax(youden_j))
        return float(thresholds[idx]), float(youden_j[idx]), "selected"

    if name == "recall_at_precision":
        min_p = float(objective.get("min_precision", 0.5))
        valid = precision >= min_p
        if not valid.any():
            return float(default_threshold), float("nan"), "fallback_no_candidate"
        masked_recall = np.where(valid, recall, -np.inf)
        idx = int(np.argmax(masked_recall))
        return float(thresholds[idx]), float(recall[idx]), "selected"

    if name == "precision_at_recall":
        min_r = float(objective.get("min_recall", 0.5))
        valid = recall >= min_r
        if not valid.any():
            return float(default_threshold), float("nan"), "fallback_no_candidate"
        masked_precision = np.where(valid, precision, -np.inf)
        # Among ties on precision prefer the higher threshold (more conservative).
        idx = int(len(masked_precision) - 1 - np.argmax(masked_precision[::-1]))
        return float(thresholds[idx]), float(precision[idx]), "selected"

    raise ValueError(f"Unhandled objective '{name}'")


def tune_thresholds(
    val_targets: np.ndarray,
    val_pred_proba: np.ndarray,
    pathology_list: list[str],
    threshold_params: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Pick a per-class decision threshold on the validation set.

    Candidate thresholds come from `sklearn.metrics.precision_recall_curve`,
    which yields the minimal set of operating points (one per unique decision
    score). The chosen threshold is the one that maximizes the configured
    objective (or satisfies the configured constraint).

    Parameters expected under `threshold_params`:
        default_threshold: float
            Fallback threshold for classes with zero positives in val_targets
            or when a constrained objective has no feasible candidate.
        objective: dict
            Global objective spec, e.g. {"name": "fbeta", "beta": 1.0}.
            Supported names: fbeta, youden_j, recall_at_precision,
            precision_at_recall.
        per_class_overrides: dict[str, dict]
            Optional class -> objective spec overrides.
    """
    default_threshold = float(threshold_params.get("default_threshold", 0.5))
    base_objective = threshold_params.get("objective", _DEFAULT_OBJECTIVE)
    per_class_overrides = threshold_params.get("per_class_overrides", {}) or {}

    best_thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for class_idx, class_name in tqdm(
        enumerate(pathology_list),
        total=len(pathology_list),
        desc="Tuning thresholds per class",
    ):
        y_true = val_targets[:, class_idx].astype(int)
        y_proba = val_pred_proba[:, class_idx]
        n_pos = int(y_true.sum())

        objective = _resolve_objective(base_objective, per_class_overrides, class_name)

        if n_pos == 0:
            best_thresholds[class_name] = default_threshold
            rows.append(
                {
                    "class_name": class_name,
                    "threshold": float(default_threshold),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "fbeta": float("nan"),
                    "youden_j": float("nan"),
                    "chosen": True,
                    "objective": objective["name"],
                    "n_pos": 0,
                    "status": "fallback_no_positives",
                }
            )
            continue

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # precision_recall_curve appends a sentinel (P=1, R=0) without a threshold.
        precision = precision[:-1]
        recall = recall[:-1]

        f1 = _fbeta_curve(precision, recall, beta=1.0)
        beta = float(objective.get("beta", 1.0)) if objective["name"] == "fbeta" else 1.0
        fbeta = f1 if beta == 1.0 else _fbeta_curve(precision, recall, beta=beta)
        youden = _youden_j_curve(y_true, y_proba, thresholds)

        best_thr, _score, status = _select_best(
            thresholds=thresholds,
            precision=precision,
            recall=recall,
            fbeta_score=fbeta,
            youden_j=youden,
            objective=objective,
            default_threshold=default_threshold,
        )
        best_thresholds[class_name] = best_thr

        chosen_mask = np.isclose(thresholds, best_thr) if status == "selected" else np.zeros_like(thresholds, dtype=bool)
        for i, thr in enumerate(thresholds):
            rows.append(
                {
                    "class_name": class_name,
                    "threshold": float(thr),
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "fbeta": float(fbeta[i]),
                    "youden_j": float(youden[i]),
                    "chosen": bool(chosen_mask[i]),
                    "objective": objective["name"],
                    "n_pos": n_pos,
                    "status": status,
                }
            )

        if status != "selected":
            # Append a synthetic row for the fallback threshold so it's discoverable.
            rows.append(
                {
                    "class_name": class_name,
                    "threshold": float(best_thr),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "fbeta": float("nan"),
                    "youden_j": float("nan"),
                    "chosen": True,
                    "objective": objective["name"],
                    "n_pos": n_pos,
                    "status": status,
                }
            )

    results_df = pd.DataFrame(rows)
    return best_thresholds, results_df
