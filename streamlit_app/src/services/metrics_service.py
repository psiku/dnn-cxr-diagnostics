import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import DISEASES
from src.services.model_service import _load_thresholds


def get_project_name(model_output_dir: Path) -> str:
    return [p.name for p in model_output_dir.iterdir() if p.is_dir()]


def _load_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Metric file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Unsupported file format. Use .json, .csv, .npy, or .parquet.")


def _calculate_summary_metrics(targets: np.ndarray, preds: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    summary_df = pd.DataFrame(
        {
            "f1": [
                f1_score(targets, preds, average="micro", zero_division=0),
                f1_score(targets, preds, average="macro", zero_division=0),
            ],
            "precision": [
                precision_score(targets, preds, average="micro", zero_division=0),
                precision_score(targets, preds, average="macro", zero_division=0),
            ],
            "recall": [
                recall_score(targets, preds, average="micro", zero_division=0),
                recall_score(targets, preds, average="macro", zero_division=0),
            ],
            "auroc": [
                roc_auc_score(targets, proba, average="micro"),
                roc_auc_score(targets, proba, average="macro"),
            ],
            "ap": [
                average_precision_score(targets, proba, average="micro"),
                average_precision_score(targets, proba, average="macro"),
            ],
        },
        index=["micro", "macro"],
    )
    return summary_df


def _calculate_per_class_metrics(targets: np.ndarray, preds: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    per_class_df = pd.DataFrame(
        {
            "f1": f1_score(targets, preds, average=None, zero_division=0),
            "precision": precision_score(targets, preds, average=None, zero_division=0),
            "recall": recall_score(targets, preds, average=None, zero_division=0),
            "auroc": roc_auc_score(targets, proba, average=None),
            "ap": average_precision_score(targets, proba, average=None),
        },
        index=DISEASES,
    )
    return per_class_df.T


def _calculate_metrics(
    proba_path: Path,
    targets_path: Path,
    threshold_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proba = _load_file(proba_path)
    targets = _load_file(targets_path)
    thresholds = _load_thresholds(threshold_path)
    threshold_vector = np.array([thresholds[d] for d in DISEASES], dtype=float)
    preds = (proba > threshold_vector).astype(int)

    summary_df = _calculate_summary_metrics(targets, preds, proba)
    per_class_df = _calculate_per_class_metrics(targets, preds, proba)
    return summary_df, per_class_df


def _style_best_micro_macro(df: pd.DataFrame):
    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in {"ap", "auroc"}
    ]
    styled = pd.DataFrame("", index=df.index, columns=df.columns)

    for col in numeric_cols:
        micro_rows = [idx for idx in df.index if str(idx).startswith("micro_")]
        macro_rows = [idx for idx in df.index if str(idx).startswith("macro_")]

        if micro_rows:
            best_micro = df.loc[micro_rows, col].max()
            for idx in micro_rows:
                if df.loc[idx, col] == best_micro:
                    styled.loc[idx, col] = "background-color: rgba(34, 197, 94, 0.28);"
        if macro_rows:
            best_macro = df.loc[macro_rows, col].max()
            for idx in macro_rows:
                if df.loc[idx, col] == best_macro:
                    styled.loc[idx, col] = "background-color: rgba(34, 197, 94, 0.28);"

    return df.style.format(precision=4).set_table_styles([]).apply(lambda _: styled, axis=None)
