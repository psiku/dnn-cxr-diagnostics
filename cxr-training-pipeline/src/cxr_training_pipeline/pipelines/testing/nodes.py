# testing/nodes.py
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from tqdm import tqdm

from cxr_training_pipeline.lightning_utils.factory import (
    ClassifierFactory,
    LightningModuleFactory,
)


def _apply_thresholds(
    y_proba: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    # Use >= so this matches the convention of sklearn's precision_recall_curve,
    # which is what the threshold tuning pipeline optimizes against.
    return (y_proba >= thresholds.reshape(1, -1)).astype(int)


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
            "accuracy": accuracy_score(yt, yp),
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

    for metric_name in ("f1", "precision", "recall", "accuracy", "auroc", "ap"):
        default_col = f"{metric_name}_default_0_5"
        tuned_col = f"{metric_name}_tuned"
        if default_col in per_class_comparison.columns and tuned_col in per_class_comparison.columns:
            per_class_comparison[f"{metric_name}_delta"] = (
                per_class_comparison[tuned_col] - per_class_comparison[default_col]
            )

    return comparison_df, per_class_comparison


# Localization IoU evaluation

# NIH BBox file labels don't perfectly match our pathology_list
_BBOX_LABEL_ALIASES: dict[str, str] = {
    "Infiltrate": "Infiltration",
}


def _scale_bbox_to_image(
    x: float, y: float, w: float, h: float, src: int, dst: int
) -> tuple[int, int, int, int]:
    s = dst / src
    x0 = int(round(x * s))
    y0 = int(round(y * s))
    x1 = int(round((x + w) * s))
    y1 = int(round((y + h) * s))
    x0 = max(0, min(dst, x0))
    y0 = max(0, min(dst, y0))
    x1 = max(0, min(dst, x1))
    y1 = max(0, min(dst, y1))
    return x0, y0, x1, y1


def _binarize_cam(cam: np.ndarray, mode: str, top_percent: float, value: float) -> np.ndarray:
    if mode == "value":
        return cam >= value
    if mode == "top_percent":
        if cam.size == 0:
            return cam.astype(bool)
        thr = float(np.quantile(cam, 1.0 - top_percent))
        return cam >= thr
    raise ValueError(f"Unknown threshold mode: {mode!r}")


def _iou_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    union = float(np.logical_or(pred_mask, gt_mask).sum())
    return inter / (union + 1e-12) if union > 0 else 0.0


def _resolve_class_index(
    finding_label: str, pathology_list: list[str]
) -> int | None:
    canonical = _BBOX_LABEL_ALIASES.get(finding_label, finding_label)
    if canonical in pathology_list:
        return pathology_list.index(canonical)
    return None


def _build_model_from_checkpoint(
    classifier_params: dict[str, Any],
    lit_module_params: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Rebuild the inner model + lightning wrapper from params and load weights.

    Mirrors how `data_science` constructs the model so the testing pipeline
    can run standalone (without needing `lit_model` to be in memory).
    """
    classifier_kwargs = classifier_params.get("kwargs", {})
    inner_model = ClassifierFactory.create(
        classifier_name=classifier_params["name"], **classifier_kwargs
    )

    lit_kwargs = lit_module_params.get("kwargs", {})
    lit_model = LightningModuleFactory.create(
        module_name=lit_module_params.get("name", "default_classifier"),
        model=inner_model,
        **lit_kwargs,
    )

    lit_model = lit_model.__class__.load_from_checkpoint(
        checkpoint_path, model=inner_model, map_location=device
    )
    return lit_model.model.to(device).eval()


@dataclass(frozen=True)
class _LocalizationConfig:
    image_size: int
    original_size: int
    threshold_mode: str
    top_percent: float
    threshold_value: float
    filter_to_test_set: bool
    device: torch.device
    use_mask_channel: bool = False
    mask_path_col: str | None = None
    apply_normalization: bool = True
    normalize_mean: tuple[float, ...] | None = None
    normalize_std: tuple[float, ...] | None = None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "_LocalizationConfig":
        device_param = params.get("device")
        device = (
            torch.device(device_param)
            if device_param
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return cls(
            image_size=int(params.get("image_size", 224)),
            original_size=int(params.get("original_size", 1024)),
            threshold_mode=str(params.get("threshold_mode", "top_percent")),
            top_percent=float(params.get("top_percent", 0.20)),
            threshold_value=float(params.get("threshold_value", 0.5)),
            filter_to_test_set=bool(params.get("filter_to_test_set", True)),
            device=device,
        )

    @classmethod
    def from_training_params(
        cls,
        localization_params: dict[str, Any],
        classifier_params: dict[str, Any],
        datamodule_kwargs: dict[str, Any],
        transform_params: dict[str, Any],
    ) -> "_LocalizationConfig":
        cfg = cls.from_params(localization_params)
        classifier_kwargs = classifier_params.get("kwargs", {})
        normalize_mean = transform_params.get("normalize_mean")
        normalize_std = transform_params.get("normalize_std")

        return cls(
            image_size=cfg.image_size,
            original_size=cfg.original_size,
            threshold_mode=cfg.threshold_mode,
            top_percent=cfg.top_percent,
            threshold_value=cfg.threshold_value,
            filter_to_test_set=cfg.filter_to_test_set,
            device=cfg.device,
            use_mask_channel=bool(classifier_kwargs.get("use_mask_channel", False)),
            mask_path_col=datamodule_kwargs.get("mask_path_col"),
            apply_normalization=bool(transform_params.get("apply_normalization", True)),
            normalize_mean=tuple(normalize_mean) if normalize_mean is not None else None,
            normalize_std=tuple(normalize_std) if normalize_std is not None else None,
        )


def _load_model_input_tensor(
    image_path: str,
    device: torch.device,
    mask_path: str | None = None,
    cfg: _LocalizationConfig | None = None,
) -> torch.Tensor:
    """Load precomputed tensors the same way as ``TensorCXRDataset`` at inference."""
    tensor = torch.load(image_path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict):
        tensor = tensor.get("image", tensor)

    if cfg is not None and cfg.use_mask_channel:
        if not mask_path:
            raise ValueError(
                "Model expects a mask channel (use_mask_channel=true) but no mask path was provided."
            )
        mask_tensor = torch.load(mask_path, map_location="cpu", weights_only=False)
        if isinstance(mask_tensor, dict):
            mask_tensor = mask_tensor.get("image", mask_tensor)
        tensor = torch.cat([tensor, mask_tensor], dim=0)

    tensor = tensor.unsqueeze(0).float().to(device)

    if cfg is not None and cfg.apply_normalization and cfg.normalize_mean and cfg.normalize_std:
        mean = torch.tensor(cfg.normalize_mean, device=device, dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(cfg.normalize_std, device=device, dtype=tensor.dtype).view(-1, 1, 1)
        tensor = (tensor - mean) / std

    return tensor


def _tensor_dataset_index(tensor_test_dataset: pd.DataFrame) -> pd.DataFrame:
    return tensor_test_dataset.set_index("image_index", drop=False)


def _compute_full_resolution_cam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_idx: int,
    image_size: int,
) -> np.ndarray:
    cam_small, _ = model.cam(image_tensor, class_idx=class_idx)
    cam_full = F.interpolate(
        cam_small.unsqueeze(1),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return cam_full.squeeze(1).squeeze(0).detach().cpu().numpy()


def _bbox_row_to_gt_mask(row: pd.Series, cfg: _LocalizationConfig) -> np.ndarray:
    gt_mask = np.zeros((cfg.image_size, cfg.image_size), dtype=bool)
    x0, y0, x1, y1 = _scale_bbox_to_image(
        float(row["x"]), float(row["y"]),
        float(row["w"]), float(row["h"]),
        src=cfg.original_size, dst=cfg.image_size,
    )
    gt_mask[y0:y1, x0:x1] = True
    return gt_mask


def _union_bbox_mask(rows: pd.DataFrame, cfg: _LocalizationConfig) -> np.ndarray:
    mask = np.zeros((cfg.image_size, cfg.image_size), dtype=bool)
    for _, row in rows.iterrows():
        mask |= _bbox_row_to_gt_mask(row, cfg)
    return mask


@torch.no_grad()
def _compute_weighted_cam_full(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    image_size: int,
) -> np.ndarray:
    """Probability-weighted CAM at full image resolution (single forward pass).

        out = upsample( normalize( ReLU( sum_d (sum_c p_c * W[c, d]) * F_d ) ) )
    """
    model.eval()
    out = model.forward(image_tensor, retain_transition_grad=False)
    transition_maps = out["transition_maps"]

    # model's cam method works on singular class_idx, but we want to compute a weighted sum across all classes
    # so we compute the weights manually here and then combine them with the transition maps.

    probs = torch.sigmoid(out["logits"])[0]
    combined_weights = probs @ model.prediction.weight
    cam = torch.einsum("d,bdhw->bhw", combined_weights, transition_maps)
    cam = F.relu(cam)
    cam = model._normalize_map(cam)
    cam_full = F.interpolate(
        cam.unsqueeze(1),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return cam_full.squeeze(1).squeeze(0).detach().cpu().numpy()


def _score_bbox_row(
    row: pd.Series,
    model: torch.nn.Module,
    tensor_index: pd.DataFrame,
    pathology_list: list[str],
    cfg: _LocalizationConfig,
) -> dict[str, Any]:
    image_index = row["image_index"]
    finding = row["finding_label"]
    base = {"image_index": image_index, "finding_label": finding}

    class_idx = _resolve_class_index(finding, pathology_list)
    if class_idx is None:
        return {**base, "class_name": None, "iou": float("nan"),
                "status": "label_not_in_pathology_list"}

    class_name = pathology_list[class_idx]
    dataset_row = tensor_index.loc[image_index] if image_index in tensor_index.index else None
    if dataset_row is None:
        return {**base, "class_name": class_name, "iou": float("nan"),
                "status": "image_not_in_test_split"}

    mask_path = None
    if cfg.use_mask_channel and cfg.mask_path_col:
        mask_path = dataset_row[cfg.mask_path_col]

    image_tensor = _load_model_input_tensor(
        dataset_row["image_tensor_path"],
        cfg.device,
        mask_path=mask_path,
        cfg=cfg,
    )
    cam_full = _compute_full_resolution_cam(model, image_tensor, class_idx, cfg.image_size)
    pred_mask = _binarize_cam(
        cam_full, mode=cfg.threshold_mode,
        top_percent=cfg.top_percent, value=cfg.threshold_value,
    )
    gt_mask = _bbox_row_to_gt_mask(row, cfg)
    iou = _iou_from_masks(pred_mask, gt_mask)
    return {**base, "class_name": class_name, "iou": iou, "status": "ok"}


def _aggregate_iou_results(
    per_image: pd.DataFrame, cfg: _LocalizationConfig
) -> tuple[pd.DataFrame, dict[str, float]]:
    scored = per_image[per_image["status"] == "ok"]
    per_class = (
        scored.groupby("class_name")["iou"]
        .agg(n="size", mean_iou="mean", median_iou="median", std_iou="std")
        .reset_index()
        .sort_values("mean_iou", ascending=False)
    )
    overall = {
        "n_bboxes_total": float(len(per_image)),
        "n_bboxes_scored": float(len(scored)),
        "mean_iou": float(scored["iou"].mean()) if len(scored) else float("nan"),
        "median_iou": float(scored["iou"].median()) if len(scored) else float("nan"),
        "macro_mean_iou": float(per_class["mean_iou"].mean()) if len(per_class) else float("nan"),
        "threshold_mode": cfg.threshold_mode,
        "top_percent": cfg.top_percent if cfg.threshold_mode == "top_percent" else float("nan"),
        "threshold_value": cfg.threshold_value if cfg.threshold_mode == "value" else float("nan"),
    }
    return per_class, overall


def evaluate_localization_iou_node(
    cxr8_bbox: pd.DataFrame,
    tensor_test_dataset: pd.DataFrame,
    best_checkpoint_path: str,
    classifier_params: dict[str, Any],
    lit_module_params: dict[str, Any],
    pathology_list: list[str],
    localization_params: dict[str, Any],
    datamodule_kwargs: dict[str, Any],
    transform_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Compute CAM-vs-bbox IoU on the test set.

    `localization_params` keys: image_size, original_size, threshold_mode
    ("top_percent" | "value"), top_percent, threshold_value, filter_to_test_set,
    device (optional, defaults to cuda if available).
    """
    cfg = _LocalizationConfig.from_training_params(
        localization_params=localization_params,
        classifier_params=classifier_params,
        datamodule_kwargs=datamodule_kwargs,
        transform_params=transform_params,
    )
    model = _build_model_from_checkpoint(
        classifier_params=classifier_params,
        lit_module_params=lit_module_params,
        checkpoint_path=best_checkpoint_path,
        device=cfg.device,
    )

    tensor_index = _tensor_dataset_index(tensor_test_dataset)
    bbox_df = cxr8_bbox.copy()
    if cfg.filter_to_test_set:
        bbox_df = bbox_df[bbox_df["image_index"].isin(tensor_index.index)].copy()

    rows = [
        _score_bbox_row(row, model, tensor_index, pathology_list, cfg)
        for _, row in tqdm(
            bbox_df.iterrows(), total=len(bbox_df), desc="Computing localization IoU"
        )
    ]
    per_image = pd.DataFrame(rows)
    per_class, overall = _aggregate_iou_results(per_image, cfg)
    return per_image, per_class, overall


def _score_image_weighted(
    image_index: str,
    image_rows: pd.DataFrame,
    model: torch.nn.Module,
    tensor_index: pd.DataFrame,
    cfg: _LocalizationConfig,
) -> dict[str, Any]:
    findings = sorted(image_rows["finding_label"].unique().tolist())
    base = {
        "image_index": image_index,
        "n_bboxes": int(len(image_rows)),
        "findings": ", ".join(findings),
    }

    dataset_row = tensor_index.loc[image_index] if image_index in tensor_index.index else None
    if dataset_row is None:
        return {**base, "iou": float("nan"), "status": "image_not_in_test_split"}

    mask_path = None
    if cfg.use_mask_channel and cfg.mask_path_col:
        mask_path = dataset_row[cfg.mask_path_col]

    image_tensor = _load_model_input_tensor(
        dataset_row["image_tensor_path"],
        cfg.device,
        mask_path=mask_path,
        cfg=cfg,
    )
    cam_full = _compute_weighted_cam_full(model, image_tensor, cfg.image_size)
    pred_mask = _binarize_cam(
        cam_full, mode=cfg.threshold_mode,
        top_percent=cfg.top_percent, value=cfg.threshold_value,
    )
    gt_mask = _union_bbox_mask(image_rows, cfg)
    return {**base, "iou": _iou_from_masks(pred_mask, gt_mask), "status": "ok"}


def evaluate_weighted_localization_iou_node(
    cxr8_bbox: pd.DataFrame,
    tensor_test_dataset: pd.DataFrame,
    best_checkpoint_path: str,
    classifier_params: dict[str, Any],
    lit_module_params: dict[str, Any],
    localization_params: dict[str, Any],
    datamodule_kwargs: dict[str, Any],
    transform_params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-image IoU between the probability-weighted CAM and the union of all
    GT bboxes for that image. Answers: "does the model's overall attention
    coincide with anywhere there is a finding?"
    """
    cfg = _LocalizationConfig.from_training_params(
        localization_params=localization_params,
        classifier_params=classifier_params,
        datamodule_kwargs=datamodule_kwargs,
        transform_params=transform_params,
    )
    model = _build_model_from_checkpoint(
        classifier_params=classifier_params,
        lit_module_params=lit_module_params,
        checkpoint_path=best_checkpoint_path,
        device=cfg.device,
    )

    tensor_index = _tensor_dataset_index(tensor_test_dataset)
    bbox_df = cxr8_bbox.copy()
    if cfg.filter_to_test_set:
        bbox_df = bbox_df[bbox_df["image_index"].isin(tensor_index.index)].copy()

    grouped = bbox_df.groupby("image_index", sort=False)
    rows = [
        _score_image_weighted(image_index, image_rows, model, tensor_index, cfg)
        for image_index, image_rows in tqdm(
            grouped, total=grouped.ngroups, desc="Computing weighted localization IoU"
        )
    ]
    per_image = pd.DataFrame(rows)
    scored = per_image[per_image["status"] == "ok"]
    overall = {
        "n_images_total": float(len(per_image)),
        "n_images_scored": float(len(scored)),
        "mean_iou": float(scored["iou"].mean()) if len(scored) else float("nan"),
        "median_iou": float(scored["iou"].median()) if len(scored) else float("nan"),
        "threshold_mode": cfg.threshold_mode,
        "top_percent": cfg.top_percent if cfg.threshold_mode == "top_percent" else float("nan"),
        "threshold_value": cfg.threshold_value if cfg.threshold_mode == "value" else float("nan"),
    }
    return per_image, overall
