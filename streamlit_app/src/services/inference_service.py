import torch
import pandas as pd

from src.constants import DISEASES


def get_real_targets(row):
    return [d for d in DISEASES if row.get(d, 0) == 1]


def predict(model, image_tensor):
    with torch.no_grad():
        out = model(image_tensor)
        logits = out["logits"]
        probs = torch.sigmoid(logits)[0].numpy()
    return probs


def build_predictions_df(probs, real_targets, threshold: list[float] | float | dict = 0.5):
    if isinstance(threshold, dict):
        threshold = [float(threshold.get(d, 0.5)) for d in DISEASES]
    elif isinstance(threshold, float):
        threshold = [threshold] * len(DISEASES)
    elif isinstance(threshold, list):
        if len(threshold) != len(DISEASES):
            raise ValueError(f"Threshold list must have {len(DISEASES)} values.")
        threshold = [float(t) for t in threshold]

    return (
        pd.DataFrame(
            {
                "Disease": DISEASES,
                "Probability": probs,
                "Predicted Target": probs >= threshold,
                "Is Target": [1 if d in real_targets else 0 for d in DISEASES],
            }
        )
        .sort_values(by="Probability", ascending=False)
        .reset_index(drop=True)
    )


def generate_cam(model, image_tensor, class_idx):
    cam_tensor, _ = model.cam(image_tensor, class_idx)
    return cam_tensor[0].cpu().numpy()
