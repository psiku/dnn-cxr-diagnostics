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


def build_predictions_df(probs, real_targets):
    return (
        pd.DataFrame({
            "Disease": DISEASES,
            "Probability": probs,
            "Is Target": [1 if d in real_targets else 0 for d in DISEASES],
        })
        .sort_values(by="Probability", ascending=False)
        .reset_index(drop=True)
    )


def generate_cam(model, image_tensor, class_idx):
    cam_tensor, _ = model.cam(image_tensor, class_idx)
    return cam_tensor[0].cpu().numpy()