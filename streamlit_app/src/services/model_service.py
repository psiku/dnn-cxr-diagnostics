import json
from pathlib import Path

import streamlit as st
import torch

from src.config import CKPT_PATH, KEDRO_ROOT
from src.constants import DISEASES
from src.cxr_training_pipeline.models.classifier import ChestXRayClassifier
from src.cxr_training_pipeline.models.backbone import TorchvisionBackbone  # noqa: F401


REPORTING_DIR = (KEDRO_ROOT / "data" / "08_reporting").resolve()


def _resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else Path(CKPT_PATH)
    if not ckpt.is_absolute():
        ckpt = (KEDRO_ROOT / ckpt).resolve()
    return ckpt


def list_saved_models():
    checkpoints_dir = (KEDRO_ROOT / "data" / "06_models").resolve()

    if not checkpoints_dir.exists():
        return []

    checkpoints = [p for p in checkpoints_dir.rglob("*.ckpt") if p.is_file()]
    return sorted(checkpoints, reverse=True)


def find_thresholds_for_checkpoint(checkpoint_path: Path) -> Path | None:
    """Look up `best_thresholds.json` under data/08_reporting for the
    experiment folder that matches the given checkpoint."""
    if not REPORTING_DIR.exists():
        return None

    parts = Path(checkpoint_path).parts
    if "06_models" not in parts:
        return None

    experiment = parts[parts.index("06_models") + 1]
    experiment_dir = REPORTING_DIR / experiment
    if not experiment_dir.exists():
        return None

    candidates = [p for p in experiment_dir.rglob("best_thresholds.json") if p.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {d: 0.5 for d in DISEASES}
    thresholds = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(thresholds, dict):
        raise ValueError("Unsupported threshold file schema.")
    return {k: float(v) for k, v in thresholds.items()}


def _detect_model_config(checkpoint_path: Path):
    path_str = str(checkpoint_path).lower()

    if "efficientb0" in path_str:
        return "efficientnet_b0", 1280
    elif "efficientb7" in path_str:
        return "efficientnet_b7", 2560
    elif "resnet18" in path_str:
        return "resnet18", 512
    elif "resnet50" in path_str:
        return "resnet50", 2048
    elif "resnet101" in path_str:
        return "resnet101", 2048
    elif "densenet121" in path_str:
        return "densenet121", 1024

    return "resnet50", 2048


@st.cache_resource
def load_model(checkpoint_path: str | Path | None = None, num_classes: int = 14):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
        if not k.startswith("model.thresholds") and "thresholds" not in k
    }

    grayscale = False
    for k, v in state_dict.items():
        if "conv1.weight" in k or k.endswith(".features.0.0.weight") or "features.0.weight" in k:
            if v.shape[1] == 1:
                grayscale = True
            break

    model_type, transition_dim = _detect_model_config(checkpoint_path)
    model = ChestXRayClassifier(
        backbone_name=model_type,
        pretrained=True,
        grayscale=grayscale,
        num_classes=num_classes,
        transition_dim=transition_dim,
    )
    model.grayscale = grayscale

    model.load_state_dict(state_dict)
    model.eval()
    return model
