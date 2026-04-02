import torch
import streamlit as st
from pathlib import Path
import json
from src.config import CKPT_PATH, KEDRO_ROOT
from src.constants import DISEASES
from src.cxr_training_pipeline.models.paper_based.classifier import CXRClassifier
from src.cxr_training_pipeline.models.resnet_backbone import ResNet50Backbone


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

def _load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        return {d: 0.5 for d in DISEASES}
    if not path.exists():
        return {d: 0.5 for d in DISEASES}
    if not path.suffix.lower() == ".json":
        raise ValueError("Unsupported threshold file schema.")
    thresholds = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(thresholds, dict):
        return {k: float(v) for k, v in thresholds.items()}
    raise ValueError("Unsupported threshold file schema.")


@st.cache_resource
def load_model(checkpoint_path: str | Path | None = None):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    backbone = ResNet50Backbone(pretrained=False, grayscale=True)
    model = CXRClassifier(num_classes=14, backbone=backbone)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        k.replace("model.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }

    model.load_state_dict(state_dict)
    model.eval()
    return model