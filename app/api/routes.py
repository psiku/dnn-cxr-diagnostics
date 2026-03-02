"""FastAPI route handlers for the CXR Diagnostics service."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from dnn_cxr_diagnostics.data.dataset import DEFAULT_TRANSFORM
from dnn_cxr_diagnostics.models.cnn import CXRClassifier

router = APIRouter(prefix="/api/v1")

# ---------------------------------------------------------------------------
# Model singleton — loaded once at startup
# ---------------------------------------------------------------------------
_MODEL: CXRClassifier | None = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEFAULT_CHECKPOINT = Path("models/best_model.pt")


def get_model() -> CXRClassifier:
    global _MODEL
    if _MODEL is None:
        _MODEL = CXRClassifier(num_classes=1, pretrained=False)
        if _DEFAULT_CHECKPOINT.exists():
            _MODEL.load_state_dict(torch.load(_DEFAULT_CHECKPOINT, map_location=_DEVICE))
        _MODEL = _MODEL.to(_DEVICE).eval()
    return _MODEL


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health")
def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.post("/predict")
async def predict(file: Annotated[UploadFile, File(description="Chest X-ray image")]) -> JSONResponse:
    """Classify a chest X-ray image.

    Returns a JSON object with the predicted class label and confidence score.
    """
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Only PNG and JPEG images are supported.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image.") from exc

    tensor = DEFAULT_TRANSFORM(image).unsqueeze(0).to(_DEVICE)

    model = get_model()
    with torch.no_grad():
        logit = model(tensor).squeeze()
        score = torch.sigmoid(logit).item()

    label = "PNEUMONIA" if score >= 0.5 else "NORMAL"
    return JSONResponse({"label": label, "confidence": round(score if label == "PNEUMONIA" else 1 - score, 4)})
