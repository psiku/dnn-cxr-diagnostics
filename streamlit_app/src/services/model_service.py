import torch
import streamlit as st

from src.config import CKPT_PATH
from src.cxr_training_pipeline.models.paper_based.classifier import CXRClassifier
from src.cxr_training_pipeline.models.resnet_backbone import ResNet50Backbone


@st.cache_resource
def load_model():
    backbone = ResNet50Backbone(pretrained=False, grayscale=True)
    model = CXRClassifier(num_classes=14, backbone=backbone)

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = {
        k.replace("model.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }

    model.load_state_dict(state_dict)
    model.eval()
    return model