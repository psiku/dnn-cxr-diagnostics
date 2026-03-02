"""Tests for CNN model definitions."""

import torch
import pytest

from dnn_cxr_diagnostics.models.cnn import CXRClassifier


class TestCXRClassifier:
    def test_output_shape_binary(self):
        model = CXRClassifier(num_classes=1, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"

    def test_output_shape_multiclass(self):
        model = CXRClassifier(num_classes=5, pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"

    def test_no_pretrained(self):
        """Model can be instantiated without downloading ImageNet weights."""
        model = CXRClassifier(pretrained=False)
        assert model is not None

    def test_forward_does_not_raise(self):
        model = CXRClassifier(num_classes=1, pretrained=False)
        model.eval()
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            _ = model(x)
