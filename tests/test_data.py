"""Tests for dataset loading and preprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dnn_cxr_diagnostics.data.dataset import CXRDataset, DEFAULT_TRANSFORM


def _make_fake_dataset(root: Path, classes: list[str], n_per_class: int = 3) -> None:
    """Create a minimal fake dataset on disk."""
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(cls_dir / f"img_{i:03d}.png")


class TestCXRDataset:
    def test_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = ["NORMAL", "PNEUMONIA"]
            _make_fake_dataset(root, classes, n_per_class=4)
            ds = CXRDataset(root)
            assert len(ds) == 8

    def test_class_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_dataset(root, ["A", "B", "C"], n_per_class=2)
            ds = CXRDataset(root)
            assert ds.classes == ["A", "B", "C"]

    def test_explicit_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_dataset(root, ["NORMAL", "PNEUMONIA"], n_per_class=2)
            ds = CXRDataset(root, classes=["NORMAL", "PNEUMONIA"])
            assert ds.class_to_idx == {"NORMAL": 0, "PNEUMONIA": 1}

    def test_getitem_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_dataset(root, ["NORMAL", "PNEUMONIA"], n_per_class=1)
            ds = CXRDataset(root)
            image, label = ds[0]
            assert image.shape == (3, 224, 224)
            assert label in {0, 1}

    def test_empty_class_dir_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_dataset(root, ["NORMAL"], n_per_class=2)
            (root / "EMPTY").mkdir()
            ds = CXRDataset(root)
            assert len(ds) == 2
