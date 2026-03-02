"""Tests for training loop helpers."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dnn_cxr_diagnostics.training.train import evaluate, train_one_epoch


def _make_loader(n: int = 16, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class _TinyModel(nn.Module):
    """Lightweight stand-in for tests to avoid loading ResNet weights."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class TestTrainOneEpoch:
    def test_returns_float(self):
        model = _TinyModel()
        loader = _make_loader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cpu")
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_loss_decreases_over_epochs(self):
        """Training loss should generally decrease over multiple epochs on fixed data."""
        model = _TinyModel()
        loader = _make_loader(n=32, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cpu")

        losses = [
            train_one_epoch(model, loader, optimizer, criterion, device)
            for _ in range(5)
        ]
        assert losses[-1] < losses[0], "Loss did not decrease over 5 epochs"


class TestEvaluate:
    def test_returns_loss_and_accuracy(self):
        model = _TinyModel()
        loader = _make_loader()
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cpu")
        loss, acc = evaluate(model, loader, criterion, device)
        assert isinstance(loss, float) and loss >= 0.0
        assert 0.0 <= acc <= 1.0
