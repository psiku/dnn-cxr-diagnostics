"""Dataset and data-loading helpers for chest X-ray images."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CXRDataset(Dataset):
    """Chest X-ray image dataset.

    Expects a directory layout::

        root/
          <class_a>/
            img1.png
            img2.png
          <class_b>/
            img3.png

    Args:
        root: Path to the dataset root directory.
        transform: Optional torchvision transform applied to each image.
            Defaults to :data:`DEFAULT_TRANSFORM`.
        classes: Ordered list of class names.  Inferred from sub-directory
            names when *None*.
    """

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        classes: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or DEFAULT_TRANSFORM

        if classes is not None:
            self.classes = classes
        else:
            self.classes = sorted(
                p.name for p in self.root.iterdir() if p.is_dir()
            )

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
