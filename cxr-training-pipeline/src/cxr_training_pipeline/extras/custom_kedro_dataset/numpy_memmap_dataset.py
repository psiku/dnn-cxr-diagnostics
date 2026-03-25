from pathlib import Path
import numpy as np
from kedro.io import AbstractDataset
from typing import Any


class NumpyMemmapDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def _load(self) -> np.memmap:
        return np.load(self._filepath, mmap_mode="r")

    def _save(self, data: np.ndarray) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._filepath, data)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}