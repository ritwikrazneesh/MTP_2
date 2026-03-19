from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class TransformDataset(Dataset):
    def __init__(self, base: Dataset, transform: Any):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
