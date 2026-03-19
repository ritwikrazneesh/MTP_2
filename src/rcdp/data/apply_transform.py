from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class TransformDataset(Dataset):
    def __init__(self, base: Dataset, transform: Any):
        self.base = base
        self.transform = transform

        # Proxy common ImageFolder attributes (so few-shot split logic works)
        if hasattr(base, "targets"):
            self.targets = base.targets  # type: ignore[attr-defined]
        if hasattr(base, "classes"):
            self.classes = base.classes  # type: ignore[attr-defined]
        if hasattr(base, "class_to_idx"):
            self.class_to_idx = base.class_to_idx  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
