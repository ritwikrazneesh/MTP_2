from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from .fewshot import FewShotSplit, make_fewshot_split, subset_from_split


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = False


def build_fewshot_loaders(
    dataset,
    k_shot: int,
    seed: int,
    cfg: LoaderConfig,
) -> Tuple[DataLoader, DataLoader, FewShotSplit]:
    # torchvision ImageFolder uses dataset.targets
    split = make_fewshot_split(dataset.targets, k_shot=k_shot, seed=seed)
    train_set, test_set = subset_from_split(dataset, split)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=cfg.drop_last,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, test_loader, split


def describe_split(dataset, split: FewShotSplit, classnames: List[str]) -> str:
    # Basic sanity counts
    import numpy as np

    y = np.asarray(dataset.targets, dtype=int)
    y_tr = y[split.train_indices]
    y_te = y[split.test_indices]
    lines = []
    lines.append(f"Total samples: {len(y)}")
    lines.append(f"Train: {len(y_tr)}  Test: {len(y_te)}")
    for c, name in enumerate(classnames):
        lines.append(
            f"  class[{c:02d}] {name}: train={int((y_tr==c).sum())} test={int((y_te==c).sum())}"
        )
    return "\n".join(lines)
