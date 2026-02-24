from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Subset


@dataclass(frozen=True)
class FewShotSplit:
    train_indices: List[int]
    test_indices: List[int]


def make_fewshot_split(
    targets: Sequence[int],
    k_shot: int = 4,
    seed: int = 0,
) -> FewShotSplit:
    """
    Standard supervised few-shot split:
      train = K samples per class
      test  = all remaining samples

    targets: list/array of class indices aligned with dataset samples
    """
    if k_shot <= 0:
        raise ValueError("k_shot must be > 0")

    y = np.asarray(list(targets), dtype=np.int64)
    num_classes = int(y.max()) + 1

    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    test_mask = np.ones(len(y), dtype=bool)

    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        if len(idx_c) < k_shot:
            raise ValueError(f"Class {c} has only {len(idx_c)} samples, cannot sample k={k_shot}.")
        chosen = rng.choice(idx_c, size=k_shot, replace=False)
        train_idx.extend(chosen.tolist())
        test_mask[chosen] = False

    train_idx = sorted(train_idx)
    test_idx = np.where(test_mask)[0].tolist()

    return FewShotSplit(train_indices=train_idx, test_indices=test_idx)


def subset_from_split(dataset, split: FewShotSplit) -> Tuple[Subset, Subset]:
    return Subset(dataset, split.train_indices), Subset(dataset, split.test_indices)
