from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Sequence

from torchvision.datasets import ImageFolder


DatasetName = Literal[
    "aid",
    "eurosat_rgb",
    "patternnet",
    "ucm",
    "nwpu-resisc45",
    "whu-rs19",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    root: str


def _assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


def _list_classes_from_imagefolder(root: str) -> Sequence[str]:
    ds = ImageFolder(root=root)
    # ImageFolder sorts classes alphabetically and maps to indices in that order
    return list(ds.classes)


def build_dataset(spec: DatasetSpec):
    """
    Returns:
      dataset: torchvision.datasets.ImageFolder
      classnames: list[str] in dataset index order
    """
    name = spec.name
    root = spec.root

    if name == "aid":
        # Kaggle: /kaggle/input/DATASETS/AID/AID/<Class>/*.jpg
        _assert_exists(root)
        ds = ImageFolder(root=root)
        return ds, list(ds.classes)

    if name == "eurosat_rgb":
        # Kaggle: /kaggle/input/DATASETS/eurosat_rgb/2750/<Class>/*.jpg
        _assert_exists(root)
        ds = ImageFolder(root=root)
        return ds, list(ds.classes)

    if name == "patternnet":
        # Kaggle: /kaggle/input/DATASETS/PatternNet/PatternNet/<class>/*.jpg
        _assert_exists(root)
        ds = ImageFolder(root=root)
        return ds, list(ds.classes)

    if name == "ucm":
        # Kaggle: /kaggle/input/DATASETS/UCMerced LandUse Dataset/UCMerced_LandUse/<class>/*.tif|*.jpg
        _assert_exists(root)
        ds = ImageFolder(root=root)
        return ds, list(ds.classes)

    if name == "nwpu-resisc45":
        # Kaggle:
        # NWPU-RESISC45/Dataset/train/train/<class>/*
        # NWPU-RESISC45/Dataset/test/test/<class>/*
        # We'll build two ImageFolders and later use them in split logic if desired.
        # For v1 we will treat the union and do K-shot split on full set unless user requests official split usage.
        _assert_exists(root)
        ds = ImageFolder(root=root)
        return ds, list(ds.classes)

    if name == "whu-rs19":
        # Kaggle:
        # WHU-RS19/train/<Class>/*
        # WHU-RS19/validation/<Class>/*
        # As above, treat as ImageFolder for the provided root.
        _assert_exists(root)
        ds = ImageFolder(root=root)
        return ds, list(ds.classes)

    raise ValueError(f"Unknown dataset name: {name}")
