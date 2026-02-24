from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TransformBundle:
    train: Any
    test: Any


def build_transforms(preprocess_train: Any, preprocess_val: Any) -> TransformBundle:
    # OpenCLIP already provides proper normalization, resize, center crop etc.
    return TransformBundle(train=preprocess_train, test=preprocess_val)
