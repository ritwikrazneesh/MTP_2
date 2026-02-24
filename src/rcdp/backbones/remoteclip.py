from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import open_clip


@dataclass(frozen=True)
class RemoteCLIPConfig:
    model_name: str = "ViT-B-32"
    checkpoint_path: str = "/kaggle/input/remoteclip-vitb32-pt/RemoteCLIP-ViT-B-32.pt"
    device: str = "cuda"
    precision: str = "amp"  # "fp32" or "amp"


@dataclass
class RemoteCLIPBundle:
    model: Any
    preprocess_train: Any
    preprocess_val: Any
    tokenizer: Any


def load_remoteclip(cfg: RemoteCLIPConfig) -> RemoteCLIPBundle:
    """
    Matches the official RemoteCLIP loading flow:
      model, _, preprocess = open_clip.create_model_and_transforms(model_name)
      ckpt = torch.load(path)
      model.load_state_dict(ckpt)
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=cfg.model_name,
        pretrained="openai",  # base init; then we load RemoteCLIP weights
        device=str(device),
    )
    tokenizer = open_clip.get_tokenizer(cfg.model_name)

    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    msg = model.load_state_dict(ckpt, strict=True)
    print(f"[RemoteCLIP] load_state_dict: {msg}")

    model = model.to(device).eval()
    return RemoteCLIPBundle(
        model=model,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        tokenizer=tokenizer,
    )


def freeze_remoteclip(model: Any) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
