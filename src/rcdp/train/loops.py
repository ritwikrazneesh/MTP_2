from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .metrics import accuracy_top1


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    lr: float = 5e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    eval_every: int = 1
    max_test_batches: int = 0  # 0 = full test

    # Regularization knobs (safe defaults: disabled)
    prompt_norm_max: float = 0.0  # 0 disables norm clamp


def _make_eval_loader_if_capped(test_loader: DataLoader, max_batches: int, seed: int) -> DataLoader:
    """
    If max_batches > 0, evaluate on a deterministic random subset of the *test dataset*.
    This avoids the huge bias of "first N batches" when shuffle=False.
    """
    if max_batches <= 0:
        return test_loader

    ds = test_loader.dataset
    bs = int(test_loader.batch_size or 1)
    n = min(len(ds), max_batches * bs)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False).tolist()

    sub = Subset(ds, idx)

    return DataLoader(
        sub,
        batch_size=bs,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory,
        drop_last=False,
    )


@torch.no_grad()
def _eval(model, test_loader: DataLoader, cfg: TrainConfig, device: torch.device, *, seed: int) -> float:
    model.eval()

    # Use random subset loader if capped
    eval_loader = _make_eval_loader_if_capped(test_loader, cfg.max_test_batches, seed=seed)

    correct = 0
    total = 0

    for images, labels in eval_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.type == "cuda")):
            logits = model(images)

        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    return float(correct / max(1, total))


def _clamp_prompt_norms(params, max_norm: float):
    if max_norm <= 0:
        return
    with torch.no_grad():
        for p in params:
            # Flatten per-parameter tensor norm clamp
            n = p.data.norm()
            if n > max_norm:
                p.data.mul_(max_norm / (n + 1e-12))


def train_fewshot(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    *,
    seed: int = 0,
) -> Dict:
    model = model.to(device)

    train_params = [p for p in model.trainable_parameters() if p.requires_grad]
    if cfg.epochs <= 0 or len(train_params) == 0:
        print(f"[train] Skipping training (epochs={cfg.epochs}, trainable_params={len(train_params)}). Eval-only.")
        test_acc = _eval(model, test_loader, cfg, device, seed=seed)
        return {"best_test_acc": test_acc, "best_state_dict": None}

    optim = torch.optim.AdamW(train_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    best_test = -1.0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(cfg.epochs):
        # model.train() will keep backbone eval due to our override
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.type == "cuda")):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            # Optional prompt norm clamp (regularization)
            _clamp_prompt_norms(train_params, cfg.prompt_norm_max)

            acc = accuracy_top1(logits.detach(), labels)
            total_loss += float(loss.item())
            total_acc += acc
            n_batches += 1
            pbar.set_postfix(loss=total_loss / n_batches, acc=total_acc / n_batches)

        train_loss = total_loss / max(1, n_batches)
        train_acc = total_acc / max(1, n_batches)
        print(f"[epoch {epoch+1:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        if cfg.eval_every <= 0:
            continue
        if (epoch + 1) % cfg.eval_every != 0:
            continue

        # Make eval subset deterministic but different per epoch if you want:
        # use seed + epoch; if you want same subset every epoch, use seed only.
        test_acc = _eval(model, test_loader, cfg, device, seed=seed + epoch)

        if test_acc > best_test:
            best_test = test_acc
            # Save best weights
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[epoch {epoch+1:03d}] test_acc={test_acc:.4f} best={best_test:.4f}")

    return {"best_test_acc": best_test, "best_state_dict": best_state_dict}
