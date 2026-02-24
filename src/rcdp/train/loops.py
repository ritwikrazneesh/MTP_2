from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import accuracy_top1


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    lr: float = 5e-3
    weight_decay: float = 1e-4
    use_amp: bool = True


def train_fewshot(
    model,
    train_loader,
    test_loader,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    model = model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_test = -1.0

    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            acc = accuracy_top1(logits.detach(), labels)
            total_loss += float(loss.item())
            total_acc += acc
            n_batches += 1
            pbar.set_postfix(loss=total_loss / n_batches, acc=total_acc / n_batches)

        # ---- eval ----
        model.eval()
        all_acc = []
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                logits = model(images)
            all_acc.append(accuracy_top1(logits, labels))

        test_acc = float(sum(all_acc) / max(1, len(all_acc)))
        best_test = max(best_test, test_acc)
        print(f"[epoch {epoch+1:03d}] test_acc={test_acc:.4f} best={best_test:.4f}")

    return {"best_test_acc": best_test}
