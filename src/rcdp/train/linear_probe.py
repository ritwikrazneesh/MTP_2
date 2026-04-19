from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass(frozen=True)
class LinearProbeConfig:
    epochs: int = 200
    lr: float = 1e-2
    weight_decay: float = 0.0
    use_amp: bool = True
    eval_every: int = 10


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def eval_linearprobe(backbone, head: nn.Module, loader, device: torch.device, use_amp: bool) -> float:
    backbone.eval()
    head.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            feats = backbone.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = head(feats)

        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())

    return float(correct / max(1, total))


def train_linearprobe(
    backbone,
    train_loader,
    test_loader,
    num_classes: int,
    cfg: LinearProbeConfig,
    device: torch.device,
) -> Dict:
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # infer embedding dim by a single forward pass
    images0, _ = next(iter(train_loader))
    images0 = images0.to(device)
    with torch.no_grad():
        feats0 = backbone.encode_image(images0)
        in_dim = int(feats0.shape[-1])

    head = LinearHead(in_dim=in_dim, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    best = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(cfg.epochs):
        head.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"lp epoch {epoch+1}/{cfg.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.type == "cuda")):
                feats = backbone.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = head(feats)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item())
            n += 1
            pbar.set_postfix(loss=total_loss / max(1, n))

        if cfg.eval_every > 0 and ((epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1):
            acc = eval_linearprobe(backbone, head, test_loader, device=device, use_amp=cfg.use_amp)
            if acc > best:
                best = acc
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(f"[lp epoch {epoch+1:03d}] test_acc={acc:.4f} best={best:.4f}")

    # load best weights into head for saving/eval
    if best_state is not None:
        head.load_state_dict(best_state)

    return {"best_test_acc": best, "head_state_dict": best_state, "in_dim": in_dim}
