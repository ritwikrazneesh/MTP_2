from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LayerwisePrefixConfig:
    n_layers: int
    width: int
    prefix_len: int
    g_layers: int  # number of early layers to receive G prompt
    e_layers: int  # number of late layers to receive E prompt
    init_std: float = 0.02


class LayerwisePrefixTokens(nn.Module):
    """
    Stable batched prefix tokens.

    - Always returns prefixes as [B, P_total, D] (never [P, D])
    - Layout detection in injector becomes stable by matching B.
    """

    def __init__(
        self,
        cfg: LayerwisePrefixConfig,
        num_classes: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # G: [L, P, D]
        self.g = nn.Parameter(
            torch.randn(cfg.n_layers, cfg.prefix_len, cfg.width, device=device) * cfg.init_std
        )

        # E: [C, L, P, D]
        self.e = nn.Parameter(
            torch.randn(num_classes, cfg.n_layers, cfg.prefix_len, cfg.width, device=device) * cfg.init_std
        )

        # Routing signal:
        # MUST be set before encoding:
        #   - for text: LongTensor[C] = [0..C-1]
        #   - for vision: LongTensor[B] dummy (values irrelevant when e_layers=0)
        self._rcdp_class_idx: Optional[torch.Tensor] = None

    def prefix_for_layer(self, layer_idx: int, batch_class_idx: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns prefix tokens for this layer as [B, P_total, D] or None.
        """
        if batch_class_idx is None:
            raise ValueError("batch_class_idx must be a LongTensor[B]. Set prefix_module._rcdp_class_idx before encode.")
        if batch_class_idx.dtype != torch.long:
            batch_class_idx = batch_class_idx.long()

        if batch_class_idx.dim() != 1:
            raise ValueError(f"batch_class_idx must be 1D [B], got {tuple(batch_class_idx.shape)}")

        B = int(batch_class_idx.shape[0])
        parts = []

        # G layers: [0, g_layers)
        if layer_idx < self.cfg.g_layers:
            g = self.g[layer_idx]  # [P, D]
            parts.append(g.unsqueeze(0).expand(B, -1, -1))  # [B, P, D]

        # E layers: [n_layers - e_layers, n_layers)
        if self.cfg.e_layers > 0 and layer_idx >= (self.cfg.n_layers - self.cfg.e_layers):
            e = self.e[batch_class_idx, layer_idx]  # [B, P, D]
            parts.append(e)

        if not parts:
            return None

        return torch.cat(parts, dim=1)  # [B, P_total, D]
