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
    DualPrompt-like prefix tokens:
      - G prompt tokens injected for first g_layers
      - E prompt tokens injected for last e_layers

    Supports batched routing:
      - class_idx is None                 -> no E prompt (G-only possible)
      - class_idx is Tensor[B] (long)     -> per-sample E prompts

    Returned prefix shapes:
      - if class_idx is None: [P_total, D]
      - if class_idx is Tensor[B]: [B, P_total, D]
    """

    def __init__(self, cfg: LayerwisePrefixConfig, num_classes: int, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # G: [L, P, D]
        self.g = nn.Parameter(torch.randn(cfg.n_layers, cfg.prefix_len, cfg.width, device=device) * cfg.init_std)

        # E: [C, L, P, D]
        self.e = nn.Parameter(torch.randn(num_classes, cfg.n_layers, cfg.prefix_len, cfg.width, device=device) * cfg.init_std)

        # Routing signal (set by model before encode_* calls)
        # None OR LongTensor[B]
        self._rcdp_class_idx: Optional[torch.Tensor] = None

    def prefix_for_layer(self, layer_idx: int, class_idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Returns prefix tokens for this layer, already combined (G then E).

        If class_idx is:
          - None: returns [P_total, D] or None
          - Tensor[B]: returns [B, P_total, D] or None
        """
        out = []

        # G layers: [0, g_layers)
        if layer_idx < self.cfg.g_layers:
            g = self.g[layer_idx]  # [P, D]
            out.append(g)

        # E layers: [n_layers - e_layers, n_layers)
        use_e = layer_idx >= (self.cfg.n_layers - self.cfg.e_layers)
        if use_e:
            if class_idx is None:
                raise ValueError("class_idx must be provided for E-prompt layers (use None only for G-only encoders).")
            if class_idx.dtype != torch.long:
                class_idx = class_idx.long()

            # Gather E for each sample in batch: [B, P, D]
            e = self.e[class_idx, layer_idx]  # advanced indexing
            out.append(e)

        if not out:
            return None

        # Combine G + E
        if class_idx is None:
            # Only possible tensors in out are [P,D]
            return torch.cat(out, dim=0)  # [P_total, D]

        # class_idx is Tensor[B], ensure all parts are [B,P,D]
        B = int(class_idx.shape[0])
        parts = []
        for t in out:
            if t.dim() == 2:
                # [P,D] -> [B,P,D]
                parts.append(t.unsqueeze(0).expand(B, -1, -1))
            elif t.dim() == 3:
                parts.append(t)
            else:
                raise RuntimeError(f"Unexpected prefix tensor dim: {t.shape}")
        return torch.cat(parts, dim=1)  # [B, P_total, D]
