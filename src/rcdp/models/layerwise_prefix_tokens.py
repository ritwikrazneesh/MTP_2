from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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
    Implements DualPrompt-like:
      - G prompt tokens injected for first g_layers
      - E prompt tokens injected for last e_layers

    Injection mechanism:
      for a given transformer block input sequence x: [T, B, D] or [B, T, D]
      we concatenate prefix tokens along T dimension:
         x' = cat([prefix, x], dim=T)

    We then remove the prefix tokens after the block to keep sequence length stable.

    This requires patching each residual block forward with a wrapper.
    """
    def __init__(self, cfg: LayerwisePrefixConfig, num_classes: int, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # G: [L, P, D]
        self.g = nn.Parameter(torch.randn(cfg.n_layers, cfg.prefix_len, cfg.width, device=device) * cfg.init_std)

        # E: [C, L, P, D]
        self.e = nn.Parameter(torch.randn(num_classes, cfg.n_layers, cfg.prefix_len, cfg.width, device=device) * cfg.init_std)

    def prefix_for_layer(self, layer_idx: int, class_idx: Optional[int]) -> Optional[torch.Tensor]:
        """
        Returns prefix tokens [P, D] for this layer, already combined:
          - if layer is in G range -> include G
          - if layer is in E range -> include E[class]
        If both apply, we concatenate: [P_g + P_e, D]
        """
        P = self.cfg.prefix_len
        out = []

        # G layers: [0, g_layers)
        if layer_idx < self.cfg.g_layers:
            out.append(self.g[layer_idx])

        # E layers: [n_layers - e_layers, n_layers)
        if layer_idx >= (self.cfg.n_layers - self.cfg.e_layers):
            if class_idx is None:
                raise ValueError("class_idx must be provided for E-prompt layers")
            out.append(self.e[class_idx, layer_idx])

        if not out:
            return None
        return torch.cat(out, dim=0)  # [P_total, D]
