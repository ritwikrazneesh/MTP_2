from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PrefixKVConfig:
    n_layers: int
    n_heads: int
    head_dim: int
    prefix_len: int
    init_std: float = 0.02


class PrefixKV(nn.Module):
    """
    Stores learnable prefix key/value for multiple transformer layers.

    We store per-layer:
      key:   [prefix_len, n_heads, head_dim]
      value: [prefix_len, n_heads, head_dim]

    During forward, these are expanded to batch and concatenated to the attention KV.
    """
    def __init__(self, cfg: PrefixKVConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg

        shape = (cfg.n_layers, cfg.prefix_len, cfg.n_heads, cfg.head_dim)
        self.key = nn.Parameter(torch.randn(*shape, device=device) * cfg.init_std)
        self.value = nn.Parameter(torch.randn(*shape, device=device) * cfg.init_std)

    def layer_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns [P, H, Dh] each
        return self.key[layer_idx], self.value[layer_idx]


class ClasswisePrefixKV(nn.Module):
    """
    Prefix KV per class (E prompts). Shape:
      key:   [C, L, P, H, Dh]
      value: [C, L, P, H, Dh]
    """
    def __init__(self, num_classes: int, cfg: PrefixKVConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.num_classes = num_classes
        self.cfg = cfg

        shape = (num_classes, cfg.n_layers, cfg.prefix_len, cfg.n_heads, cfg.head_dim)
        self.key = nn.Parameter(torch.randn(*shape, device=device) * cfg.init_std)
        self.value = nn.Parameter(torch.randn(*shape, device=device) * cfg.init_std)

    def layer_kv(self, class_idx: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key[class_idx, layer_idx], self.value[class_idx, layer_idx]
