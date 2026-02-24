from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def _get_blocks(transformer: nn.Module):
    if hasattr(transformer, "resblocks"):
        return transformer.resblocks
    if hasattr(transformer, "blocks"):
        return transformer.blocks
    raise RuntimeError("Unsupported transformer: expected .resblocks or .blocks")


def _is_seq_first(x: torch.Tensor) -> bool:
    # Heuristic: OpenCLIP text transformer commonly uses [T,B,D]
    # ViT blocks sometimes use [B,T,D]
    # We'll detect by comparing dimensions (B usually smaller than T? not always).
    # Instead: if x.dim==3 and x.shape[0] <= 512 and x.shape[1] <= 1024, ambiguous.
    # We'll use a safer rule: treat as [T,B,D] if second dim looks like batch (<=1024) and first dim looks like tokens (>1)
    return True  # default; per-block wrappers will handle both by checking shape after creating prefix.


def inject_layerwise_prefix(
    transformer: nn.Module,
    prefix_module: nn.Module,
    *,
    get_prefix,  # callable(layer_idx, class_idx) -> [P,D] or None
) -> None:
    """
    Patches each block.forward to:
      - prepend prefix tokens for selected layers
      - run original forward
      - remove prefix tokens (so shape stays same as original)
    """
    blocks = _get_blocks(transformer)

    for layer_idx, blk in enumerate(blocks):
        if hasattr(blk, "_rcdp_orig_forward"):
            continue
        blk._rcdp_orig_forward = blk.forward  # type: ignore[attr-defined]

        def make_forward(orig_fwd, this_layer_idx: int):
            def forward_patched(x, *args, **kwargs):
                class_idx = getattr(prefix_module, "_rcdp_class_idx", None)
                pref = get_prefix(this_layer_idx, class_idx)
                if pref is None:
                    return orig_fwd(x, *args, **kwargs)

                # pref: [P,D] -> expand to batch
                if x.dim() != 3:
                    raise RuntimeError(f"Expected x to be 3D [T,B,D] or [B,T,D], got {x.shape}")

                # Determine layout by matching last dim
                D = x.shape[-1]
                if pref.shape[-1] != D:
                    raise RuntimeError(f"Prefix dim {pref.shape[-1]} != token dim {D}")

                if x.shape[0] == D and x.shape[-1] != D:
                    # impossible; ignore
                    pass

                # If x is [T,B,D]
                if x.shape[2] == D and x.shape[0] != x.shape[1]:
                    T, B, _ = x.shape
                    pref_tb = pref.unsqueeze(1).expand(-1, B, -1)  # [P,B,D]
                    x2 = torch.cat([pref_tb, x], dim=0)  # [P+T,B,D]
                    y2 = orig_fwd(x2, *args, **kwargs)
                    y = y2[pref.shape[0] :, :, :]  # remove prefix
                    return y

                # If x is [B,T,D]
                B, T, _ = x.shape
                pref_bt = pref.unsqueeze(0).expand(B, -1, -1)  # [B,P,D]
                x2 = torch.cat([pref_bt, x], dim=1)  # [B,P+T,D]
                y2 = orig_fwd(x2, *args, **kwargs)
                y = y2[:, pref.shape[0] :, :]  # remove prefix
                return y
            return forward_patched

        blk.forward = make_forward(blk._rcdp_orig_forward, layer_idx)  # type: ignore[method-assign]
