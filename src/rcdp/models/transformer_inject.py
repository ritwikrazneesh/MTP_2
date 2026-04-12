from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _get_blocks(transformer: nn.Module):
    if hasattr(transformer, "resblocks"):
        return transformer.resblocks
    if hasattr(transformer, "blocks"):
        return transformer.blocks
    raise RuntimeError("Unsupported transformer: expected .resblocks or .blocks")


def inject_layerwise_prefix(
    transformer: nn.Module,
    prefix_module: nn.Module,
    *,
    get_prefix,  # callable(layer_idx, class_idx_tensor_or_none) -> [P,D] or [B,P,D] or None
) -> None:
    """
    Patches each block.forward to:
      - prepend prefix tokens for selected layers
      - run original forward
      - remove prefix tokens (so shape stays same as original)

    Supports:
      - x: [T,B,D] or [B,T,D]
      - pref: [P,D] (shared) or [B,P,D] (per-sample)
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

                if x.dim() != 3:
                    raise RuntimeError(f"Expected x to be 3D [T,B,D] or [B,T,D], got {x.shape}")

                D = x.shape[-1]
                if pref.shape[-1] != D:
                    raise RuntimeError(f"Prefix dim {pref.shape[-1]} != token dim {D}")

                # Case 1: x is [T,B,D]
                if x.shape[0] != x.shape[1]:
                    T, B, _ = x.shape

                    if pref.dim() == 2:
                        # [P,D] -> [P,B,D]
                        pref_tb = pref.unsqueeze(1).expand(-1, B, -1)
                        P = pref.shape[0]
                    elif pref.dim() == 3:
                        # [B,P,D] -> [P,B,D]
                        if pref.shape[0] != B:
                            raise RuntimeError(f"Batch mismatch: x has B={B} but pref has {pref.shape}")
                        pref_tb = pref.permute(1, 0, 2).contiguous()
                        P = pref.shape[1]
                    else:
                        raise RuntimeError(f"Unsupported pref shape: {pref.shape}")

                    x2 = torch.cat([pref_tb, x], dim=0)  # [P+T,B,D]
                    y2 = orig_fwd(x2, *args, **kwargs)
                    return y2[P:, :, :]

                # Case 2: x is [B,T,D]
                B, T, _ = x.shape
                if pref.dim() == 2:
                    pref_bt = pref.unsqueeze(0).expand(B, -1, -1)  # [B,P,D]
                    P = pref.shape[0]
                elif pref.dim() == 3:
                    if pref.shape[0] != B:
                        raise RuntimeError(f"Batch mismatch: x has B={B} but pref has {pref.shape}")
                    pref_bt = pref
                    P = pref.shape[1]
                else:
                    raise RuntimeError(f"Unsupported pref shape: {pref.shape}")

                x2 = torch.cat([pref_bt, x], dim=1)  # [B,P+T,D]
                y2 = orig_fwd(x2, *args, **kwargs)
                return y2[:, P:, :]

            return forward_patched

        blk.forward = make_forward(blk._rcdp_orig_forward, layer_idx)  # type: ignore[method-assign]
