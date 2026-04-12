from __future__ import annotations

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
    get_prefix,  # callable(layer_idx, batch_class_idx) -> [B,P,D] or None
) -> None:
    """
    Stable injection (no heuristics).

    Requires:
      - prefix_module._rcdp_class_idx is set to LongTensor[B] before encode
      - get_prefix returns either None or [B,P,D]
      - x is [T,B,D] or [B,T,D]
    Layout is determined ONLY by matching B.
    """
    blocks = _get_blocks(transformer)

    for layer_idx, blk in enumerate(blocks):
        if hasattr(blk, "_rcdp_orig_forward"):
            continue
        blk._rcdp_orig_forward = blk.forward  # type: ignore[attr-defined]

        def make_forward(orig_fwd, this_layer_idx: int):
            def forward_patched(x, *args, **kwargs):
                batch_class_idx = getattr(prefix_module, "_rcdp_class_idx", None)
                if batch_class_idx is None:
                    raise RuntimeError(
                        "prefix_module._rcdp_class_idx must be set to LongTensor[B] before calling encoder."
                    )

                pref = get_prefix(this_layer_idx, batch_class_idx)
                if pref is None:
                    return orig_fwd(x, *args, **kwargs)

                if x.dim() != 3:
                    raise RuntimeError(f"Expected x to be 3D, got {x.shape}")
                if pref.dim() != 3:
                    raise RuntimeError(f"Expected pref to be [B,P,D], got {pref.shape}")

                B, P, Dp = pref.shape
                D = x.shape[-1]
                if Dp != D:
                    raise RuntimeError(f"Prefix dim {Dp} != token dim {D}")

                # Decide layout by matching B
                if x.shape[0] == B:
                    # x: [B,T,D]
                    x2 = torch.cat([pref, x], dim=1)  # [B,P+T,D]
                    y2 = orig_fwd(x2, *args, **kwargs)
                    return y2[:, P:, :]
                elif x.shape[1] == B:
                    # x: [T,B,D]
                    pref_tb = pref.permute(1, 0, 2).contiguous()  # [P,B,D]
                    x2 = torch.cat([pref_tb, x], dim=0)  # [P+T,B,D]
                    y2 = orig_fwd(x2, *args, **kwargs)
                    return y2[P:, :, :]
                else:
                    raise RuntimeError(f"Cannot determine layout: x={tuple(x.shape)}, pref={tuple(pref.shape)}")

            return forward_patched

        blk.forward = make_forward(blk._rcdp_orig_forward, layer_idx)  # type: ignore[method-assign]
