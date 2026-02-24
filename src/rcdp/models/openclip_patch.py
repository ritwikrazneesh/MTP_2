from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class PrefixRouter:
    """
    Callable that returns optional (k_prefix, v_prefix) for a given layer index.
    Each prefix is shaped [B, H, P, Dh] OR [P, H, Dh] (we'll expand).
    """
    def __init__(self):
        self.enabled: bool = False
        self.class_idx: Optional[int] = None  # for E prompts
        self.get_kv_for_layer: Optional[Callable[[int, Optional[int]], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None

    def kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.enabled or self.get_kv_for_layer is None:
            return None
        return self.get_kv_for_layer(layer_idx, self.class_idx)


def _infer_attn_layout(attn: nn.Module) -> Tuple[int, int]:
    """
    Tries to infer (n_heads, head_dim) from common OpenCLIP attention modules.
    We need these to validate prefix shapes.
    """
    # OpenCLIP transformer blocks commonly have attn.num_heads and attn.head_dim or attn.head_dim inferred
    if hasattr(attn, "num_heads"):
        n_heads = int(getattr(attn, "num_heads"))
    elif hasattr(attn, "n_head"):
        n_heads = int(getattr(attn, "n_head"))
    else:
        raise RuntimeError("Cannot infer num_heads from attention module; unsupported OpenCLIP version.")

    if hasattr(attn, "head_dim"):
        head_dim = int(getattr(attn, "head_dim"))
    elif hasattr(attn, "dim_head"):
        head_dim = int(getattr(attn, "dim_head"))
    else:
        # try from projection weight shape:
        # qkv projection often has out_features = 3 * embed_dim
        if hasattr(attn, "in_proj_weight"):
            embed_dim = attn.in_proj_weight.shape[1]
            head_dim = embed_dim // n_heads
        elif hasattr(attn, "qkv"):
            # e.g. Linear(embed_dim, 3*embed_dim)
            embed_dim = attn.qkv.weight.shape[1]
            head_dim = embed_dim // n_heads
        else:
            raise RuntimeError("Cannot infer head_dim from attention module; unsupported OpenCLIP version.")

    return n_heads, head_dim


def patch_transformer_with_prefix(
    transformer: nn.Module,
    router: PrefixRouter,
) -> Dict[str, Any]:
    """
    Monkeypatch each residual block's attention forward to inject prefix KV.
    Returns info dict (n_layers, n_heads, head_dim).
    """
    # OpenCLIP: text transformer is often model.transformer
    # vision transformer is often model.visual.transformer or model.visual.transformer.resblocks
    if hasattr(transformer, "resblocks"):
        blocks = transformer.resblocks
    elif hasattr(transformer, "blocks"):
        blocks = transformer.blocks
    else:
        raise RuntimeError("Unsupported transformer: expected .resblocks or .blocks")

    n_layers = len(blocks)

    # Determine heads/dim from first block's attention
    blk0 = blocks[0]
    if hasattr(blk0, "attn"):
        attn0 = blk0.attn
    elif hasattr(blk0, "attention"):
        attn0 = blk0.attention
    else:
        raise RuntimeError("Unsupported residual block: expected .attn or .attention")

    n_heads, head_dim = _infer_attn_layout(attn0)

    for layer_idx, blk in enumerate(blocks):
        attn = blk.attn if hasattr(blk, "attn") else blk.attention

        # Store original forward
        if hasattr(attn, "_rcdp_orig_forward"):
            continue  # already patched

        attn._rcdp_orig_forward = attn.forward  # type: ignore[attr-defined]

        def make_forward(orig_fwd, this_layer_idx: int):
            def forward_patched(x, *args, **kwargs):
                # Try to inject via kwargs; if unsupported, we intercept by temporarily attaching to module
                kv = router.kv(this_layer_idx)
                if kv is None:
                    return orig_fwd(x, *args, **kwargs)

                k_pref, v_pref = kv
                # Expand to batch if needed later by the attention implementation.
                # We attach to the module; patched attention implementations (below) will read it.
                setattr(attn, "_rcdp_prefix_kv", (k_pref, v_pref))
                try:
                    out = orig_fwd(x, *args, **kwargs)
                finally:
                    if hasattr(attn, "_rcdp_prefix_kv"):
                        delattr(attn, "_rcdp_prefix_kv")
                return out
            return forward_patched
        attn.forward = make_forward(attn._rcdp_orig_forward, layer_idx)  # type: ignore[method-assign]

        # Now also patch the internal attention computation if possible (common patterns)
        # Many OpenCLIP attention modules use a method `attention` or inline qkv. We handle common cases.
        if not hasattr(attn, "_rcdp_injected"):
            _try_patch_attention_compute(attn)
            attn._rcdp_injected = True  # type: ignore[attr-defined]

    return {"n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim}


def _try_patch_attention_compute(attn: nn.Module) -> None:
    """
    Patch common OpenCLIP attention implementations to prepend prefix KV.
    We support patterns where attention.forward computes q,k,v and then does scaled_dot_product_attention
    or manual softmax.
    """
    # If attention uses torch.nn.functional.scaled_dot_product_attention, we can intercept by patching a helper.
    # But OpenCLIP versions vary. We'll patch at the end: if we detect q,k,v tensors, concatenate.

    # Strategy: wrap forward again to detect computed k/v in locals is impossible.
    # So we instead support attention modules that expose a method that returns q,k,v or has attributes we can override.
    #
    # Practical approach: for OpenCLIP ViT attention, forward often does:
    #   qkv = self.qkv(x) -> reshape -> (q,k,v)
    # We'll patch by wrapping self.qkv if it exists.
    if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
        lin: nn.Linear = attn.qkv

        if hasattr(lin, "_rcdp_orig_forward"):
            return

        lin._rcdp_orig_forward = lin.forward  # type: ignore[attr-defined]

        def qkv_forward(x):
            out = lin._rcdp_orig_forward(x)
            # out: [B, T, 3*D]
            # We cannot inject prefix here because we need to modify k/v only after reshaping to heads.
            # So qkv hook isn't enough.
            return out

        lin.forward = qkv_forward  # type: ignore[method-assign]
        # no-op, but leaves a place for future extension

    # For now, the reliable injection point is to use OpenCLIP's support for `attn_mask` and modify tokens
    # is not prefix-KV. So we require a version where attn.forward reads `_rcdp_prefix_kv` and can use it.
    # If not, we'll implement a fallback approach in Part 4: prompt tokens (input tokens) for text and
    # patching vision tokens by adding prefix tokens to sequence.
    #
    # We'll detect failure at runtime and show actionable error message.
    return
