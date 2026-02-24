from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .layerwise_prefix_tokens import LayerwisePrefixConfig, LayerwisePrefixTokens
from .transformer_inject import inject_layerwise_prefix


@dataclass(frozen=True)
class DualPromptConfig:
    template: str = "a satellite image of {}."
    prefix_len: int = 5
    g_layers: int = 6
    e_layers: int = 6  # applied to TEXT only in Option X


class RemoteCLIPDualPromptModel(nn.Module):
    """
    Option X (recommended):
      - Vision encoder: G-only (class-agnostic)
      - Text encoder: G + E[class]
    Both encoders remain frozen (outside). Trainable parameters: prefix tokens only.
    """
    def __init__(
        self,
        remoteclip_model,
        tokenizer,
        classnames: List[str],
        cfg: DualPromptConfig,
        device: torch.device,
    ):
        super().__init__()
        self.model = remoteclip_model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.cfg = cfg
        self.device = device

        self.num_classes = len(classnames)

        # --- discover transformer widths and layer counts ---
        # Text transformer
        text_transformer = getattr(self.model, "transformer", None)
        if text_transformer is None or not hasattr(text_transformer, "width"):
            raise RuntimeError("RemoteCLIP/OpenCLIP model missing .transformer.width (text transformer).")
        text_width = int(text_transformer.width)
        text_blocks = getattr(text_transformer, "resblocks", None) or getattr(text_transformer, "blocks", None)
        if text_blocks is None:
            raise RuntimeError("Unsupported text transformer: missing .resblocks/.blocks")
        text_layers = len(text_blocks)

        # Vision transformer (ViT expected)
        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise RuntimeError("Model missing .visual")
        vis_transformer = getattr(visual, "transformer", None)
        if vis_transformer is None or not hasattr(vis_transformer, "width"):
            raise RuntimeError("Unsupported visual backbone: expected ViT with .transformer.width")
        vis_width = int(vis_transformer.width)
        vis_blocks = getattr(vis_transformer, "resblocks", None) or getattr(vis_transformer, "blocks", None)
        if vis_blocks is None:
            raise RuntimeError("Unsupported visual transformer: missing .resblocks/.blocks")
        vis_layers = len(vis_blocks)

        # --- create prefix modules ---
        # TEXT: G early + E late
        self.text_prefix = LayerwisePrefixTokens(
            LayerwisePrefixConfig(
                n_layers=text_layers,
                width=text_width,
                prefix_len=cfg.prefix_len,
                g_layers=min(cfg.g_layers, text_layers),
                e_layers=min(cfg.e_layers, text_layers),
            ),
            num_classes=self.num_classes,
            device=device,
        )

        # VISION: G-only (set e_layers=0)
        self.vision_prefix = LayerwisePrefixTokens(
            LayerwisePrefixConfig(
                n_layers=vis_layers,
                width=vis_width,
                prefix_len=cfg.prefix_len,
                g_layers=min(cfg.g_layers, vis_layers),
                e_layers=0,
            ),
            num_classes=self.num_classes,
            device=device,
        )

        inject_layerwise_prefix(
            text_transformer,
            self.text_prefix,
            get_prefix=lambda layer_idx, class_idx: self.text_prefix.prefix_for_layer(layer_idx, class_idx),
        )
        inject_layerwise_prefix(
            vis_transformer,
            self.vision_prefix,
            get_prefix=lambda layer_idx, class_idx: self.vision_prefix.prefix_for_layer(layer_idx, class_idx),
        )

        prompts = [cfg.template.format(c) for c in classnames]
        tokenized = self.tokenizer(prompts)
        self.register_buffer("tokenized_prompts", tokenized)

    def trainable_parameters(self):
        return list(self.text_prefix.parameters()) + list(self.vision_prefix.parameters())

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # G-only, class-agnostic
        self.vision_prefix._rcdp_class_idx = None
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text_all_classes(self) -> torch.Tensor:
        # Compute text features for all classes with E routing.
        feats = []
        for c in range(self.num_classes):
            self.text_prefix._rcdp_class_idx = c
            tokens = self.tokenized_prompts[c : c + 1].to(self.device)
            t = self.model.encode_text(tokens)
            t = t / t.norm(dim=-1, keepdim=True)
            feats.append(t.squeeze(0))
        return torch.stack(feats, dim=0)  # [C,D]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Standard CLIP logits: [B,C]
        """
        img = self.encode_image(images)             # [B,D]
        txt = self.encode_text_all_classes()        # [C,D]
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * (img @ txt.t())
