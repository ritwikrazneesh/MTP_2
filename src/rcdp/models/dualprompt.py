from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layerwise_prefix_tokens import LayerwisePrefixConfig, LayerwisePrefixTokens
from .transformer_inject import inject_layerwise_prefix


@dataclass(frozen=True)
class DualPromptConfig:
    template: str = "a satellite image of {}."
    prefix_len: int = 5
    g_layers: int = 6
    e_layers: int = 6


class RemoteCLIPDualPromptModel(nn.Module):
    """
    Class-wise DualPrompt:
      - 1 global prompt (G) for early layers
      - 1 expert prompt per class (E_c) for late layers
    Applied to BOTH text and vision transformers using layerwise prefix tokens.

    Encoders are expected frozen externally.
    Trainable parameters: prefix tokens only.
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
        # Text
        text_transformer = getattr(self.model, "transformer", None)
        if text_transformer is None:
            raise RuntimeError("RemoteCLIP/OpenCLIP model missing .transformer (text transformer).")

        if not hasattr(text_transformer, "width"):
            # openclip: transformer.width
            raise RuntimeError("Unsupported OpenCLIP transformer: missing .width")
        text_width = int(text_transformer.width)

        text_blocks = getattr(text_transformer, "resblocks", None) or getattr(text_transformer, "blocks", None)
        if text_blocks is None:
            raise RuntimeError("Unsupported OpenCLIP transformer: missing .resblocks/.blocks")
        text_layers = len(text_blocks)

        # Vision
        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise RuntimeError("Model missing .visual")

        # Many openclip ViT visuals have .transformer with .resblocks and .width
        vis_transformer = getattr(visual, "transformer", None)
        if vis_transformer is None:
            raise RuntimeError("Unsupported OpenCLIP visual: missing .transformer (ViT expected).")

        if not hasattr(vis_transformer, "width"):
            raise RuntimeError("Unsupported OpenCLIP visual transformer: missing .width")
        vis_width = int(vis_transformer.width)

        vis_blocks = getattr(vis_transformer, "resblocks", None) or getattr(vis_transformer, "blocks", None)
        if vis_blocks is None:
            raise RuntimeError("Unsupported OpenCLIP visual transformer: missing .resblocks/.blocks")
        vis_layers = len(vis_blocks)

        # --- create prefix modules ---
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
        self.vision_prefix = LayerwisePrefixTokens(
            LayerwisePrefixConfig(
                n_layers=vis_layers,
                width=vis_width,
                prefix_len=cfg.prefix_len,
                g_layers=min(cfg.g_layers, vis_layers),
                e_layers=min(cfg.e_layers, vis_layers),
            ),
            num_classes=self.num_classes,
            device=device,
        )

        # Patch transformers to inject prefixes layerwise.
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

        # Pre-tokenize class prompts once (token IDs fixed; embeddings will change due to prefixes)
        prompts = [cfg.template.format(c) for c in classnames]
        tokenized = self.tokenizer(prompts)
        self.register_buffer("tokenized_prompts", tokenized)

    def trainable_parameters(self):
        return list(self.text_prefix.parameters()) + list(self.vision_prefix.parameters())

    def encode_image(self, images: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        # Set class idx for E prompt routing (needed for deeper layers)
        self.vision_prefix._rcdp_class_idx = class_idx
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text_for_class(self, class_idx: int) -> torch.Tensor:
        self.text_prefix._rcdp_class_idx = class_idx
        tokens = self.tokenized_prompts[class_idx : class_idx + 1].to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0)  # [D]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns logits [B,C]
        """
        B = images.size(0)
        device = images.device

        # Image features: we need E prompt per class too if E applied to vision.
        # For simplicity, compute ONE image feature per class (slow but correct).
        # Later we can refactor: apply only G on vision, E on text, etc.
        img_feats = []
        for c in range(self.num_classes):
            img_feats.append(self.encode_image(images, class_idx=c))  # [B,D]
        img_feats = torch.stack(img_feats, dim=1)  # [B,C,D]

        # Text features per class
        txt = []
        for c in range(self.num_classes):
            txt.append(self.encode_text_for_class(c))  # [D]
        txt = torch.stack(txt, dim=0).to(device)  # [C,D]

        # logits per class using matching (image_feat_c dot text_feat_c) OR full matrix?
        # We want standard classification logits: for each image, score against each class.
        # But image features are class-conditioned if we apply E to vision.
        # We'll use class-conditioned image feat for its class, and compute scores against all classes
        # using that image feature. This yields a [B,C,C] tensor. We'll take diagonal as consistent score.
        # For classification, better: do NOT condition image encoder on class; instead apply prompts on text only.
        #
        # Since you asked "both encoders", we keep it but use diagonal scoring (each class uses its own vision E).
        # This makes logits [B,C] where logit for class c uses image_feat conditioned on c and text_feat of c.
        diag_scores = (img_feats * txt.unsqueeze(0)).sum(dim=-1)  # [B,C]

        # Apply CLIP logit scale
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * diag_scores
