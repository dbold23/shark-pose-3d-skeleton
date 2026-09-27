"""
Hybrid encoder: frozen DINOv2 features + SAM2 silhouette mask CNN.

Experiment D — tests whether adding an explicit silhouette input on top
of DINOv2 features improves pose estimation (hypothesis: silhouette provides
structural body-outline cues that general-purpose features may lack).

Input:
    features: (B, 768) pre-extracted DINOv2 CLS tokens
    mask:     (B, 1, 256, 256) binary SAM2 silhouette mask

Output:
    (B, projection_dim) fused features
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from .dino_encoder import _build_projection_head, _dino_feature_dim

class HybridEncoder(nn.Module):
    """DINOv2 features + SAM2 silhouette mask → fused feature vector.

    The DINOv2 branch: pre-extracted CLS token → projection to ``projection_dim``.
    The mask branch: small CNN reduces (1, 256, 256) to a ``mask_cnn_channels``-D vector.
    Fusion: concatenate both → linear projection back to ``projection_dim``.

    Interface:
        forward(encoder_input: dict) -> (B, projection_dim)
            encoder_input must have keys "features" and "mask".
        output_dim -> projection_dim
    """

    def __init__(self, dino_model: str='dinov2_vitb14', projection_dim: int=2048, mask_cnn_channels: int=128):
        ...

    def forward(self, encoder_input: dict) -> Tensor:
        """Fuse DINOv2 features with silhouette mask.

        Args:
            encoder_input: dict with:
                "features": (B, dino_dim) pre-extracted DINOv2 CLS token.
                "mask": (B, 1, 256, 256) SAM2 silhouette mask.

        Returns:
            (B, projection_dim) fused feature vector.
        """
        ...

    @property
    def output_dim(self) -> int:
        ...
