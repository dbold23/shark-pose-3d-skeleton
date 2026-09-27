"""
DINOv2-based encoders for the encoder comparison experiments.

Two variants:
    - DINOv2FrozenEncoder: pre-extracted CLS features → learned projection head.
      Fast, cheap, no ViT in the training graph.
    - DINOv2LoRAEncoder: full ViT with LoRA adapters on QKV projections.
      Slower, needs RGB images online, but can adapt features to the domain.

LoRA is implemented inline (~40 lines) to avoid peft/loralib dependencies.
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around a frozen nn.Linear.

    Computes: y = W_frozen @ x + (B @ A) @ x * (alpha / rank)
    Only A and B are trainable.
    """

    def __init__(self, original: nn.Linear, rank: int=8, alpha: float=16.0):
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...

def _build_projection_head(in_dim: int, out_dim: int) -> nn.Sequential:
    """MLP projection: in_dim → out_dim with LayerNorm + GELU."""
    ...

class DINOv2FrozenEncoder(nn.Module):
    """Encoder that consumes pre-extracted DINOv2 CLS token features.

    The actual DINOv2 forward pass happens offline in extract_dino_features.py.
    This module is just a learned projection head from 768-D to ``projection_dim``.

    Interface:
        forward(features: (B, 768)) -> (B, projection_dim)
        output_dim -> projection_dim
    """

    def __init__(self, dino_model: str='dinov2_vitb14', projection_dim: int=2048):
        ...

    def forward(self, features: Tensor) -> Tensor:
        """Project pre-extracted DINOv2 CLS features.

        Args:
            features: (B, dino_dim) pre-extracted CLS token.

        Returns:
            (B, projection_dim) projected features.
        """
        ...

    @property
    def output_dim(self) -> int:
        ...

class DINOv2LoRAEncoder(nn.Module):
    """DINOv2 ViT with LoRA adapters on QKV projections.

    Takes raw RGB images, runs the full ViT forward pass (with LoRA),
    extracts CLS token, and projects to ``projection_dim``.

    Base ViT weights are frozen; only LoRA matrices + projection head train.

    Interface:
        forward(images: (B, 3, 224, 224)) -> (B, projection_dim)
        output_dim -> projection_dim
    """

    def __init__(self, dino_model: str='dinov2_vitb14', projection_dim: int=2048, lora_rank: int=8, lora_alpha: float=16.0):
        ...

    def _inject_lora(self, rank: int, alpha: float):
        """Replace each block's qkv Linear with a LoRALinear wrapper."""
        ...

    def forward(self, images: Tensor) -> Tensor:
        """Extract features from RGB images via DINOv2 + LoRA.

        Args:
            images: (B, 3, 224, 224) normalized RGB images
                    (ImageNet mean/std normalization expected).

        Returns:
            (B, projection_dim) projected features.
        """
        ...

    @property
    def output_dim(self) -> int:
        ...

def _dino_feature_dim(model_name: str) -> int:
    """Return CLS token dimension for a DINOv2 model variant."""
    ...
