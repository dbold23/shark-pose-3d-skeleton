"""
Adversarial mesh discriminator for anatomical plausibility.

Forces the encoder to output shapes on the manifold of real sharks,
preventing anatomically impossible "monster" outputs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from ..core.constants import NUM_JOINTS, NUM_BETAS

class SharkMeshDiscriminator(nn.Module):
    """Discriminator for 3D shark mesh plausibility.

    Input: per-joint rotation matrices (flattened) + shape parameters.
    Output: real/fake score.

    Applied at EVERY iteration of the IEF regressor (not just final output)
    to keep predictions on the valid shark manifold throughout refinement.
    """

    def __init__(self, num_joints: int=NUM_JOINTS, num_betas: int=NUM_BETAS, hidden_dims: list[int] | None=None, use_spectral_norm: bool=True):
        ...

    def forward(self, rotmats: Tensor, betas: Tensor) -> Tensor:
        """Score the plausibility of a shark pose + shape.

        Args:
            rotmats: (batch, 16, 3, 3) per-joint rotation matrices
            betas: (batch, 10) shape coefficients

        Returns:
            score: (batch, 1) — higher = more plausible
        """
        ...

class PerJointDiscriminator(nn.Module):
    """Per-joint discriminator that scores each joint's rotation independently.

    More fine-grained than the global discriminator — can identify
    which specific joint is anatomically implausible.
    """

    def __init__(self, num_joints: int=NUM_JOINTS, hidden_dim: int=256):
        ...

    def forward(self, rotmats: Tensor) -> dict:
        """
        Args:
            rotmats: (batch, 16, 3, 3)

        Returns:
            dict with per_joint_scores (B, 16) and global_score (B, 1)
        """
        ...
