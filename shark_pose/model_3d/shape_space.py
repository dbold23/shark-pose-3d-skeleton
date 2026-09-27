"""
PCA-based shape space for the SharkSMPL model.

The shape space models inter-shark body variation (size, girth, fin proportions)
via a low-dimensional PCA basis learned from shape variations.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from ..core.constants import NUM_BETAS

class ShapeSpace(nn.Module):
    """PCA shape model: beta (10D) -> per-vertex offsets (N x 3).

    The mean shape plus PCA components define the shape space:
        V_shaped = V_mean + sum_i(beta_i * S_i)

    where S_i are the PCA components (each N x 3).
    """

    def __init__(self, mean_shape: Tensor, pca_components: Tensor, num_betas: int=NUM_BETAS):
        """
        Args:
            mean_shape: (N, 3) mean vertex positions (T-pose)
            pca_components: (num_betas, N, 3) PCA basis vectors
            num_betas: number of shape coefficients to use
        """
        ...

    def forward(self, betas: Tensor) -> Tensor:
        """Decode shape coefficients to vertex offsets.

        Args:
            betas: (batch, num_betas) shape coefficients

        Returns:
            shaped_vertices: (batch, N, 3) = mean_shape + blend
        """
        ...

    def decode_offsets(self, betas: Tensor) -> Tensor:
        """Return only the offsets from mean shape (without adding mean).

        Args:
            betas: (batch, num_betas)

        Returns:
            offsets: (batch, N, 3)
        """
        ...

    @classmethod
    def from_numpy(cls, mean_shape_path: str, pca_components_path: str, num_betas: int=NUM_BETAS) -> 'ShapeSpace':
        """Load shape space from numpy files.

        Args:
            mean_shape_path: path to .npy file with mean shape (N, 3)
            pca_components_path: path to .npy file with PCA components (K, N, 3)
            num_betas: number of shape components to use
        """
        ...

    @classmethod
    def create_random(cls, num_verts: int, num_betas: int=NUM_BETAS, scale: float=0.02, seed: int=0) -> 'ShapeSpace':
        """Create a placeholder shape space with small *non-zero* PCA components.

        Unlike :meth:`create_identity` (all-zero basis), this yields a
        non-degenerate shape space, so varying ``beta`` actually deforms the
        mesh. Use this for tests and smoke-checks that need to exercise the
        shape pathway; the basis is deterministic given ``seed``. Betas of
        zero still reproduce the mean shape exactly.
        """
        ...

    @classmethod
    def create_identity(cls, num_verts: int, num_betas: int=NUM_BETAS) -> 'ShapeSpace':
        """Create a placeholder shape space with zero PCA components.

        Useful for testing when no real shape data is available.
        """
        ...
