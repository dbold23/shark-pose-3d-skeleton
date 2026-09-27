"""
Mesh-based losses: per-vertex supervision, edge-length preservation, and collision penalty.

The collision loss uses a simplified signed-distance-field (SDF) approach
to penalise interpenetration between fin vertex groups and the body surface.
It is enabled only during Stage 2 training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MeshVertexLoss(nn.Module):
    """Per-vertex L2 loss (synthetic supervision only).

    Computes the mean squared Euclidean distance between predicted and
    ground-truth mesh vertices.

    Args:
        weight: Global scalar weight for this loss term.
    """

    def __init__(self, weight: float=1.0) -> None:
        ...

    def forward(self, pred_verts: Tensor, gt_verts: Tensor) -> Tensor:
        """Compute per-vertex L2 loss.

        Args:
            pred_verts: Predicted vertices, shape (B, V, 3).
            gt_verts: Ground-truth vertices, shape (B, V, 3).

        Returns:
            Scalar loss tensor.
        """
        ...

class MeshEdgeLoss(nn.Module):
    """Edge-length preservation loss.

    Penalises changes in mesh edge lengths between predicted and ground-truth
    meshes to maintain surface regularity.

    Args:
        weight: Global scalar weight for this loss term.
    """

    def __init__(self, weight: float=1.0) -> None:
        ...

    @staticmethod
    def _edges_from_faces(faces: Tensor) -> Tensor:
        """Extract unique undirected edges from a face tensor.

        Args:
            faces: Triangle face indices, shape (F, 3) with dtype long.

        Returns:
            Unique edges, shape (E, 2) with dtype long.
        """
        ...

    def forward(self, pred_verts: Tensor, gt_verts: Tensor, faces: Tensor) -> Tensor:
        """Compute edge-length preservation loss.

        Args:
            pred_verts: Predicted vertices, shape (B, V, 3).
            gt_verts: Ground-truth vertices, shape (B, V, 3).
            faces: Triangle face indices, shape (F, 3), dtype long.

        Returns:
            Scalar loss tensor.
        """
        ...

class CollisionLoss(nn.Module):
    """Fin-specific interpenetration penalty using simplified SDF.

    Computes a one-sided penalty when fin vertices penetrate the body
    surface.  The body surface is represented by a set of oriented
    triangles (from the mesh faces excluding fin regions).  For each fin
    vertex we compute a simplified signed distance: negative values
    indicate penetration, and we penalise max(0, -sdf).

    Stage usage:
        - Stage 1 (synthetic pre-train): OFF
        - Stage 2 (real fine-tune): ON
        - Stage 3 / inference: OFF

    Args:
        weight: Global scalar weight.
        enabled: Whether the loss is active.  Set to False to disable
            without removing from the loss dict.
    """

    def __init__(self, weight: float=1.0, enabled: bool=False) -> None:
        ...

    def _compute_sdf(self, vertices: Tensor, faces: Tensor, fin_vertex_ids: Tensor) -> Tensor:
        """Simplified signed-distance field for fin vertices.

        For each fin vertex, find the closest body triangle and compute
        the signed distance using the triangle normal.

        Args:
            vertices: Full mesh vertices, shape (B, V, 3).
            faces: Body-surface face indices, shape (F_body, 3), dtype long.
            fin_vertex_ids: Indices of fin vertices, shape (N_fin,), dtype long.

        Returns:
            Signed distances for fin vertices, shape (B, N_fin).
            Negative = penetration.
        """
        ...

    def forward(self, vertices: Tensor, faces: Tensor, fin_vertex_ids: Tensor) -> Tensor:
        """Compute fin interpenetration penalty.

        Args:
            vertices: Full mesh vertices, shape (B, V, 3).
            faces: Body-surface face indices (excluding fins), shape (F_body, 3).
            fin_vertex_ids: Indices of fin vertices, shape (N_fin,).

        Returns:
            Scalar loss tensor (0 if disabled).
        """
        ...
