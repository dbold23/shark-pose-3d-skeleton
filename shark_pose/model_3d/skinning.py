"""
Linear Blend Skinning (LBS) for the SharkSMPL model.

Follows the SMPL paper exactly:
  rest pose -> shape blend -> pose blend -> world transform per joint -> skinning
"""
from __future__ import annotations
import torch
from torch import Tensor
from ..core.skeleton import NUM_JOINTS, PARENT_INDICES

def _topological_sort(parents: list[int], num_joints: int) -> list[int]:
    """Return joint indices in topological order (parents before children)."""
    ...

def lbs(vertices: Tensor, pose_rotmats: Tensor, lbs_weights: Tensor, joint_locations: Tensor, parents: list[int] | None=None) -> tuple[Tensor, Tensor]:
    """Apply Linear Blend Skinning to transform vertices from rest pose to posed configuration.

    Args:
        vertices: (batch, N, 3) vertices in rest pose (after shape + pose blend shapes)
        pose_rotmats: (batch, num_joints, 3, 3) per-joint local rotation matrices
        lbs_weights: (N, num_joints) skinning weights (how much each joint affects each vertex)
        joint_locations: (batch, num_joints, 3) joint positions in rest pose
        parents: parent index per joint (-1 for root). Defaults to PARENT_INDICES.

    Returns:
        posed_vertices: (batch, N, 3) vertices in posed configuration
        posed_joints: (batch, num_joints, 3) joint positions after posing
    """
    ...
