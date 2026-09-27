"""
Pose-dependent blend shapes for the SharkSMPL model.

Corrects LBS artifacts (e.g., fin fold-through during sharp turns) by adding
per-vertex offsets conditioned on joint rotation matrices.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from ..core.skeleton import NUM_JOINTS

class PoseBlendShapes(nn.Module):
    """Pose-dependent corrective blend shapes.

    Input: flattened rotation matrix elements (excluding identity) for each joint.
    Output: per-vertex offsets to add before skinning.

    Following SMPL: the pose blend shape function takes (R - I) flattened
    as input, where R are the per-joint rotation matrices.
    """

    def __init__(self, num_verts: int, num_joints: int=NUM_JOINTS, pose_blend_shapes: Tensor | None=None):
        """
        Args:
            num_verts: number of mesh vertices
            num_joints: number of joints
            pose_blend_shapes: (num_pose_basis, N, 3) pre-learned blend shapes.
                num_pose_basis = num_joints * 9 (flattened 3x3 rotation - identity)
                If None, initialized to zeros (no correction).
        """
        ...

    def forward(self, pose_rotmats: Tensor) -> Tensor:
        """Compute pose-dependent vertex offsets.

        Args:
            pose_rotmats: (batch, num_joints, 3, 3) per-joint rotation matrices

        Returns:
            offsets: (batch, N, 3) pose-dependent vertex offsets
        """
        ...
