"""
Keypoint-based losses: 2D reprojection, 3D supervision, and structure-aware skeleton loss.

References:
    - L1-smooth (Huber) loss for robustness to outlier detections
    - Structure-aware loss uses skeleton graph topology (adjacency matrix)
      to enforce bone length and relative angle consistency under occlusion.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from shark_pose.core.skeleton import ADJACENCY_MATRIX, BONE_PAIRS, NUM_JOINTS, build_adjacency_matrix

class KeypointReprojectionLoss(nn.Module):
    """L1-smooth (Huber) loss between projected 3D joints and detected 2D joints.

    Weighted by per-joint confidence scores w_i so that low-confidence
    or occluded detections contribute less to the gradient.

    Args:
        weight: Global scalar weight for this loss term.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, weight: float=1.0, reduction: str='mean', joint_weights: Optional[Tensor]=None) -> None:
        ...

    def forward(self, pred_2d: Tensor, gt_2d: Tensor, confidence: Tensor) -> Tensor:
        """Compute confidence-weighted Huber loss on 2D keypoints.

        Args:
            pred_2d: Predicted 2D joint positions, shape (B, J, 2).
            gt_2d: Ground-truth 2D joint positions, shape (B, J, 2).
            confidence: Per-joint confidence weights, shape (B, J).

        Returns:
            Scalar loss tensor.
        """
        ...

class Keypoint3DLoss(nn.Module):
    """L1-smooth loss in 3D space (synthetic supervision only).

    Directly penalises the distance between predicted and ground-truth
    3D joint positions when full 3D annotations are available.

    Args:
        weight: Global scalar weight for this loss term.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, weight: float=1.0, reduction: str='mean', joint_weights: Optional[Tensor]=None) -> None:
        ...

    def forward(self, pred_3d: Tensor, gt_3d: Tensor) -> Tensor:
        """Compute smooth-L1 loss on 3D joint positions.

        Args:
            pred_3d: Predicted 3D joints, shape (B, J, 3).
            gt_3d: Ground-truth 3D joints, shape (B, J, 3).

        Returns:
            Scalar loss tensor.
        """
        ...

class StructureAwareLoss(nn.Module):
    """Graph-based skeleton loss preserving anatomical consistency.

    Two components:
        1. **Bone length error**: penalises discrepancies in the Euclidean
           length of each bone (edge in the kinematic tree).
        2. **Relative angle error**: penalises changes in the angle between
           adjacent bones sharing a common joint.

    The adjacency matrix from the skeleton definition is used to determine
    which joints are connected.  This loss remains informative even under
    partial occlusion because bone length ratios are viewpoint-invariant.

    Args:
        weight: Global scalar weight.
        bone_weight: Relative weight for bone-length term.
        angle_weight: Relative weight for relative-angle term.
    """

    def __init__(self, weight: float=1.0, bone_weight: float=1.0, angle_weight: float=0.5) -> None:
        ...

    def _bone_vectors(self, joints: Tensor) -> Tensor:
        """Compute bone vectors for all bone pairs.

        Args:
            joints: (B, J, D) joint positions (D = 2 or 3).

        Returns:
            (B, num_bones, D) directed bone vectors parent -> child.
        """
        ...

    def _bone_lengths(self, joints: Tensor) -> Tensor:
        """Compute Euclidean bone lengths.

        Returns:
            (B, num_bones) bone lengths.
        """
        ...

    def _bone_length_loss(self, pred_joints: Tensor, gt_joints: Tensor) -> Tensor:
        """Smooth-L1 on bone length discrepancies."""
        ...

    def _relative_angle_loss(self, pred_joints: Tensor, gt_joints: Tensor) -> Tensor:
        """Penalise changes in relative angles between adjacent bones.

        For each joint that has at least two connected bones, compute the
        cosine of the angle between each pair of incident bones and penalise
        the difference from the ground-truth angle.
        """
        ...

    def forward(self, pred_joints: Tensor, gt_joints: Tensor) -> Tensor:
        """Compute structure-aware skeleton loss.

        Args:
            pred_joints: Predicted joint positions, shape (B, J, D) where D is 2 or 3.
            gt_joints: Ground-truth joint positions, shape (B, J, D).

        Returns:
            Scalar loss tensor.
        """
        ...
