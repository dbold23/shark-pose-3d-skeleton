"""
Anatomical joint-angle limit penalty.

The synthetic pose sampler already encodes how far each shark joint can plausibly
rotate (``pose_sampler.JOINT_ANGLE_LIMITS``, degrees, per axis). Fitting had no
such constraint, so SharkSMPLify was free to rotate e.g. the eye and gill joints
by 12-67 degrees against 8/12-degree limits to reconcile a landmark mismatch,
crumpling the skinned head. This hinge penalty makes those rotations expensive
while leaving anything inside the limits completely free (exactly zero loss and
zero gradient), so it cannot bias an already-plausible fit.

Axis convention: the sampler emits ``(lateral, dorsoventral, twist)`` and the
model consumes axis-angle ``[0, 1, 2]`` per joint; the two share that ordering
1:1, so the limits table maps straight onto the pose vector components.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from ..core.constants import NUM_JOINTS, POSE_DIM
from ..core.skeleton import ROOT_IDX, SHARK_KEYPOINT_SEQUENCE

def default_joint_limits_rad() -> Tensor:
    """Build the (16, 3) limit table in radians, in SHARK_KEYPOINT_SEQUENCE order."""
    ...

class JointLimitLoss(nn.Module):
    """Hinge penalty on axis-angle components exceeding their anatomical limit.

    ``L = mean_batch sum_j sum_c relu(|theta_jc| - limit_jc)^2``

    The ROOT joint (``body_midpoint_dorsal``, index 14) is EXEMPT: its rotation
    is the model's global orientation, not an articulation, and a shark may be
    seen from any heading.
    """

    def __init__(self, weight: float=1.0, limits_rad: Optional[Tensor]=None, exempt_root: bool=True):
        """
        Args:
            weight: Multiplier applied to the summed hinge penalty.
            limits_rad: (16, 3) per-joint, per-axis limits in radians. Defaults
                to ``default_joint_limits_rad()``.
            exempt_root: Exclude the root joint (global orientation) from the
                penalty. Leave True unless you are deliberately constraining
                the global heading.
        """
        ...

    def forward(self, pose: Tensor) -> Tensor:
        """
        Args:
            pose: (B, 48) axis-angle pose, or (B, 16, 3).

        Returns:
            Scalar loss (mean over the batch).
        """
        ...
