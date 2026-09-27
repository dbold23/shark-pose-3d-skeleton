"""
Iterative Error Feedback (IEF) regressor for SPIN.

Follows Kanazawa et al. (HMR) and Kolotouros et al. (SPIN): the regressor
takes image features concatenated with the current parameter estimate and
outputs a *residual* update. This is repeated for NUM_REGRESSOR_ITERATIONS
(default 3) iterations, progressively refining the estimate.

Pose is parameterized in the 6D continuous rotation representation
(Zhou et al. 2019) during regression for gradient stability, then converted
to axis-angle before being passed to SharkSMPL.

Each iteration also predicts a per-joint matrix-Fisher concentration
scalar kappa_j, which quantifies rotational uncertainty.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..core.constants import CAMERA_DIM, NUM_BETAS, NUM_JOINTS, NUM_REGRESSOR_ITERATIONS, POSE_DIM, SHAPE_DIM, PARAM_DIM
from ..core.transforms import rotation_6d_to_matrix, rotation_matrix_to_6d, rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix

def focal_to_feature(focal: Tensor) -> Tensor:
    """Map per-sample focal lengths in proxy px to the regressor's input scalar.

    ``log(focal / FOCAL_REFERENCE_PX)``: zero at the pipeline pinhole, and
    linear in the quantity that actually matters, since the projected size of a
    body of extent L at depth z is ``focal * L / z`` -- a ratio, so its
    logarithm is additive in the focal.

    Args:
        focal: (B,) or (B, 1) focal lengths in proxy pixels.

    Returns:
        (B, 1) focal feature.
    """
    ...

class IterativeRegressor(nn.Module):
    """Iterative Error Feedback regressor.

    At each iteration:
        1. Concatenate image features (2048) with current parameter vector (61).
        2. Pass through two FC layers with batch norm, ReLU, dropout, and
           residual connections.
        3. Output residual delta in [pose_6d (96), shape (10), camera (3)] = 109D,
           plus per-joint kappa (16) for uncertainty.
        4. Accumulate: params_{t+1} = params_t + delta_t.

    The initial parameter estimate (mean pose/shape) is learnable.
    """

    def __init__(self, feature_dim: int=2048, hidden_dim: int=1024, num_iterations: int=NUM_REGRESSOR_ITERATIONS, dropout: float=0.5, focal_conditioned: bool=False):
        """
        Args:
            feature_dim: Dimensionality of encoder features (2048 for ResNet-50).
            hidden_dim: Hidden layer size in the FC blocks.
            num_iterations: Number of IEF iterations.
            dropout: Dropout probability.
            focal_conditioned: Append one scalar per sample --
                ``log(focal / FOCAL_REFERENCE_PX)`` -- to the regressor input.
                The regressed translation is otherwise expressed in the units of
                whatever single focal the training projection used, so a network
                trained across a focal range cannot tell which depth a given
                apparent size implies.  Default False keeps the v2 architecture
                exactly.
        """
        ...

    def forward(self, features: Tensor, init_pose: Optional[Tensor]=None, init_shape: Optional[Tensor]=None, init_camera: Optional[Tensor]=None, focal: Optional[Tensor]=None) -> Dict[str, Tensor | List[Dict[str, Tensor]]]:
        """Run iterative regression.

        Args:
            features: (B, 2048) image features from ProxyEncoder.
            init_pose: (B, 48) initial axis-angle pose. If None, use learned mean.
            init_shape: (B, 10) initial shape betas. If None, use zeros.
            init_camera: (B, 3) initial camera (tx, ty, tz). If None, use learned mean.
            focal: (B,) per-sample focal length in proxy pixels. Only accepted
                when the regressor is focal-conditioned; None then means the
                reference focal, whose feature is exactly zero.

        Returns:
            Dictionary containing:
                pose: (B, 48) final axis-angle pose
                shape: (B, 10) final shape betas
                camera: (B, 3) final camera translation
                pose_6d: (B, 96) final 6D rotation pose
                kappa: (B, 16) final per-joint concentration (uncertainty)
                per_iteration: list of dicts with intermediate outputs
        """
        ...

    def _axis_angle_to_6d_flat(self, pose_aa: Tensor) -> Tensor:
        """Convert (B, 48) axis-angle to (B, 96) flat 6D rotation.

        Args:
            pose_aa: (B, NUM_JOINTS*3) axis-angle per joint.

        Returns:
            pose_6d: (B, NUM_JOINTS*6) 6D rotation per joint.
        """
        ...

    def _6d_flat_to_axis_angle(self, pose_6d: Tensor) -> Tensor:
        """Convert (B, 96) flat 6D rotation to (B, 48) axis-angle.

        Args:
            pose_6d: (B, NUM_JOINTS*6) 6D rotation per joint.

        Returns:
            pose_aa: (B, NUM_JOINTS*3) axis-angle per joint.
        """
        ...
