"""
Temporal losses for video-based pose estimation.

Enforce smooth, physically plausible motion across frames:
    - Acceleration penalty (second-order finite differences on joints)
    - Parameter smoothness (first-order differences on pose params)
    - DCT smoothness (penalise high-frequency spectral components)
    - Rotational smoothness with anti-spin regularisation
    - Temporal friction on root depth
"""
import math
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from shark_pose.core.skeleton import ROOT_IDX
from shark_pose.core.transforms import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle

class AccelerationLoss(nn.Module):
    """Penalise joint acceleration across frames.

    Acceleration is approximated by second-order finite differences:
        a_t = x_{t-1} - 2*x_t + x_{t+1}

    Minimising acceleration encourages constant-velocity (linear) motion.

    Args:
        weight: Global scalar weight.
    """

    def __init__(self, weight: float=1.0) -> None:
        ...

    def forward(self, joints_sequence: Tensor) -> Tensor:
        """Compute acceleration loss.

        Args:
            joints_sequence: Joint positions over time, shape (B, T, J, 3).

        Returns:
            Scalar loss tensor.
        """
        ...

class SmoothnessLoss(nn.Module):
    """L2 penalty on consecutive pose parameter differences.

    Encourages smooth transitions between adjacent frames in parameter
    space (axis-angle, shape, or camera parameters).

    Args:
        weight: Global scalar weight.
    """

    def __init__(self, weight: float=1.0) -> None:
        ...

    def forward(self, params_sequence: Tensor) -> Tensor:
        """Compute parameter smoothness loss.

        Args:
            params_sequence: Parameters over time, shape (B, T, D).

        Returns:
            Scalar loss tensor.
        """
        ...

class DCTSmoothnessLoss(nn.Module):
    """DCT-based smoothness constraint.

    Penalises high-frequency components in joint trajectories by
    computing the DCT along the temporal axis and applying increasing
    weights to higher-frequency coefficients.

    Uses a type-II DCT implemented via torch.fft.

    Args:
        weight: Global scalar weight.
        freq_cutoff: Fraction of DCT coefficients considered
            'high-frequency' (0.0 to 1.0). Coefficients above this
            fraction are penalised. Default: 0.5 (top half).
    """

    def __init__(self, weight: float=1.0, freq_cutoff: float=0.5) -> None:
        ...

    @staticmethod
    def _dct_type2(x: Tensor) -> Tensor:
        """Compute type-II DCT along the last dimension using FFT.

        Args:
            x: Input tensor, DCT computed along dim=-1.

        Returns:
            DCT coefficients, same shape as input.
        """
        ...

    def forward(self, joints_sequence: Tensor) -> Tensor:
        """Compute DCT smoothness loss.

        Args:
            joints_sequence: Joint positions over time, shape (B, T, J, 3).

        Returns:
            Scalar loss tensor.
        """
        ...

class RotationalSmoothnessLoss(nn.Module):
    """Penalise angular velocity between consecutive frames.

    Computes the geodesic distance (rotation angle) between consecutive
    rotation matrices and penalises large angular changes.  Includes an
    anti-spin term that specifically penalises rotation around the camera
    optical axis (z-axis) to prevent in-plane spinning artefacts.

    Args:
        weight: Global scalar weight.
        anti_spin_weight: Additional weight for the anti-spin z-axis term.
    """

    def __init__(self, weight: float=1.0, anti_spin_weight: float=0.5) -> None:
        ...

    @staticmethod
    def _geodesic_distance(R1: Tensor, R2: Tensor) -> Tensor:
        """Compute geodesic distance between rotation matrices.

        d(R1, R2) = arccos( (trace(R1^T @ R2) - 1) / 2 )

        Args:
            R1, R2: Rotation matrices, shape (..., 3, 3).

        Returns:
            Geodesic angles, shape (...,).
        """
        ...

    def forward(self, rotmats_sequence: Tensor) -> Tensor:
        """Compute rotational smoothness loss.

        Args:
            rotmats_sequence: Per-joint rotation matrices over time,
                shape (B, T, J, 3, 3).

        Returns:
            Scalar loss tensor.
        """
        ...

class TemporalFrictionLoss(nn.Module):
    """Constrain root joint depth to be mostly stationary.

    In monocular video, depth is poorly constrained.  This loss penalises
    changes in the root joint position (especially along the depth / z-axis)
    to prevent jittery depth estimates.

    Args:
        weight: Global scalar weight.
        depth_weight: Extra weight for z-axis (depth) changes relative
            to x/y translation changes.
    """

    def __init__(self, weight: float=1.0, depth_weight: float=2.0) -> None:
        ...

    def forward(self, root_positions_sequence: Tensor) -> Tensor:
        """Compute temporal friction loss on root positions.

        Args:
            root_positions_sequence: Root joint 3D positions over time,
                shape (B, T, 3).

        Returns:
            Scalar loss tensor.
        """
        ...

def make_temporal_loss_fn(smoothness_weight: float=1.0, acceleration_weight: float=1.0, friction_weight: float=2.0, root_idx: int=ROOT_IDX) -> Callable[[Tensor, Tensor], Tensor]:
    """Compose a per-clip temporal-smoothness loss for SharkSMPLify.

    Returns a callable ``(pose (T,48), joints_3d (T,16,3)) -> scalar`` suitable
    for ``SharkSMPLify(temporal_loss_fn=...)``, where the fit's batch dimension
    is the clip's time axis. Combines three existing losses:
      - ``SmoothnessLoss`` on the pose-parameter sequence (first-order),
      - ``AccelerationLoss`` on the 3D joint trajectory (constant-velocity prior),
      - ``TemporalFrictionLoss`` on the root joint (depth-jitter prior).

    Any weight set to 0 disables that term.
    """
    ...
