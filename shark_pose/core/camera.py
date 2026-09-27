"""
Camera models and projection utilities.

All 3D estimation operates in camera coordinate frame (not world space).
Viewpoint angles encoded as (sin(theta), cos(theta)) for feature-space continuity.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from .transforms import angle_to_viewpoint_vector, viewpoint_vector_to_angle

class PerspectiveCamera:
    """Perspective projection camera operating in camera coordinate frame.

    Attributes:
        focal_length: (batch, 2) or scalar — (fx, fy)
        principal_point: (batch, 2) — (cx, cy), defaults to image center
        image_size: (H, W) resolution
    """

    def __init__(self, focal_length: float | Tensor=1000.0, principal_point: Tensor | None=None, image_size: tuple[int, int]=(256, 256)):
        ...

    def project(self, points_3d: Tensor) -> Tensor:
        """Project 3D points in camera frame to 2D pixel coordinates.

        Args:
            points_3d: (batch, N, 3) points in camera coordinate frame

        Returns:
            points_2d: (batch, N, 2) pixel coordinates
        """
        ...

    def to(self, device: torch.device) -> 'PerspectiveCamera':
        ...

class WeakPerspectiveCamera:
    """Weak-perspective (scaled orthographic) projection.

    Parameters: scale s, translation (tx, ty).
    Projects as: p_2d = s * (p_3d[:2]) + t
    """

    @staticmethod
    def project(points_3d: Tensor, camera_params: Tensor) -> Tensor:
        """Project 3D joints using weak perspective.

        Args:
            points_3d: (batch, N, 3)
            camera_params: (batch, 3) — [scale, tx, ty]

        Returns:
            points_2d: (batch, N, 2)
        """
        ...

def world_to_camera(points_world: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """Transform points from world coordinate frame to camera frame.

    Args:
        points_world: (batch, N, 3)
        R: (batch, 3, 3) camera rotation (world -> camera)
        t: (batch, 3) camera translation

    Returns:
        points_camera: (batch, N, 3)
    """
    ...

def camera_to_world(points_camera: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """Transform points from camera coordinate frame to world frame.

    Args:
        points_camera: (batch, N, 3)
        R: (batch, 3, 3) camera rotation (world -> camera)
        t: (batch, 3) camera translation

    Returns:
        points_world: (batch, N, 3)
    """
    ...

def estimate_camera_translation(joints_2d: Tensor, joints_3d: Tensor, focal_length: float=1000.0, image_center: tuple[float, float] | None=None, image_size: tuple[int, int]=(256, 256)) -> Tensor:
    """Estimate camera translation (tx, ty, tz) from 2D-3D correspondences.

    Uses a closed-form weak-perspective approximation:
    tz ≈ focal_length * bone_length_3d / bone_length_2d

    Args:
        joints_2d: (batch, N, 2) detected 2D joints
        joints_3d: (batch, N, 3) predicted 3D joints (in root-relative coords)
        focal_length: camera focal length in pixels
        image_center: (cx, cy), defaults to image center

    Returns:
        translation: (batch, 3)
    """
    ...

def soft_argmax_2d(heatmaps: Tensor, temperature: float=1.0) -> Tensor:
    """Differentiable soft arg-max to extract (x, y) from heatmaps.

    Provides continuous gradients — strictly better than hard argmax.

    Args:
        heatmaps: (batch, num_joints, H, W)
        temperature: softmax temperature (lower = sharper)

    Returns:
        coords: (batch, num_joints, 2) — (x, y) coordinates
    """
    ...
