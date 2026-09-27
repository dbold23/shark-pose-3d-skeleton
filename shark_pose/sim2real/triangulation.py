"""
Multi-view triangulation for stereo/orthogonal camera setups.

Given 2D keypoints detected in two synchronized camera views and the
cameras' calibration (intrinsics + extrinsics), recover 3D joint
positions via the Direct Linear Transform (DLT) method.

This is the path to real 3D ground truth from captive shark footage
filmed with two cameras (e.g., the Del Raye 2013 orthogonal setup).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraCalibration:
    """Calibration for a single camera.

    Args:
        K: (3, 3) intrinsic matrix (fx, fy, cx, cy).
        R: (3, 3) rotation matrix (world -> camera).
        t: (3,) translation vector (camera position in world frame is -R^T @ t).
    """
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def projection_matrix(self) -> np.ndarray:
        """(3, 4) projection matrix P = K [R | t]."""
        ...

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project (N, 3) world-space 3D points to (N, 2) pixel coordinates."""
        ...

def triangulate_dlt(points_2d_view1: np.ndarray, points_2d_view2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Triangulate matched 2D points across two views via DLT.

    For each point pair (x1, x2), solves the linear system A @ X = 0 where:
        A = [x1[0]*P1[2] - P1[0],
             x1[1]*P1[2] - P1[1],
             x2[0]*P2[2] - P2[0],
             x2[1]*P2[2] - P2[1]]

    Then takes the right-singular vector of the smallest singular value
    of A as the homogeneous 3D point.

    Args:
        points_2d_view1: (N, 2) 2D points from camera 1.
        points_2d_view2: (N, 2) 2D points from camera 2.
        P1: (3, 4) projection matrix for camera 1.
        P2: (3, 4) projection matrix for camera 2.

    Returns:
        (N, 3) triangulated 3D points in world space.
    """
    ...

def triangulate_keypoints(kp_view1: np.ndarray, kp_view2: np.ndarray, cam1: CameraCalibration, cam2: CameraCalibration, confidence_view1: np.ndarray | None=None, confidence_view2: np.ndarray | None=None, min_confidence: float=0.3) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate matched keypoints with optional confidence-based masking.

    Args:
        kp_view1: (J, 2) joint pixel coordinates in view 1.
        kp_view2: (J, 2) joint pixel coordinates in view 2.
        cam1: calibration for view 1.
        cam2: calibration for view 2.
        confidence_view1: optional (J,) per-joint confidence in view 1.
        confidence_view2: optional (J,) per-joint confidence in view 2.
        min_confidence: joints with conf below this in either view are
            returned as NaN.

    Returns:
        joints_3d: (J, 3) triangulated 3D joint positions, NaN for joints
            below the confidence threshold in either view.
        valid_mask: (J,) bool, True where the triangulation is valid.
    """
    ...

def reprojection_error(points_3d: np.ndarray, points_2d: np.ndarray, cam: CameraCalibration) -> np.ndarray:
    """Per-point reprojection error in pixels.

    Useful as a sanity check on triangulation quality. If the triangulation
    is geometrically consistent, projecting the 3D point back into either
    camera should land near the original 2D detection.

    Args:
        points_3d: (N, 3) triangulated 3D points.
        points_2d: (N, 2) original 2D detections in this camera.
        cam: calibration for the camera.

    Returns:
        (N,) per-point reprojection error in pixels (NaN where points_3d is NaN).
    """
    ...

def calibration_from_synthetic_camera(focal_length: float, image_size: int, R: np.ndarray, t: np.ndarray) -> CameraCalibration:
    """Build a CameraCalibration from synthetic dataset camera params.

    The synthetic data generator stores:
        focal_length (pixels at proxy resolution),
        R (3, 3) world->camera rotation,
        t (3,) world->camera translation.

    Args:
        focal_length: focal length in pixels.
        image_size: square image dimension (typically 256).
        R: (3, 3) rotation.
        t: (3,) translation.

    Returns:
        CameraCalibration with K = diag(f, f, 1) + (cx, cy) at image center.
    """
    ...
