"""Tests for stereo triangulation."""
import numpy as np
import pytest
from shark_pose.sim2real.triangulation import CameraCalibration, calibration_from_synthetic_camera, reprojection_error, triangulate_dlt, triangulate_keypoints

def _make_camera(position: np.ndarray, look_at: np.ndarray=np.zeros(3), focal: float=500.0, image_size: int=256) -> CameraCalibration:
    """Build a camera calibration from a position + look-at point."""
    ...

class TestTriangulationDLT:

    def test_two_orthogonal_views_recovers_3d(self):
        """Place two cameras at orthogonal positions, triangulate known points."""
        ...

    def test_baseline_stereo_recovers_3d(self):
        """Two cameras with horizontal baseline (typical stereo)."""
        ...

    def test_triangulation_robust_to_small_noise(self):
        """1-pixel noise in detections produces small 3D error."""
        ...

class TestTriangulateKeypoints:

    def test_confidence_masks_out_low_conf_joints(self):
        ...

class TestReprojectionError:

    def test_perfect_triangulation_zero_reproj_error(self):
        ...

class TestCameraCalibration:

    def test_projection_matrix_shape(self):
        ...

    def test_project_round_trip(self):
        """Projecting the camera origin in world space should give principal point."""
        ...
