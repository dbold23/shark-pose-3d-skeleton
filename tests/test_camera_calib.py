"""Lens calibration: the ideal pinhole every observation is mapped into.

The claims this file pins down are the ones the rest of the camera pass rests
on:

* ``"none"`` is genuinely inert -- it returns no camera, so every consumer
  keeps its pre-camera code path and the shipped reference fits reproduce;
* ``preset:gopro_wide`` reproduces the checked-in profile the user's
  ``marine-cv/video-distortion`` tool wrote for the demo source, so the
  re-implemented preset table has not drifted from the tool's;
* undistort -> distort is an identity to well under a tenth of a pixel over
  the WHOLE frame, which is what makes it safe to draw a fit back onto the raw
  video; and
* the proxy-space pinhole (focal AND principal point) is the letterbox image
  of ``new_K``, so the fitter, the rasteriser and the letterbox agree.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from shark_pose.sim2real.camera_calib import GOPRO_PRESETS, CameraIntrinsics, load_camera, preset_intrinsics
from shark_pose.sim2real.proxy_geometry import image_to_proxy_xy, letterbox_params
DEMO_W, DEMO_H = (2704, 1520)

@pytest.fixture(scope='module')
def demo_camera() -> CameraIntrinsics:
    ...

def _frame_grid(width: int, height: int, nx: int=41, ny: int=31) -> np.ndarray:
    ...

def test_none_spec_is_no_camera():
    """"none" must return None so callers keep their pre-camera behaviour."""
    ...

def test_zero_distortion_camera_is_the_identity():
    """A lens with no distortion moves no point, in either direction."""
    ...

def test_preset_gopro_wide_matches_the_tool_profile(demo_camera):
    """The re-implemented preset table reproduces video-distortion's output."""
    ...

def test_preset_spec_and_bad_specs():
    ...

def test_calibration_json_must_carry_a_size(tmp_path):
    ...

def test_undistort_distort_round_trip_over_the_frame(demo_camera):
    """The inverse is exact to far better than the 0.05 px the pipeline needs.

    OpenCV's DEFAULT inverse solve (5 fixed-point steps) leaves 0.48 px here,
    which is why camera_calib runs undistortPointsIter to convergence.
    """
    ...

def test_undistortion_straightens_a_line(demo_camera):
    """The plumb-line property, and it is the whole reason for this module.

    A straight line in the ideal pinhole bows visibly once the lens is applied
    -- which is what makes a straight shark read as bent -- and undistorting
    that bowed curve straightens it to floating-point noise.
    """
    ...

def test_alpha_one_keeps_every_source_pixel_in_frame(demo_camera):
    """alpha=1.0: nothing observable is cropped by the undistortion."""
    ...

def test_non_finite_points_pass_through(demo_camera):
    ...

def test_undistort_image_agrees_with_undistort_points(demo_camera):
    """The mask remap and the keypoint map are the same transform.

    A single lit pixel is remapped and its centre of mass compared with where
    ``distort_points`` says the source pixel comes from -- i.e. the maps are
    consistent inverses, which is what keeps a mask and a keypoint on the same
    animal.
    """
    ...

def test_undistort_image_rejects_the_wrong_size(demo_camera):
    ...

def test_proxy_focal_and_principal_point_are_the_letterbox_of_new_K(demo_camera):
    ...

def test_proxy_principal_point_is_not_assumed_central():
    """A camera whose principal point is off-centre must say so.

    ``getOptimalNewCameraMatrix`` is free to move it, and a fitter told
    ``res / 2`` would silently absorb the offset into the translation.
    """
    ...

def test_rescale_to_another_encode_of_the_same_recording(demo_camera):
    ...

def test_rescale_refuses_a_different_aspect_ratio(demo_camera):
    ...

def test_load_camera_rescales_a_json_to_the_frame_size(tmp_path):
    ...

def test_stamp_carries_the_provenance(demo_camera):
    ...

def test_intrinsics_validation():
    ...
