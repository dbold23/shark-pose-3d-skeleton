"""Tests for observation landmarks and the anatomical joint-limit penalty.

The calibrated asset (``data/model_assets/keypoint_landmarks.npz``) is produced
by the rig-calibration step; these tests never depend on it. They build a
SYNTHETIC stand-in (nearest mesh vertex to each rig bone, mirrored across the
sagittal plane) into a throwaway copy of the assets directory, so they exercise
the same code path whether or not the real asset exists yet.
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.core.constants import NUM_BETAS, POSE_DIM
from shark_pose.core.skeleton import JOINT_NAME_TO_IDX, NUM_JOINTS, ROOT_IDX, SHARK_KEYPOINT_SEQUENCE
from shark_pose.encoder.smplify_fitting import SharkSMPLify
from shark_pose.losses import JointLimitLoss, default_joint_limits_rad
from shark_pose.model_3d.shark_smpl import SharkSMPL, select_visible_landmarks

def _require_assets():
    ...

def _assets_without_landmarks(tmp_path: Path) -> Path:
    """Copy of the real assets dir guaranteed NOT to contain the landmark asset."""
    ...

def _assets_with_stand_in(tmp_path: Path) -> Path:
    """Copy of the assets dir plus a synthetic keypoint_landmarks.npz.

    Each landmark is the mesh vertex nearest the corresponding rig bone; the
    mirror instance is the vertex nearest that bone reflected across x=0. This
    is deliberately crude -- it only has to have the right SHAPE and semantics.
    """
    ...

class TestVisibleSelection:

    def test_picks_nearer_instance_for_bilateral_rows(self):
        ...

    def test_picks_left_when_left_is_nearer(self):
        ...

    def test_selection_mask_is_detached(self):
        """Gradient flows through the chosen values, never through the choice."""
        ...

class TestSharkSMPLLandmarks:

    def test_landmarks_equal_joints_without_asset(self, tmp_path):
        ...

    def test_placeholder_model_has_no_landmarks(self):
        ...

    def test_landmarks_loaded_from_asset(self, tmp_path):
        ...

    def test_visible_landmarks_shape(self, tmp_path):
        ...

class TestJointLimitLoss:

    def test_zero_inside_limits(self):
        ...

    def test_positive_and_quadratic_outside(self):
        ...

    def test_root_joint_exempt(self):
        ...

    def test_weight_scales_loss(self):
        ...

    def test_accepts_both_pose_shapes(self):
        ...

    def test_limits_match_pose_sampler(self):
        ...

class TestSMPLifyWithLandmarks:

    def test_fit_lowers_loss(self, tmp_path):
        ...

    def test_joint_term_penalizes_out_of_range_pose(self, tmp_path):
        ...

    def test_landmark_and_joint_targets_differ(self, tmp_path):
        """use_landmarks actually changes what gets projected."""
        ...
