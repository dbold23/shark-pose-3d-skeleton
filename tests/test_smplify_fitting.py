"""Regression tests for SharkSMPLify fitting.

Guards the 2026-07-08 bug: the pose prior regularized toward the *zero/rest*
pose (while shape regularized toward its init), so under sparse real-world
visibility (~5/16 joints) the optimizer collapsed a good non-zero pose toward
rest and *raised* reprojection error (66 -> 114px on real data). The fix
regularizes pose toward the incoming estimate and sharpens the data term.
"""
from __future__ import annotations
from pathlib import Path
import pytest
import torch
from shark_pose.model_3d.shark_smpl import SharkSMPL
from shark_pose.encoder.smplify_fitting import SharkSMPLify

def _setup():
    ...

def _visible_reproj(model, cam, pose, shape, cam_t, obs, conf):
    ...

def test_smplify_reduces_error_from_good_init_under_sparse_visibility():
    """The real Stage 3 condition: good non-zero init, only ~5/16 joints visible.
    Fitting must NOT increase visible-joint reprojection error."""
    ...

def test_smplify_does_not_explode_unobserved_joints():
    """With a rest-pose init and sparse observations, unobserved joints have no
    data signal and must stay bounded (regularized toward init), not diverge."""
    ...

def test_pose_prior_regularizes_toward_init_not_zero():
    """Direct check of the fix: with a good init and NO visible keypoints, the
    fit should stay at the init pose (not collapse toward zero)."""
    ...
