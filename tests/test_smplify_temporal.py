"""Tests for the per-clip temporal-coupling hook in SharkSMPLify.

The hook (temporal_weight + temporal_loss_fn) treats a fit()'s batch dimension
as TIME so a whole clip is fit jointly with smoothness across frames. Guards:
  - temporal_weight=0 must leave the single-frame/non-temporal fit unchanged.
  - with temporal coupling, the fitted joint trajectory must be SMOOTHER
    (lower acceleration) than independent per-frame fitting.
"""
from __future__ import annotations
from pathlib import Path
import pytest
import torch
from shark_pose.model_3d.shark_smpl import SharkSMPL
from shark_pose.encoder.smplify_fitting import SharkSMPLify
from shark_pose.losses.temporal_loss import make_temporal_loss_fn

def _make_clip(T=12, seed=0):
    """A short clip: one joint sweeps smoothly over time; sparse noisy 2D obs."""
    ...

def _joint_accel(model, pose, shape, cam_t):
    ...

def test_temporal_weight_zero_matches_no_temporal():
    """The guard must make temporal_weight=0 identical to a plain fitter."""
    ...

def test_temporal_coupling_reduces_trajectory_acceleration():
    """Joint-over-clip fitting with temporal coupling should be smoother than
    independent per-frame fitting, without wrecking reprojection."""
    ...
