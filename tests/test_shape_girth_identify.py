"""TASK F1 -- the shape prior per component, and the girth observation.

Three changes, each of which must be INERT at its default:

* ``shape_prior_scale`` / ``shape_freeze_components`` -- the shape prior read
  component by component, because the shipped basis is not one family of modes
  (0-3 structured, 4-9 the generator's isotropic noise floor);
* ``width_align_stations`` -- the width target read at the arc fraction the
  MODEL's own projected station falls on, instead of at the record's, which is
  the station mismatch measured in
  ``outputs/demo_2026-09-03/work/identify/girth/align_ref.json``;
* ``girth_prior_mean`` -- where the girth prior pulls to, because the width
  series sees one of the two axes and the other one reports its prior.

The bit-identity tests are the load-bearing ones: every default must take the
ORIGINAL code path, not a scaled version of it, so a fit that sets none of
these is unchanged to the last bit.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.encoder.smplify_fitting import SpineSMPLify
from shark_pose.model_3d import SharkSMPL, girth_prior_loss
from shark_pose.model_3d.shark_smpl import select_visible_landmarks

@pytest.fixture(scope='module')
def model_v2() -> SharkSMPL:
    ...

def _frames(model, fitter, n=3, seed=0):
    """A synthetic clip: the rest body placed in front of the camera."""
    ...

def _loss_of(fitter, model, n=3, **leaf_kw):
    """The objective at a fixed set of leaves, for a bit-identity comparison."""
    ...

def test_shape_prior_defaults_are_bit_identical(model_v2):
    """`shape_prior_scale=None` must run the ORIGINAL expression, not ones."""
    ...

def test_unit_shape_prior_scale_matches_the_shipped_prior(model_v2):
    """A scale of all-ones is the shipped prior -- to float tolerance."""
    ...

def test_shape_prior_scale_moves_only_its_own_components(model_v2):
    """Zeroing 4-9 removes exactly their share of the prior."""
    ...

def test_girth_prior_defaults_are_bit_identical(model_v2):
    ...

def test_girth_prior_mean_of_one_matches_the_shipped_prior(model_v2):
    ...

def test_width_align_defaults_to_off(model_v2):
    ...

def test_shape_prior_scale_validates(model_v2):
    ...

def test_frozen_components_never_move(model_v2):
    """A frozen beta comes back EXACTLY at its init; a free one moves."""
    ...

def test_girth_prior_mean_pulls_the_field_to_the_mean(model_v2):
    """With no observation at all the field goes where the prior says.

    The confidences are zero on purpose.  The 16 landmarks are points on the
    mesh, so they DO see the girth field (a fatter body moves the pectoral and
    dorsal tips), and against keypoints synthesised at girth = 1 they would
    simply pull it back to 1 -- which is the correct behaviour and not what
    this test is about.
    """
    ...

def test_girth_tie_is_zero_at_the_prior_mean(model_v2):
    """The rescaled tie must not fight an anisotropic prior."""
    ...

def test_girth_prior_mean_validates(model_v2):
    ...

def test_aligned_target_reads_the_profile_at_the_model_station(model_v2):
    """A ramp profile: the aligned target must track the model's own station.

    The polyline is a straight horizontal line and the profile a linear ramp
    along it, so the target at each station is a direct read-out of where that
    station lands -- which is exactly what the alignment is for.
    """
    ...

def test_alignment_changes_the_target_and_the_loss(model_v2):
    """Aligned and record-station targets are different numbers."""
    ...

def test_alignment_warns_and_falls_back_without_a_profile(model_v2):
    ...
