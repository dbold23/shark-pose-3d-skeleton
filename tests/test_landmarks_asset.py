"""Tests for data/model_assets/keypoint_landmarks.npz.

The asset is produced by scripts/calibrate_landmarks.py. These tests check the
contract that SharkSMPLify / SPIN rely on: shapes, row-stochastic barycentric
weights, bilateral mirroring, and that each landmark sits at the axis station it
was calibrated for.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from calibrate_landmarks import CORRECTED_ASSET_DIR, MEASURED_OFFSETS, MIN_ABS_PCT_SPAN, MIN_LATERAL_RETENTION, MIN_SYSTEMATIC, _dorsal_shift, _mean_offset, corrections_allowed, gated_corrections, main as calibrate_main, BILATERAL_NAMES, DEFAULT_FRACTION_TARGETS, INTERIOR_NAMES, AxisFrame
from shark_pose.core.skeleton import JOINT_NAME_TO_IDX, NUM_JOINTS, SHARK_KEYPOINT_SEQUENCE
FRACTION_TOLERANCE = 0.03
SYMMETRY_TOLERANCE = 0.05

@pytest.fixture(scope='module')
def asset():
    ...

@pytest.fixture(scope='module')
def template():
    ...

@pytest.fixture(scope='module')
def bones():
    ...

@pytest.fixture(scope='module')
def axis(bones):
    ...

def test_shapes(asset, template):
    ...

@pytest.mark.parametrize('key', ['regressor', 'regressor_mirror'])
def test_row_stochastic(asset, key):
    ...

@pytest.mark.parametrize('key', ['regressor', 'regressor_mirror'])
def test_rows_are_sparse_barycentric(asset, key):
    """Each row is one vertex or one triangle -- never a dense blend."""
    ...

def test_rest_landmarks_match_regressor(asset, template):
    ...

def test_bilateral_flags(asset):
    ...

def test_midline_rows_share_the_mirror(asset):
    ...

def test_bilateral_pairs_are_symmetric(asset, template):
    ...

def test_snout_is_the_most_anterior_vertex(asset, template):
    ...

def test_frac_axis_matches_rest_landmarks(asset, axis):
    ...

def test_calibrated_fractions_within_tolerance(asset, axis):
    """Each fraction landmark sits at its own target, correction included."""
    ...

def test_frac_target_matches_stored_targets(asset):
    ...

def test_landmarks_are_monotonic_along_the_body(asset):
    """Sanity: the head landmarks precede the trunk, which precedes the tail."""
    ...

def test_interior_override(asset, bones):
    ...

def test_json_sidecar_agrees_with_npz(asset):
    ...

def test_only_confirmed_corrections_are_applied():
    """The default asset carries exactly the two corrections the ablation kept."""
    ...

def test_correction_sign_undoes_the_measured_offset():
    """A correction is the negated offset: the model is pulled onto the label."""
    ...

def test_every_applied_correction_passes_every_gate_clause():
    ...

def test_unconfirmed_corrections_are_refused_but_reachable():
    """eye_center passes clauses 1-5 and is held out only by the ablation."""
    ...

def test_offset_scale_zero_applies_nothing():
    ...

def test_scale_for_overrides_one_landmark():
    ...

def test_dorsal_shift_refuses_to_leave_the_flank():
    """Gate clause 5: a shift that lands on the throat keeps the baseline."""
    ...

@pytest.mark.skipif(not V2_NPZ_PATH.exists(), reason='data/model_assets_v2/keypoint_landmarks.npz not built')
def test_v2_asset_carries_the_correction_stamp():
    """The shipped v2 asset is the corrected one, and says so."""
    ...

def test_corrections_are_gated_on_the_destination():
    """Only data/model_assets_v2 gets the measured body-frame corrections."""
    ...

def test_bare_rebuild_reproduces_the_uncorrected_v1_arrays(tmp_path):
    """`calibrate_landmarks.py` with default flags must not correct v1.

    The v1 asset is gitignored, so a fresh checkout rebuilds it with the bare
    command; if that picked up the corrections it would silently move the frozen
    v1 fit path.
    """
    ...

def test_a_zero_offset_is_not_reported_as_a_sign_disagreement():
    """Gate clause 1 vs 3: an exactly-zero axis is rejected on magnitude."""
    ...
