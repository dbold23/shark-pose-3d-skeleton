"""Tests for the sevengill-derived swimming prior (shark_pose/model_3d/spine_prior.py).

Two groups: contract tests that build small synthetic gain tables and always
run, and asset tests that check the numbers in
``data/model_assets_v2/motion_prior.npz`` (skipped when it has not been built).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.core.transforms import axis_angle_to_rotation_matrix
from shark_pose.model_3d.spine_prior import DEFAULT_AMP, HEAD_JOINT_SLOTS, NUM_SPINE_SEGMENTS, NUM_V2_JOINTS, PARAM_DIM, SPINE_PRIOR_VERSION, V2_FIN_JOINTS, V2_JOINT_NAMES, V2_SPINE_JOINTS, SpinePoseParams, SpinePrior, canonical_key, resolve_joint_order, ypr_to_axis_angle

def synthetic_tables() -> dict[str, np.ndarray]:
    """Small hand-made gain tables with the same layout as the real asset."""
    ...

@pytest.fixture
def prior() -> SpinePrior:
    ...

@pytest.fixture
def asset_prior() -> SpinePrior:
    ...

@pytest.fixture
def report() -> dict:
    ...

def test_rig_spec_is_13_spine_plus_18_fins():
    ...

def test_resolve_joint_order_identity():
    ...

def test_resolve_joint_order_handles_renaming_and_shuffling():
    ...

def test_resolve_joint_order_rejects_incomplete_rig():
    ...

def test_canonical_key_ignores_foreign_bones():
    ...

def test_ypr_to_axis_angle_matches_rz_ry_rx():
    ...

def test_ypr_to_axis_angle_is_differentiable_at_zero():
    ...

def test_to_axis_angle_shapes(prior):
    ...

def test_clip_matches_per_frame_calls(prior):
    ...

def test_clip_accepts_per_frame_turn(prior):
    ...

def test_phase_advances_with_frequency(prior):
    ...

def test_param_vector_roundtrip(prior):
    ...

def test_gradients_flow_to_every_parameter(prior):
    ...

def test_limits_enforce_head_rigidity(prior):
    ...

def test_clamp_is_a_noop_at_default_cruise(prior):
    ...

def test_prior_loss_is_zero_at_cruise_defaults(prior):
    ...

def test_prior_loss_grows_away_from_defaults(prior):
    ...

def test_prior_loss_prefers_damped_amplitude(prior):
    ...

def test_prior_loss_is_laplace_in_turn(prior):
    ...

def test_asset_wave_fit_r2_on_trunk_segments(asset_prior):
    ...

def test_asset_amplitude_grows_towards_the_tail(report):
    ...

def test_asset_phase_lags_travel_backwards(report):
    ...

def test_asset_head_limits_are_tiny(asset_prior):
    ...

def test_asset_turn_shape_is_normalised_and_monotone(report):
    ...

def test_asset_escape_is_a_c_start(report):
    ...

def test_asset_rest_residual_is_small(report):
    ...

def test_asset_cruise_cycle_loops(asset_prior):
    ...

def test_asset_cruise_cycle_matches_the_animation_amplitude(asset_prior, report):
    ...

def test_asset_default_amplitude_damps_the_animation(asset_prior):
    ...

def test_asset_midline_stays_straight_at_rest(asset_prior):
    ...

def test_asset_midline_bends_more_at_full_amplitude(asset_prior):
    ...

def test_asset_reordered_prior_matches_slot_order(asset_prior):
    ...

@pytest.fixture
def sevengill() -> dict:
    ...

def test_asset_kappa_reproduces_the_turn_offsets(asset_prior, report):
    ...

def test_asset_escape_one_reproduces_the_peak_bend(asset_prior, report):
    ...

def test_asset_root_segment_carries_no_wave(asset_prior, report):
    """Neither chain end can carry a bend: station 0 is a leaf on the v2 chain
    (the cranium) and station 5 is the kinematic root."""
    ...

def test_asset_basis_is_a_proper_rotation(asset_prior):
    ...

def test_asset_basis_columns_are_along_bone_lateral_vertical(asset_prior):
    """The spine runs snout(+y) -> tail(-y), so along-bone is ~-y, lateral ~+x,
    vertical ~+z in the model frame."""
    ...

def test_asset_local_axes_map_to_model_axes(asset_prior):
    """yaw -> model z (lateral bend), pitch -> model x (dorsoventral),
    roll -> model -y (about the long axis).  Before the basis change pitch
    rotated about the long axis, i.e. it was a roll."""
    ...

def test_missing_basis_falls_back_to_identity():
    """Hand-made tables (these tests) may omit the frame metadata."""
    ...

def test_a_built_asset_without_the_frame_metadata_is_refused():
    """data/ is gitignored: a pre-fix asset must not load silently.

    Without this the identity-basis fallback puts the anterior wave back on the
    head and turns "pitch" into a roll about the body's long axis.
    """
    ...

@pytest.mark.skipif(not ASSET.exists(), reason='motion_prior.npz not built')
def test_the_shipped_asset_carries_its_provenance():
    ...

def test_basis_travels_with_a_reordered_rig(asset_prior):
    ...

def _local_yaw_deg(prior: SpinePrior, aa_model) -> np.ndarray:
    """The 13 spine joints' LOCAL yaw, in degrees, from a ``(1, ..., 31, 3)`` pose.

    Yaw is a rotation about the joint-local vertical, i.e. the LAST axis-angle
    component once the model-frame rotation is mapped back into the bone frame.
    """
    ...

def _curvature_deg(prior: SpinePrior, per_joint_deg: np.ndarray) -> np.ndarray:
    """Undo the chain sign: per-joint local angle -> midline curvature."""
    ...

def _redistributed(profile: np.ndarray, report: dict) -> np.ndarray:
    """The root station's curvature split evenly onto its two neighbours."""
    ...

def test_asset_chain_sign_matches_the_rig_parenting(asset_prior, report):
    ...

def test_asset_anterior_wave_bends_opposite_to_the_posterior(asset_prior):
    """The stations ahead of the root are parented backwards, so at any instant
    their local angle has the opposite sign to the same curvature behind the
    root.  Copying the sevengill's angles across (the pre-fix build) left the
    anterior wave running the wrong way."""
    ...

def test_asset_wave_reproduces_the_sevengill_curvature(asset_prior, report, sevengill):
    """The per-joint wave, read back as midline curvature, is the sevengill's
    own cruise wave with the root station's share moved onto its neighbours."""
    ...

def test_params_may_be_float64(prior):
    ...
