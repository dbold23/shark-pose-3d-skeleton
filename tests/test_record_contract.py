"""The world-frame record contract and the default-off ego compensation.

Plan items **P4**, **P5**, **P6** and **P7** of
``outputs/demo_2026-09-03/work/egomotion/report.md`` section 4.

Two things these tests exist to hold:

* **P4 is inert until it is asked for.**  Ship rule 5 makes ``--ego auto`` equal
  ``--ego none`` on both shipped clips, which guarantees the compensated branch
  is never executed end to end on this footage.  "Inert" is therefore not
  evidence of "correct", so the branch carries numerical tests of its own loss
  (report.md F5): identity bit-identity, the closed-form geodesic, and an
  injected camera rotation whose compensated ``d1`` must be zero.
* **A null is not a satisfied contract.**  Every need this footage cannot meet is
  written UNMET *with its reason* (Gate A), and the tests assert the reasons are
  there rather than merely that the fields exist.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.core.transforms import axis_angle_to_rotation_matrix
from shark_pose.model_3d.shark_smpl import SharkSMPL
from shark_pose.sim2real import world_frame as wf

@pytest.fixture(scope='module')
def model_v2() -> SharkSMPL:
    ...

@pytest.fixture(scope='module')
def fitter(model_v2):
    """A SpineSMPLify built exactly as the video path builds it (0 steps)."""
    ...

def _raw(T: int, n_joints: int, seed: int=0) -> dict:
    ...

def _reference_temporal(raw, root, trans):
    """The shipped expression, written out, so bit-identity is testable."""
    ...

def test_ego_none_leaves_the_temporal_loss_bit_identical(fitter, model_v2):
    """No ego track -> the objective is the one that shipped, to the bit."""
    ...

def test_fit_signature_defaults_ego_to_none(fitter):
    """`fit()` gained ONE trailing keyword and it defaults to off."""
    ...

def test_identity_R_cw_gives_the_closed_form_geodesic(fitter, model_v2):
    """With R_cw == I the compensated terms are the closed-form geodesic of
    R_cb, and the de-rotated translation terms are bit-identical to today's."""
    ...

def test_injected_camera_rotation_costs_the_compensated_d1_nothing(fitter, model_v2):
    """A world-static body under a pure camera rotation: the compensated root
    term is ZERO and the uncompensated one is the injected rate.

    This is the only test that exercises what the branch is FOR (report.md F5c).
    """
    ...

def test_invalid_pairs_are_dropped_and_the_mean_renormalised(fitter, model_v2):
    """R_cw = I at an invalid endpoint is prohibited: the pair leaves the sum."""
    ...

def test_an_unmeasured_camera_is_refused_by_the_fitter(fitter):
    """A track with no valid frame must not reach the loss as identity."""
    ...

def test_ego_is_ignored_without_share_clip_params(model_v2):
    """`share_clip_params=False` fits unrelated frames: no temporal term at all,
    so a camera track there would be meaningless rather than merely unused."""
    ...

def test_R_wb_composed_back_reproduces_the_fits_own_projection(model_v2):
    """R_cw @ R_wb @ X_body + t == X_cam, to 1e-6, against the model's forward.

    The body frame here is the model's own: the same pose with the root joint's
    rotation removed.  This is the test the convention has to pass -- it uses
    ``pose[root_joint]``, NOT ``root_orient``, because ``SpineSMPLify._pose``
    composes the prior's root recoil into the former and only the former rotates
    the chain.
    """
    ...

def test_R_wb_without_a_camera_track_is_the_camera_frame_attitude():
    ...

def test_omega_b_raw_recovers_a_known_constant_body_rate():
    ...

def test_heading_is_measured_about_the_named_up_proxy():
    """psi = atan2(u_x, u_z) about camera -y, and it UNWRAPS across +-180."""
    ...

def test_quality_byte_bits():
    ...

def test_boxcar_is_zero_phase_and_shrinks_symmetrically():
    ...

def test_attitude_quality_stamps_the_window_and_the_edge_rule():
    ...

def test_length_label_is_read_only_when_it_is_unambiguous():
    ...

def test_scale_is_unmet_with_a_reason_when_the_clip_has_no_label():
    ...

def test_scale_from_the_filename_field_reports_a_non_discriminating_interval():
    ...

def test_a_one_focal_range_check_is_flagged_degenerate_not_agreed():
    """One candidate is an ABSENCE of candidates, not a confirmation."""
    ...

def test_consumer_needs_are_stated_unmet_never_left_null():
    ...

def test_time_block_says_when_it_is_only_clip_local():
    ...

def test_gravity_is_null_and_says_why():
    ...

def test_a_reader_refuses_an_unknown_frames_convention():
    ...

def test_ego_none_records_unmeasured_and_never_pretends_identity():
    ...

def test_ego_auto_with_no_estimator_output_is_unmeasured_not_identity(tmp_path):
    ...

def test_a_refused_gate_is_recorded_and_not_applied(tmp_path):
    """Gate B refuses -> the estimate is stored, the fit does not see it."""
    ...

def test_consumer_need_5_requires_absolute_time_as_well_as_scale():
    """Need 5 is scale AND absolute time; a clip-local time base leaves it UNMET.

    The default CLI path has no --clip-meta, so time is clip_local even when the
    filename gives a scale (review of the ego-motion branch, 2026-09-07).
    """
    ...

def test_length_label_never_reads_an_underscore_as_a_decimal():
    """Tag ids precede the length in this corpus (corpus mining, 2026-09-07)."""
    ...
