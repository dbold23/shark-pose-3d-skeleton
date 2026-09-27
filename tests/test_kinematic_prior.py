"""TASK F -- the anisotropic root smoother and the kinematic (swimming) prior.

``outputs/demo_2026-09-03/work/kinematic/report.md``.  Two new objects, both
default-off:

* ``root_temporal_weight`` now also accepts a 3-sequence ``(w_x, w_y, w_z)``,
  absolute weights on the three CAMERA axes of the LEFT increment
  ``log(R_{k+1} R_k^T)`` -- the only increment whose rotation-vector components
  lie on camera axes when ``root_orient`` is the body->camera rotation.  Three
  equal entries are the scalar case and dispatch into the untouched isotropic
  geodesic path, so the task-F isotropic sweep stays exactly reproducible.
* ``swim_*`` -- four rails on the 0.5 s low-passed root velocity in body
  lengths per second: non-holonomic (lateral + dorsal), head-first, a
  log-normal cruise speed and a heading-rate cap.

The gate these tests exist to hold is the first one: at the defaults nothing in
the objective moves -- ``torch.equal`` on the whole loss and float equality of a
short fit.  Everything after that pins the two terms' own claims: the frame the
per-axis weights act in, the isotropic reduction, that the lateral rail is blind
to a head-first swimmer and bites on a sideways one, and that the harness path
(``share_clip_params=False``) cannot see any of it.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import pytest
import torch
from shark_pose.core.transforms import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from shark_pose.encoder.smplify_fitting import SpineSMPLify
from shark_pose.model_3d.shark_smpl import SharkSMPL

@pytest.fixture(scope='module')
def model_v2() -> SharkSMPL:
    ...

def _fitter(model_v2, **kw) -> SpineSMPLify:
    ...

def _root_trans(T: int, seed: int=1):
    ...

def _clip(model_v2, fitter, n_frames: int=10, seed: int=7):
    """A short synthetic clip: root, translation, the landmarks they project to."""
    ...

def _loss_args(fitter, root, trans, kp, conf):
    ...

def _pose_with(fitter, model_v2, root: torch.Tensor) -> torch.Tensor:
    ...

def test_every_new_leaf_defaults_to_off(model_v2):
    ...

def test_absent_keys_and_explicit_defaults_give_one_objective(model_v2):
    """Absent keys vs every new key spelled out at its default: one objective."""
    ...

def test_a_short_fit_is_float_identical_with_the_new_leaves_at_their_defaults(model_v2):
    ...

def test_the_harness_path_sees_neither_term(model_v2):
    """`share_clip_params=False` -- every eval row -- reaches no clip term."""
    ...

def test_an_isotropic_triple_is_the_scalar_case_bit_for_bit(model_v2):
    ...

def test_an_anisotropic_triple_takes_the_per_axis_branch(model_v2):
    ...

def test_the_per_axis_term_is_the_stated_arithmetic(model_v2):
    """d1 + d2 of the LEFT increments' log coordinates, weighted, over 3."""
    ...

def test_equal_weights_reproduce_the_isotropic_geodesic_d1(model_v2):
    """`sum_a omega_a^2 = theta^2`, so the d1 halves agree exactly; d2 does not.

    The d1 identity is why an isotropic triple can be dispatched away without an
    approximation, and the d2 gap -- the second difference of the log
    coordinates against the geodesic distance between consecutive increments,
    O(theta^3) apart -- is why it MUST be dispatched rather than re-derived.
    """
    ...

def test_the_per_axis_weights_act_on_camera_axes(model_v2):
    """A pure camera-z ramp is charged by w_z alone, whatever the attitude.

    ``root_orient`` is the body->camera rotation, so the increment whose
    components lie on camera axes is the LEFT one; the right/body increment
    smears the same trajectory across all three.  This is the property the
    published per-axis rates (``clip_metric.root_rate_block``) are measured
    with, and the whole point of weighting the axes apart.
    """
    ...

def test_a_per_axis_triple_leaves_the_translation_half_alone(model_v2):
    ...

@pytest.mark.parametrize('bad', [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], '1'])
def test_a_malformed_root_temporal_weight_is_refused(model_v2, bad):
    ...

def test_negative_per_axis_weights_are_refused(model_v2):
    ...

def _straight_clip(fitter, model_v2, direction, T=13, speed=0.05):
    """Constant-velocity translation along `direction`, body axes = identity.

    With the root rotation at the identity the body's lateral / longitudinal /
    dorsal axes are camera x / y / z, so a translation along y is head-first and
    one along x is pure sideslip.
    """
    ...

def test_the_lateral_rail_is_zero_head_first_and_positive_sideways(model_v2):
    ...

def test_the_lateral_rail_is_the_stated_value_in_body_lengths(model_v2):
    ...

def test_the_head_first_rail_is_one_sided(model_v2):
    ...

def test_the_speed_prior_is_minimised_at_its_median(model_v2):
    ...

def test_the_heading_cap_is_free_below_the_cap_and_quadratic_above(model_v2):
    ...

def test_the_swim_terms_pull_the_translation_and_the_root(model_v2):
    """The body axes carry gradient, which is what makes it a SWIMMING prior."""
    ...

def test_a_weighted_swim_term_moves_the_objective(model_v2):
    ...

def test_the_boxcar_is_the_p7_smoother(model_v2):
    """The low-pass the terms see is `world_frame.boxcar_zero_phase`, exactly."""
    ...

@pytest.mark.parametrize('kw,match', [({'swim_speed_weight': 1.0, 'swim_speed_mu_bl_s': 0.0}, 'swim_speed_mu_bl_s'), ({'swim_heading_weight': 1.0, 'swim_heading_cap_deg_s': 0.0}, 'swim_heading_cap_deg_s'), ({'swim_lateral_weight': -1.0}, 'must be >= 0'), ({'swim_body_length': 0.0}, 'swim_body_length')])
def test_a_malformed_swim_leaf_is_refused(model_v2, kw, match):
    ...

def test_the_v2_config_carries_the_new_keys_and_forwards_them():
    ...

def test_per_frame_mode_clears_both_terms():
    """A velocity has no meaning for unrelated frames, so the modes drop it."""
    ...
DEFAULT_LOSS_PINNED = 2.1409568786621094

def test_the_default_loss_matches_the_pinned_head_value(model_v2):
    """Ship rule 7 -- the default objective is HEAD's, not merely self-consistent."""
    ...

def test_the_swimming_prior_refuses_an_ego_track(model_v2):
    """The prior reads camera-frame velocity; under a measured pan it must refuse."""
    ...

def test_explicit_zero_shape_leaves_reach_the_validators():
    ...
