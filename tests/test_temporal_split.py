"""TASK F -- the separately weighted root / translation temporal smoothers.

``outputs/demo_2026-09-03/work/root_jitter/report.md``.  The shipped v2 objective
holds the manoeuvre rails, the root rotation and the translation at ONE
``temporal_weight``, so the root cannot be stiffened without stiffening the tail;
at 0.5 the root half carries about a fifth of the observation gradient and the
shipped fits' ``root_orient`` came out 87.5 % (demo) / 55.6 % (clip2) frame-rate
jitter.  ``root_temporal_weight`` / ``translation_temporal_weight`` split the
term; ``None`` on both is the shipped behaviour.

The gate these tests exist to hold is the FIRST one: with the new leaves at
their defaults, nothing in the objective moves -- ``torch.equal`` on the loss,
and float equality of a short fit's every output.  Everything after that pins
the split form itself: the algebra, the geodesic's heading invariance (the
axis-angle d1 it replaces re-scales the SAME trajectory by up to 578x under a
re-framing), which target's gradient reaches ``root_orient``, and the fact that
a split term survives ``temporal_weight = 0``.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import pytest
import torch
from shark_pose.core.transforms import axis_angle_to_rotation_matrix
from shark_pose.encoder.smplify_fitting import SpineSMPLify
from shark_pose.model_3d.shark_smpl import SharkSMPL

@pytest.fixture(scope='module')
def model_v2() -> SharkSMPL:
    ...

def _fitter(model_v2, **kw) -> SpineSMPLify:
    ...

def _raw(T: int, n_seg: int, seed: int=0) -> dict:
    ...

def _root_trans(T: int, seed: int=1):
    ...

def _clip(model_v2, fitter, n_frames: int=10, seed: int=7):
    """A short synthetic clip: root, translation, the landmarks they project to."""
    ...

def test_the_new_leaves_default_to_none(model_v2):
    ...

def test_the_default_loss_is_torch_equal_to_the_coupled_one(model_v2):
    """Absent keys vs an explicit `null` on both: the same objective, bit for bit.

    Evaluated through ``_loss`` itself -- the whole objective, not just the
    temporal term -- because that is what the fit minimises.
    """
    ...

def test_a_short_fit_is_float_identical_with_the_leaves_at_their_defaults(model_v2):
    """Two fits, same seed, one built before the leaves existed in spirit."""
    ...

def test_the_harness_path_has_no_temporal_term_at_all(model_v2):
    """`share_clip_params=False` (every eval row) never reaches either branch."""
    ...

def test_setting_one_weight_switches_the_term_and_leaves_the_other_coupled(model_v2):
    ...

def test_the_split_term_survives_temporal_weight_zero(model_v2):
    """A root smoother with the manoeuvre rails free -- what the coupling cannot do."""
    ...

def test_the_geodesic_root_half_does_not_depend_on_the_heading(model_v2):
    """Re-frame every root by one fixed rotation and the MOTION is unchanged --
    only the frame the heading is written in moves.  The geodesic half must not
    notice; the axis-angle d1 it replaces re-scales the same trajectory (by 1.2x
    to 2x on this synthetic root, and by up to 578x on the shipped demo one)."""
    ...

def test_only_the_centroid_target_charges_the_rotation_translation_swap(model_v2):
    """The translation half's gradient reaches `root_orient` iff it sees the body.

    A root rotation paid for by a translation that puts the centroid back buys
    back 80-86 % of its pixel cost, and today's translation smoother is exactly
    blind to it: its gradient on `root_orient` is not merely small, it does not
    exist -- the term is not a function of the rotation at all.
    """
    ...

def test_both_is_the_mean_of_the_two_targets(model_v2):
    ...

def test_the_centroid_target_needs_the_posed_body(model_v2):
    ...

def test_an_unknown_target_is_refused(model_v2):
    ...

def test_the_v2_config_carries_the_keys_and_they_are_forwarded():
    ...

def test_non_clip_modes_clear_the_split_weights_in_the_effective_config():
    """A per-frame record must report the split smoothers as off.

    run_fit_video builds SpineSMPLify from the EFFECTIVE config and writes it
    into summary.fit_config; for mode != "clip" the temporal weight is zeroed
    and both split weights are cleared even when the YAML sets them (the split
    term would otherwise run at temporal_weight 0).
    """
    ...
