"""TASK B1 -- the joint (per-individual) fit: shared girth + betas across clips.

The defect these tests exist for: the girth field ``(K, 2)`` is fitted PER CLIP,
so on an oblique clip the mask width -- which is a MIX of the lateral ``g_x`` and
the dorsoventral ``g_z`` -- is booked entirely to ``g_z`` (the lateral axis rides
the tie prior).  The same animal then returns trunk ``g_z`` 1.09 on its broadside
window and 1.25 on its rolling window.  Fitting an individual's clips JOINTLY,
with the girth field and the betas SHARED and everything else per clip, is what
makes the two axes separable.

The load-bearing tests here are the BIT-IDENTITY ones.  The refactor moves the
hottest driver in the codebase (``SpineSMPLify.fit``) into
:class:`~shark_pose.encoder.joint_fitting.JointSpineFit`, and the single-clip
path must come out float-for-float unchanged, or every shipped number moves.
The guard is a golden capture taken BEFORE the refactor
(``tests/data/joint_fit_golden.npz``); regenerate it only from a checkout that
predates the change:

    python tests/test_joint_fit.py --write-golden
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.encoder.smplify_fitting import SpineSMPLify
from shark_pose.model_3d import SharkSMPL
from shark_pose.model_3d.shark_smpl import select_visible_landmarks

def make_model() -> SharkSMPL:
    ...

@pytest.fixture(scope='module')
def model_v2() -> SharkSMPL:
    ...

def make_fitter(model: SharkSMPL, **over) -> SpineSMPLify:
    ...

def make_clip(model: SharkSMPL, fitter: SpineSMPLify, n: int=6, seed: int=0, depth: float=6.0, width_px: float=20.0) -> dict:
    """A synthetic clip: the rest body in front of the camera, noisy landmarks."""
    ...

def run_single(seed: int=0, **over) -> dict:
    ...

def _golden_arrays() -> dict:
    ...

@pytest.mark.skipif(not GOLDEN.exists(), reason='golden capture not present')
def test_single_clip_fit_is_bit_identical_to_the_golden():
    """The refactor must not move a single bit of the single-clip fit."""
    ...

def _leaves(model, fitter, clip, n=6):
    ...

def test_include_shared_default_is_the_shipped_loss(model_v2):
    """The keyword must be INERT at its default -- every existing call site."""
    ...

def test_include_shared_removes_exactly_the_three_shared_terms(model_v2):
    """L(True) - L(False) == shape prior + girth prior + girth tie, exactly."""
    ...

def test_joint_loss_is_clip0_shared_plus_the_rest_bare(model_v2):
    """The joint objective is L_0(shared) + sum_{c>0} L_c(no shared)."""
    ...

def _joint_fit(model, n_clips=2, seeds=(1, 2), depths=(6.0, 7.0), **over):
    ...

def test_the_shared_field_is_one_tensor_in_every_clip(model_v2):
    """Both clips' records carry the SAME girth and the SAME betas."""
    ...

def test_per_clip_leaves_stay_independent(model_v2):
    """Two clips at different depths must NOT converge to one translation."""
    ...

def test_stage_a_does_not_move_the_shared_leaves(model_v2):
    """Stage A is global placement; the girth is detached and the betas absent."""
    ...

def test_two_identical_clips_agree_with_each_other(model_v2):
    """The same clip twice must fit to the same pose in both slots."""
    ...

def test_two_identical_clips_free_the_field_further_than_one(model_v2):
    """Two likelihoods against ONE prior: the field moves further off it.

    This is the point of billing the shared terms once rather than per clip --
    at N = 2 the prior's per-clip pull is halved, so a second view is allowed
    to overrule the tie instead of being outvoted by a second copy of it.
    """
    ...

def test_girth_jacobians_are_signed_per_clip_and_summed(model_v2):
    ...

def test_decimate_spreads_over_the_clip_and_never_takes_a_prefix():
    """A prefix scores a front-loaded, per-clip-unequal slice of the Jacobian.

    ``step = n // cap`` then ``[:cap]`` reaches frame 238 of 300 (79.7 %) and
    frame 119 of 163 (73.6 %).  Measured on the shipped pairs that cost
    FAR20110101 2.7 deg of separation -- 8.14x amplification against 13.07x,
    across the gate -- purely on the sampling choice.
    """
    ...

def test_two_contexts_may_not_share_one_fitter(model_v2):
    """prepare_clip stores per-CLIP state on the fitter, so one each."""
    ...

def test_the_identifiability_sds_name_their_own_units(model_v2):
    """`sd_g_*` used to be published as px-sigma girth sds; they were neither."""
    ...

def test_a_single_clip_reports_its_lateral_axis_as_prior(model_v2):
    """One view is rank 1 in (g_x, g_z); say so rather than implying otherwise."""
    ...

def test_joint_fit_refusals(model_v2):
    ...

def test_a_joint_stage_b_over_too_many_frames_is_refused(model_v2):
    """Say it, rather than silently scoring a strided subset instead."""
    ...

def test_clips_must_agree_about_stage_b_steps(model_v2):
    ...

def test_well_combinations(model_v2):
    """Full product at N <= 2, greedy above -- and 'first' runs one."""
    ...

def _joint_record(clip_id: str, jid: str, g_x, g_z, per_clip, ident, summed_ratio: float=4.34, this_clip=_UNSET, operator: str='band', tie_weight=1.0, prior_weight=1.0, extra: dict | None=None):
    ...

def _per_clip_ab():
    ...

def test_fuse_girth_reads_a_joint_record(model_v2):
    """A joint record's field is PASSED THROUGH, not fused."""
    ...

def _observed_pair(**kw):
    """A joint pair that clears every clause of the calibrated F1a gate."""
    ...

def test_fuse_girth_joint_gate_all_four_clauses(model_v2):
    """TASK F1a: separation, an unbiased operator, and the isotropy tie OFF.

    Calibrated on the synthetic aniso/iso pair against a known
    d(g_x - g_z) = 0.10 (roll/gate/calibration.json).  Each clause is checked by
    breaking exactly one of them on a pair that otherwise passes.
    """
    ...

def test_fuse_girth_joint_gate_does_not_require_the_deviation_prior_off(model_v2):
    """TASK F1a note 3b: `prior_weight` is PUBLISHED, never a bar.

    Dropping the deviation prior lifts the aniso-iso contrast (0.57 -> 0.86 at
    the GT pose, 1.16 at a fitted one) while taking the worst per-pair `g_x`
    level error from 2.5 % to 11-16 % and the shared field's trunk spread from
    0.03 to 0.67-1.04.  A gate that demanded it off would licence a lateral
    girth that is 15 % wrong.
    """
    ...

def test_fuse_girth_joint_gate_reads_an_absent_tie_weight_as_live(model_v2):
    """A record that does not state its tie weight gets no benefit of doubt."""
    ...

def test_fuse_girth_joint_summed_ratio_is_published_not_a_bar(model_v2):
    """TASK F1a: the summed ratio no longer gates the JOINT path.

    It is anti-informative there -- the single rolled synthetic view reads 1.51,
    inside the old 2.0 bar, and returns 2 % of a known anisotropy, while the
    pair that returns 86 % of it reads 3.76.  It still gates the per-clip path,
    which is a different question, and it is still published here.
    """
    ...

def test_fuse_girth_keys_the_jacobian_table_on_the_blocks_own_clip_id(model_v2):
    """`this_clip`, not Record.clip_id -- and a miss RAISES, never scores 0."""
    ...

def test_fuse_girth_calls_a_weakly_separated_pair_prior(model_v2):
    ...

def test_fuse_girth_refuses_records_from_different_joint_fits(model_v2):
    """The joint id is what proves two records came from ONE fit."""
    ...

def test_fuse_girth_refuses_disagreeing_copies_of_a_shared_field(model_v2):
    ...

def test_a_non_joint_record_keeps_the_old_path(model_v2):
    """Every shipped record has no joint_fit block and must be unaffected."""
    ...

def test_drive_fit_video_resolves_every_request():
    """The default driver runs each request; a custom one may answer instead."""
    ...

def test_fit_request_carries_exactly_the_fit_call(model_v2):
    """``FitRequest.run()`` is the call fit_clip_v2 used to make inline."""
    ...

def test_frozen_pose_pins_the_pose_and_frees_only_the_shared_leaves(model_v2):
    """``frozen_pose`` + ``freeze_clip_params``: the arm that tests the joint
    construction with everything but the shared field at truth.

    Both switches are off in every shipped fit (the golden test above pins
    that); with them on the returned pose must be the given one EXACTLY, the
    translation and the root must not move, and the shared girth field must
    still be fitted.
    """
    ...

def test_a_param_group_with_no_leaves_is_dropped_not_handed_to_adam(model_v2):
    """``freeze_clip_params`` empties the per-clip groups; Adam refuses those."""
    ...

def test_anneal_is_off_by_default_and_the_softness_never_moves(model_v2):
    """No ``sigma_px_final`` -> the rasteriser blur is the constant it was."""
    ...

def test_anneal_walks_geometrically_from_sigma_px_to_the_final_one(model_v2):
    """0 -> ``sigma_px``, 1 -> ``sigma_px_final``, log-uniform in between."""
    ...

def test_anneal_rejects_a_softness_that_is_not_a_length(model_v2):
    ...

def test_the_driver_anneals_every_clip_from_the_same_stage_b_progress(model_v2):
    """One optimiser loop, one blur: the hook steps both fitters together."""
    ...

def test_run_calls_on_step_before_every_loss(model_v2):
    """The hook's contract: index 0..steps-1, each BEFORE its loss."""
    ...

def _unit(deg: float):
    """A unit row at ``deg`` from the g_x axis."""
    ...

def test_pairwise_separation_is_every_pair_not_just_clip_0():
    """The E3 N = 3 record's own three angles, off its published unit rows.

    ``joint_fit.observed_axis.per_clip[*].per_frame_unit_row`` at 6 dp, from
    ``f1_w3_tie0_N3/AN15092101/individual_AN15092101.json``: oblique
    [0.667943, 0.670321], broadside [0.022363, 0.999084], intermediate
    [0.14583, 0.979864].  The shipped statistic saw two of the three pairs and
    published 43.616 or 36.433 by ``--clip`` order, never the 7.183 deg
    broadside x intermediate pair.
    """
    ...

def test_pairwise_separation_does_not_move_with_the_clip_order():
    """All six orderings of the E3 rows, one answer."""
    ...

def test_pairwise_separation_at_two_clips_is_the_one_pair():
    """R3's inertness clause: at N = 2 max == min == the old number."""
    ...

def test_pairwise_separation_below_two_clips_has_no_pair():
    ...

def test_girth_jacobians_publish_the_matrix_and_agree_with_separation_deg(model_v2):
    ...

def _n3(sep_matrix_max, sep_matrix_min, legacy_sep, **kw):
    """Three clips of one joint fit, with a chosen pairwise max / min."""
    ...

def test_fuse_girth_gates_on_the_pairwise_max_not_the_published_scalar(model_v2):
    """A third, redundant clip must not be able to lower the verdict.

    The record below has a well-separated pair (43.6 deg) and a nearly parallel
    one (7.2 deg).  Whichever clip the fit happened to pass first, the pair that
    identifies the lateral axis is the same pair, so the verdict is `observed`
    even when the order-dependent legacy scalar reads below the bar.
    """
    ...

def test_fuse_girth_still_refuses_when_no_pair_clears_the_bar(model_v2):
    ...

def test_fuse_girth_reads_a_record_written_before_the_matrix_existed(model_v2):
    """Back-compat: no pairwise keys, so the old scalar is the best reading."""
    ...

def _refused_pair(attribution):
    ...

def test_a_refusal_prints_the_non_broadside_members_resolution(model_v2):
    """Spec 3.6: geometry or resolution, and the record has to say which."""
    ...

def test_a_refusal_without_attribution_says_it_cannot_attribute(model_v2):
    ...

def test_an_all_broadside_refusal_is_geometry(model_v2):
    ...

def test_well_combinations_stamp_the_resolved_mode(model_v2):
    """``auto`` is not a provenance stamp: at N = 3 it is greedy, 4 of 8."""
    ...

def test_the_schedule_block_publishes_the_resolved_wells_mode(model_v2):
    """The E3 N = 3 record must read greedy / 4 / not exhaustive."""
    ...

def test_fit_stamps_travel_from_the_fit_record_to_the_morphometrics_record():
    """P5 is scored where a ``value_m`` lives, and that is not the fit record."""
    ...

def test_fit_stamps_say_so_on_a_record_written_before_they_existed():
    ...

def test_every_g_x_refusal_carries_the_attribution(model_v2):
    """Spec 3.6 / R9: "print them beside EVERY g_x refusal".

    The attribution was appended at ONE call site -- the separation clause --
    so a refusal landing on clause 6, on the width operator, on the tie or on
    the clip count shipped with no way to tell GEOMETRY from RESOLUTION.
    """
    ...
