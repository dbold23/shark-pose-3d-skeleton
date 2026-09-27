"""SharkSMPL on the v2 rig + SpineSMPLify fitting along the swimming rails.

The v2 asset directory (``data/model_assets_v2``) ships ``skeleton.json``: a
31-joint rig -- a 13-station midline vertebral chain plus 9 fin root/tip pairs --
skinned onto the SAME 3912-vertex template as v1.  Two things must hold:

1. the model reads its joint count and kinematic tree from the asset directory,
   so v1 (16 joints, no ``skeleton.json``) is untouched, and
2. fitting happens in the low-dimensional swimming parameterisation, not in 93
   free rotations, so the head stays rigid.

Everything here is CPU-only and runs in a couple of seconds.
"""
from __future__ import annotations
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from shark_pose.core.skeleton import BONE_PAIRS
from shark_pose.encoder.smplify_fitting import SpineSMPLify, compose_axis_angle, projected_station_bend_deg, retarget_translation
from shark_pose.core.transforms import axis_angle_to_rotation_matrix
from shark_pose.model_3d.shark_smpl import SharkSMPL, select_visible_landmarks
from shark_pose.sim2real.mask_centerline import resample_polyline

@pytest.fixture(scope='module')
def model_v2() -> SharkSMPL:
    ...

@pytest.fixture(scope='module')
def model_v1() -> SharkSMPL:
    ...

def test_v2_model_loads_rig_from_skeleton_json(model_v2):
    ...

def test_v1_model_is_unchanged(model_v1):
    """No skeleton.json -> the 16-joint defaults, exactly as before."""
    ...

def test_v2_rest_pose_equals_template(model_v2):
    ...

def test_v2_rest_joints_are_the_measured_bone_positions(model_v2):
    ...

def test_v2_landmarks_are_always_sixteen(model_v2):
    """The observation landmarks are surface points, so they do not follow the rig."""
    ...

def test_v2_landmarks_differ_from_v1_only_where_the_asset_says(model_v1, model_v2):
    """v1 and v2 no longer share a template, and the differences are accounted for.

    Two things separate the two landmark assets and nothing else may:

      * the measured body-frame corrections (scripts/calibrate_landmarks.py,
        ``gated_corrections``), which ship in v2 only -- v1's fit path is a
        frozen reference.  They are stamped in ``correction_pct_span``;
      * the template RESHAPE (scripts/reshape_template.py), which extends the
        rostrum and thins the peduncle.  It moves ``snout_tip`` by design and
        re-places ``caudal_notch`` on the thinned caudal midline, and it grows
        the body span the corrections are quoted against, so a corrected row
        moves by a little more than v1's span would say.

    Every other row must still be within a fifth of a percent of the body span.
    """
    ...

def test_v2_arc_span_is_measured_from_the_shipped_asset():
    """RIG_ARC_SPAN is a property of the template, so it is MEASURED, not pasted.

    ``mask_centerline.measure_rig_arc_span`` reproduces the module's constants
    on the template they were measured on (that is what the constants' own
    docstrings record).  The SHIPPED reshape is dorsoventral only -- the
    rostrum half was withdrawn on 2026-09-06 -- so the thinning-axis span is
    still exactly RIG_ARC_SPAN, while the peduncle-terminated span moves a
    little because the thinning shifts the sqrt(width x depth) minimum the
    peduncle plane is found at.  What the extractor stamps on a record and what
    the fitter measures must be the same number either way, which is why both
    are measured off the asset rather than pasted.
    """
    ...

def test_v2_posing_moves_the_mesh(model_v2):
    """A tail-station rotation must actually deform the skin (weights are live)."""
    ...

def test_compose_axis_angle_matches_matrix_product():
    ...

def test_compose_axis_angle_is_differentiable_at_the_identity():
    """The fit starts at zero rotation, so the gradient there must be finite."""
    ...

def test_retarget_translation_preserves_the_rigid_transform():
    """Same body placement, different root pivot."""
    ...

def test_v1_to_v2_warm_start_takes_only_the_root(model_v1, model_v2):
    ...

def _synthetic_frame(model, fitter, n_frames=1, noise_px=1.5, seed=0):
    """Project the v2 rest landmarks under a known root pose; add pixel noise."""
    ...

def _reprojection_px(model, fitter, root, trans, kp, n_frames):
    ...

def test_spine_smplify_reduces_error_and_keeps_the_head_rigid(model_v2):
    ...

def test_spine_smplify_output_shapes_and_clamping(model_v2):
    ...

def test_spine_smplify_stays_near_the_cruise_prior_without_data(model_v2):
    """With zero confidence there is no data term, so the prior must hold."""
    ...

def _synthetic_wave_clip(model, fitter, amp, freq_hz, phase, n_frames, fps):
    """Project the v2 landmarks under a KNOWN travelling wave over a clip."""
    ...

def test_spine_smplify_recovers_a_known_beat_frequency(model_v2):
    """The tail-beat frequency is fit, not pinned: 0.3 Hz in, 0.3 Hz out."""
    ...

def test_amplitude_is_chosen_by_the_data_not_by_default_amp(model_v2):
    """A hostile default_amp must not decide the fitted cruise amplitude.

    The mixture amplitude prior has two wells (glide ~0.05, swimming ~0.55)
    with a barrier Adam cannot cross, so starting in the glide well used to pin
    the answer there.  The coarse (amp x freq x phase) search plus one fit per
    well must recover a 0.7-amplitude clip from a 0.05 start, and must record
    BOTH wells' converged objectives.
    """
    ...

def test_amp_restarts_off_fits_one_well_only(model_v2):
    """`amp_restarts=False` keeps the single coarse winner (the cheap path)."""
    ...

def test_whole_clip_shares_the_wave_across_every_frame(model_v2):
    """Amplitude, frequency and phase are ONE triple for the whole clip."""
    ...

def test_frequency_is_frozen_when_frames_are_independent(model_v2):
    """t = 0 makes the frequency unobservable, so it must not drift."""
    ...

def test_amplitude_and_frequency_priors_are_finite_and_minimal_at_the_mode(model_v2):
    """The broad mixture prior must be well behaved everywhere it can be asked."""
    ...

def test_spine_smplify_rejects_a_v1_model(model_v1):
    ...

def test_visualizer_bone_pairs_follow_the_rig(model_v1, model_v2):
    ...

def test_visualizer_renders_the_girth_the_fit_produced(model_v2):
    """A render that drops the girth field draws an animal no fit produced.

    girth leaves the JOINTS bit-identical by construction, so the renderer's
    joints-vs-joints_3d cross-check is blind to it -- this is the only guard.
    """
    ...

def test_v2_fit_config_is_loadable():
    ...

def test_build_spine_smplify_from_config(model_v2):
    ...

def _mask_from_pose(model, fitter, root, trans, res):
    """Render a clip's own silhouette as a hard target mask, (T, res, res)."""
    ...

def _iou_loss(model, fitter, root, trans, target):
    """Mean (1 - soft IoU) of a placement against a target mask stack."""
    ...

def test_silhouette_term_pulls_the_fit_onto_the_mask(model_v2):
    """A displaced fit driven ONLY by the silhouette must close the gap.

    The landmark term is switched off (zero confidence) so nothing but the mask
    can move the body: this measures the silhouette gradient itself, not the
    fit's ability to reach the answer some other way.
    """
    ...

def test_silhouette_weight_zero_reproduces_the_plain_fit(model_v2):
    """Weight 0 must be bit-for-bit the old objective, masks present or not."""
    ...

def test_silhouette_batches_cycle_and_validate(model_v2):
    """Slices tile the masked frames, wrap around, and reject bad inputs."""
    ...

def test_silhouette_shares_the_fitters_camera(model_v2):
    """The mask term and the landmark term must be ONE camera, whichever it is.

    Before the camera pass the fitter's focal was pinned to ``BASE_FOCAL``
    whenever the mask term was on, because the rasteriser hard-coded that
    pinhole.  It no longer does: the fitter forwards its own focal and
    principal point, so a calibrated camera (a real lens, focal ~120 px at
    256) drives both terms.  What is still required is the RESOLUTION
    convention the rasteriser scales from.
    """
    ...

def test_calibrated_camera_projects_through_the_same_pinhole_everywhere(model_v2):
    """The reprojection term and the soft rasteriser agree, at any camera.

    A mesh is posed once and rendered two ways -- the fitter's own
    ``camera.project`` on the vertices, and ``soft_silhouette``'s
    ``project_to_pixels`` at the rasteriser's resolution.  They must be the
    same projection up to the resolution scale, or the landmark term and the
    mask term are pulling the same body through two different cameras.
    """
    ...

def test_fit_video_base_focal_matches_the_rasteriser(model_v2):
    """The warm-start rescale divides by the focal the warm start was trained at."""
    ...

def test_warm_start_depth_rescale_preserves_the_initial_reprojection():
    """``tz *= focal / 1000`` keeps the warm start at the right image size.

    The SPIN warm start's translation is in focal-1000 units (the synthetic
    renders sampled U(800, 1200)).  ``f * X / Z`` is invariant when the focal
    and the depth scale together, so for a body at ONE depth the projection is
    identical up to the principal point -- asserted exactly on a planar point
    set here.

    A body with real depth also gains parallax, because the calibrated camera
    is genuinely wider and the animal is genuinely closer: at the demo fit's
    own geometry (extent 4.16 x 1.59 x 1.92 at tz 21.7, focal 1000 -> 119.57)
    the rescale leaves the projected extent 1.41x, where NOT rescaling leaves
    it 0.12x -- 8x too small, which is what stage A would otherwise have to
    walk back.  That extra perspective is the point of fitting with the real
    camera, not an error in the rescale.
    """
    ...

def test_warm_start_rescale_reads_the_checkpoint_train_focal(caplog):
    """The rescale divides by the focal the CHECKPOINT PROJECTED THROUGH.

    An unstamped checkpoint falls back to the pipeline default, which must
    reproduce the pre-stamp behaviour exactly -- ``proxy_focal / 1000``.  A
    stamped one divides by its own value, whatever that is; the stamp is the
    model's projection focal, not the focal its training renders used.
    """
    ...

def test_warm_start_rescale_refuses_a_stamp_of_unknown_meaning(caplog):
    """A pre-fix ``train_focal`` is ignored, not divided by.

    An interim build stamped ``train_focal`` with the DATASET's median render
    focal.  Ten v3 checkpoints carry that stamp (169.41 for a model that
    projects through 1000), and dividing a proxy focal by it rescales the
    warm-start depth 5.90x the wrong way -- silently, because a present stamp
    looks trustworthy.  ``checkpoint_meta`` now always writes
    ``data_focal_median`` beside ``train_focal``, so its absence dates the
    stamp and the value is refused.
    """
    ...

def test_warm_start_focal_plan_keeps_the_old_rescale_and_drops_the_new_one(caplog):
    """Which of the two warm-start conventions the fitter is in.

    A pre-focal checkpoint's tz is in the units of the one focal its training
    loss projected through, so the fitter rescales it by proxy_focal /
    train_focal and passes the network no focal at all -- bit-identical to what
    shipped.  A focal-conditioned checkpoint's tz is metric: the focal goes INTO
    the network and the depth is left alone.  Getting this backwards is the
    5.90x error the v3 checkpoints shipped with.
    """
    ...

def test_spin_warmstart_passes_the_focal_only_when_it_is_given(shark_model):
    """`_spin_warmstart(focal=...)` reaches the model as a per-frame tensor."""
    ...

def test_training_refuses_a_set_rendered_at_another_focal(caplog):
    """The 2D and 3D supervision have to be asking for the same translation.

    ``SPINModel`` projects the 2D loss through one fixed focal while
    ``Keypoint3DLoss`` supervises the absolute camera-space joints, so a set
    rendered at a different focal asks for two translations at once.  The v3
    run did exactly that and plateaued at j3d 36.22 against v2's 4.02; the
    guard turns it into a refusal instead of a quiet compromise.
    """
    ...

def test_train_spin_stamps_the_models_projection_focal(tmp_path):
    """``train_focal`` is the focal the MODEL projects through, not the data's.

    The 2D loss goes through ``SPINModel.focal_length`` whatever focal the
    renders used, so the regressed translation is in those units and that is
    the only number scripts/fit_video.py may divide by.  The set's own focal
    distribution rides along under separate keys.
    """
    ...

def test_v3_train_configs_differ_from_v2_only_in_the_dataset_and_warm_start():
    """The v3 configs must not smuggle a second change into the comparison."""
    ...

def test_undistort_centerline_record_moves_the_curve_and_drops_stale_angles():
    """A record loaded with a camera is undistorted, and its stored bend voided.

    ``stations_bend_deg`` was measured on the DISTORTED polyline -- the exact
    quantity the camera pass exists to correct -- so it must not survive into a
    fit that reads undistorted geometry.
    """
    ...

def test_build_spine_smplify_camera_defaults_to_the_pipeline_pinhole(model_v2):
    """`camera: none` must reach the fitter as the untouched legacy pinhole."""
    ...

def test_camera_summary_block_states_the_proxy_resolution_either_way():
    """`proxy_resolution` is the letterbox the focal and principal point are in.

    It was present only on the calibrated branch, so a consumer reading a
    camera-less fit had to guess which resolution `proxy_focal` 1000 and the
    centred principal point were quoted at.  Both branches carry it now, and
    both carry the same set of keys a consumer reads off either.
    """
    ...

def test_load_sam_mask_undistorts_before_letterboxing(tmp_path):
    """A mask crosses into the pinhole frame the same way a keypoint does."""
    ...

def test_amp_well_scores_report_the_silhouette_term(model_v2):
    """Well selection sees the mask term, and records it separately."""
    ...

def test_sam_masks_are_letterboxed_into_the_silhouette_canvas(tmp_path):
    """A wide mask lands where the letterbox says, padded above and below."""
    ...

def test_silhouette_config_block_and_builder(model_v2):
    ...

def _rolled_root(roll, n_frames: int) -> torch.Tensor:
    """(T, 3) root orientation at ``roll`` radians about the body's long axis."""
    ...

def _swimming_params(fitter, n_frames: int, amp: float=0.6):
    """Clip-wide wave parameters at a real cruising amplitude, plus the times."""
    ...

def _projected_stations(model, fitter, params, times, root, trans):
    ...

def _centerline_fitter(model, **kwargs):
    """A fitter whose station fractions span the WHOLE polyline.

    The synthetic centrelines below are built from the projected stations
    themselves, so they start and end exactly where the spine does; the demo's
    real span (0.049-0.881 of a snout-tip-to-caudal-tip mask centreline) is
    covered by ``test_station_arc_fractions_map_onto_the_mask_centreline``.
    """
    ...

def test_station_arc_fractions_map_onto_the_mask_centreline(model_v2):
    """Stations are ordered, inside the measured span, and the span is settable."""
    ...

def test_midline_term_pulls_a_wrong_roll_back(model_v2):
    """A 40-degree roll error is recovered from the projected midline alone.

    Roll about the body's long axis is what the 16 near-midline landmarks do not
    see: it decides how much of the tail's lateral sweep survives projection.
    The body here is genuinely bent (cruise amplitude 0.6), the "centreline" is
    the GT projected midline plus sub-pixel noise, and NOTHING else is in the
    objective -- so this measures the midline gradient itself.
    """
    ...

def test_midline_term_uses_ordered_correspondences_and_widths(model_v2):
    """Arc-length correspondence, width weighting and the chamfer fallback."""
    ...

def test_midline_data_validates_its_inputs(model_v2):
    ...

def test_upright_hinge_is_zero_inside_the_margin(model_v2):
    """Zero (and flat) while the body is upright, quadratic once it is not."""
    ...

def test_midline_and_upright_weight_zero_reproduce_the_plain_fit(model_v2):
    """Weight 0 must be bit-for-bit the old objective, centrelines or not."""
    ...

def test_fit_accepts_centerlines_for_a_subset_of_frames(model_v2):
    """Centrelines are optional per frame, and driving on them closes the gap.

    Both runs are the same fit from the same 30-degree-wrong roll with the
    landmark term switched off (zero confidence); only the midline weight
    differs, so the drop in midline distance is the term's doing and nothing
    else's.
    """
    ...

def test_midline_and_upright_config_blocks_and_builder(model_v2):
    ...

def test_centerline_json_loads_into_proxy_pixels(tmp_path):
    """The --centerline JSON lands in the same letterbox as the detections."""
    ...

def _bend_clip(model, fitter, n_frames, fps, amp, freq_hz, phase, roll=0.0):
    """A clip posed with a KNOWN wave, plus the bend series it projects to.

    The root lays the body broadside (long axis along camera +x), so the yaw
    wave sweeps in the image plane and the chord bend is at its most visible --
    the geometry the demo clip is closest to.
    """
    ...

def test_projected_station_bend_equals_the_mask_bend_metric():
    """The torch bend and the mask's own bend are the same measurement.

    ``mask_centerline.bend_of_projected_stations`` is what the SAM centreline is
    scored by; the fitter needs a differentiable version of it, and the two must
    agree or the fit would be pulled toward a number nobody reports.
    """
    ...

def test_bend_series_term_fits_the_beat_timing(model_v2):
    """A 0.2 Hz beat is recovered from the bend series alone.

    The landmark term is switched off (zero confidence), so the ONLY thing that
    knows about the wave is the mask's per-frame bend.  Without the term the fit
    keeps the prior's 0.45 Hz (its coarse grid, scored on the prior alone, picks
    0.4); with it, the grid picks the right well and stage A refines it to
    within 10% of the truth, with the bend series in phase rather than mirrored.
    """
    ...

def test_bend_weight_zero_reproduces_the_plain_fit(model_v2):
    """Weight 0 must be bit-for-bit the old objective, bend series or not."""
    ...

def test_bend_data_validates_its_inputs(model_v2):
    ...

def test_bend_series_config_block_and_builder(model_v2):
    ...

def test_bend_series_loads_from_the_centerline_json(tmp_path):
    """The --centerline JSON yields the mask's own station bend, per frame."""
    ...

def test_stored_station_bend_is_only_trusted_with_matching_provenance(tmp_path):
    """A stored bend is used only when the record proves WHERE it was measured.

    ``--span`` on the extractor and ``midline.arc_span`` in the fit config are
    two independent knobs, so a stored ``stations_bend_deg`` can be an angle
    measured on a different piece of the body.  The record has to say which.
    """
    ...

def test_recompute_uses_the_records_own_arc_span_not_the_rigs(tmp_path):
    """A dt_path record's stations live on ITS span, never on RIG_ARC_SPAN.

    The trunk axis stops at the peduncle, so the 13 rig stations cover a
    different fraction of a shorter curve than they do on a snout-to-fin-tip
    centreline.  Sampling such a polyline at RIG_ARC_SPAN's fractions reads the
    stations off the wrong places, and the fit is then taught the wrong bend.
    """
    ...

def test_midline_polylines_are_reparametrised_onto_the_fits_span():
    """A dt_path record's midline targets must land on the stations it measured.

    `SpineSMPLify._midline_targets` samples whatever polyline it is handed at
    the FITTER's station arc fractions.  A record whose axis stops at the
    peduncle carries the same 13 stations at different fractions of a shorter
    curve, so the raw polyline puts the targets on the wrong anatomy -- a mean
    121 native px out on the shipped demo records, against a 12 px sigma.
    """
    ...

def test_bend_recompute_matches_a_dt_path_records_stored_series():
    """The recompute fallback must reproduce what the extractor stored.

    The extractor samples with `extrapolate=span[1] > 1.0`, and a peduncle-cut
    record's last station sits at 1.044 -- past the end of the line.  Clamping
    it there instead put the fallback RMS 0.77 deg (demo) and 2.13 deg (clip 2)
    off the stored series, silently, for exactly the records that have to be
    recomputed.
    """
    ...

def test_bend_quality_gate_separates_truncation_from_grazing():
    """The gate rejects a CUT silhouette and keeps one that merely grazes.

    Frame-edge contact on its own is not truncation -- on the demo clip it is
    true for 95 of 150 whole-animal masks -- so the two tests with teeth are
    the centreline ENDPOINTS' distance to the border and the border contact as
    a fraction of the mask perimeter.  Of the two numeric bands, `length_rel`
    is counted but inert, while `abs_bend_z` rejects by default (measured on
    clip 2: it is the only test that catches three half-animal SAM masks whose
    station bend reads +66 to +69 deg while both edge tests pass).
    """
    ...

def test_bend_gate_ignores_edge_on_records_without_the_new_fields():
    """Old centrelines are not edge-gated unless `legacy_touches_edge` says so.

    A pre-2026-09 record carries only the `touches_edge` boolean, which is the
    blunt test this gate replaces (95 of the demo's 150 whole-animal masks trip
    it).  Absence of `edge_near_end_px` / `edge_contact_frac` is not evidence of
    truncation, so such a record passes -- unless the operator asks for the old
    behaviour explicitly.
    """
    ...

def test_bend_gate_endpoint_margin_needs_an_image_size():
    """Without a size the endpoint test is skipped and says so, not guessed."""
    ...

def test_bend_gate_thresholds_match_the_extractors():
    """One set of numbers: a record's `ok_quality` and the fit's gate agree."""
    ...

def test_bend_target_smoothing_is_a_median_filter_on_the_target_only():
    """`smooth_frames` medians the accepted target series in frame order."""
    ...

def test_bend_target_smoothing_does_not_bridge_a_gap_in_the_accepted_frames():
    """The window is frames, not list positions: rejected frames are holes.

    Records every 2 frames; the gate removes the middle of the sequence, so the
    accepted list is [0, 2, 4, 200, 202, 204].  A width-5 filter reaches
    +/- 4 FRAMES, so the two clusters must not see each other -- a positional
    filter would median frame 4 together with frames 200 and 202, 3 s away.
    """
    ...

def test_bend_series_is_inert_at_weight_zero(model_v2, tmp_path):
    """weight 0 must not read, gate or smooth a single centreline record.

    This is the eval harness's path (round_004 val89 15.487366676330566 /
    9.624824523925781 has to stay bit-identical), and it is also what a
    --centerline run with the term switched off is entitled to: the driver
    guards the whole bend block on `smplify.bend_weight > 0`, so a malformed
    or fin-contaminated centreline cannot change the objective by accident.

    The gate BLOCK is shared with the width series since 2026-09-05, so the
    guard is now "either term is on"; what keeps this promise is the line that
    empties ``bend_targets`` unless the bend weight is positive.  Both are
    pinned below, because a fit with the term off must see no bend observation
    even when a width fit is gating the same records.
    """
    ...

def test_bend_series_error_measures_the_selected_fit(model_v2):
    """The summary's error is the WRITTEN fit's, not the last scored step's."""
    ...

class _Out:
    """The two fields :meth:`SpineSMPLify._width_loss` reads off a forward."""

    def __init__(self, vertices, joints):
        ...

def _rest_forward(model, n_frames=1, depth=8.0, x_scale=1.0):
    """Identity pose at ``depth``, optionally widened LATERALLY by ``x_scale``.

    The model frame's x is lateral and z is dorsal; with an identity pose the
    camera sees x as image x, y as image y and z as depth, so scaling x alone
    scales the projected body WIDTH by exactly that factor and leaves every
    depth -- hence every projection denominator -- untouched.  That is what
    makes the factor test below exact rather than approximate.
    """
    ...

def test_station_bands_are_body_wall_not_fin(model_v2):
    """The band is a slab of the rest template minus G1's own fin gate."""
    ...

def test_radius_median_fin_rule_is_the_fallback(model_v2):
    """The local rule still works, is selectable, and is the one that differs."""
    ...

def test_station_width_scales_with_a_widened_mesh(model_v2):
    """Widen the mesh by a known factor, and the measured width scales by it."""
    ...

def _broadside_forward(model, depth=8.0):
    """Rest pose turned 90 deg about the body's long axis: the FLANK view.

    ``_rest_forward``'s identity pose puts the camera above the animal, so the
    width measured normal to the projected spine is the LATERAL extent there.
    Rotating 90 deg about y maps the model's dorsoventral axis (z) onto image x
    and its lateral axis onto depth, which is the geometry of the clips this
    repo fits -- and it is the reason a flank mask's width identifies g_z, not
    g_x.  Returns (rest verts, rest joints, R, t) so the caller can scale an
    axis BEFORE the rotation.
    """
    ...

def test_width_loss_gradient_points_at_the_target_broadside(model_v2):
    """On a FLANK view the term identifies the dorsoventral axis, not lateral.

    A projected width is measured normal to the projected spine, so which body
    axis it reads is a property of the VIEW.  Broadside -- these clips -- the
    lateral axis points at the camera and the width is the dorsoventral extent,
    so the gradient must land on the z scale and leave the x scale alone.  The
    top-down case above is the mirror image of this one, and quoting either as
    "what a silhouette width measures" without saying which view is how the
    unobserved axis gets reported as a measured biometric.
    """
    ...

def test_width_loss_gradient_points_at_the_target(model_v2):
    """The term pushes a too-thin model fatter and a too-fat model thinner.

    TOP-DOWN view (``_rest_forward``'s identity pose), so the axis in the image
    plane here is the LATERAL one; the broadside case above is the other half.
    """
    ...

def test_width_weight_zero_reproduces_the_plain_fit(model_v2):
    """Weight 0 must be bit-for-bit the old objective, width series or not."""
    ...

def test_width_data_validates_its_inputs(model_v2):
    ...

def test_width_series_config_block_and_builder(model_v2):
    ...

def test_width_targets_map_native_widths_into_proxy_pixels(model_v2):
    """The loader reads the record's own arc and scales lengths, nothing else."""
    ...

def test_width_series_error_measures_the_selected_fit(model_v2):
    """The summary's ratio is the WRITTEN fit's, measured with the fit's ruler."""
    ...

def test_width_target_reads_the_chord_and_falls_back_loudly(model_v2):
    """`target: chord` reads the extractor's chord; a record without one warns."""
    ...

def test_station_chain_length_scales_with_the_body(model_v2):
    """A body 1.1x longer projects a 1.1x longer station chain, exactly."""
    ...

def test_length_series_is_inert_at_weight_zero(model_v2):
    """Weight 0 never even looks at the target, and never changes the fit."""
    ...

def test_length_data_validates_and_drops_unmeasured_frames(model_v2):
    """A non-positive length is "the mask measured nothing", not a zero shark."""
    ...

def test_length_loss_gradient_points_at_the_target(model_v2):
    """Too long -> the gradient shortens the chain; too short -> lengthens it."""
    ...

def test_length_targets_cut_both_sides_at_the_same_arc_fraction(model_v2):
    """The mask stops at its peduncle; the model's last chord is cut to match."""
    ...

def test_length_series_config_block_and_builder(model_v2):
    ...

def test_length_series_error_measures_the_selected_fit(model_v2):
    """The summary's ratio is the WRITTEN fit's, on the fitter's own ruler."""
    ...

def test_girth_leaf_is_inert_when_disabled(model_v2):
    """Off must mean ``girth=None`` all the way into SharkSMPL.

    The whole point of the leaf's default is that a fit built after it exists
    is bit-for-bit a fit built before it did: ``apply_girth`` is skipped
    entirely rather than run at an identity field, and the identity field is
    itself bit-exact, so BOTH claims are checked.
    """
    ...

def test_girth_leaf_is_a_stage_b_parameter(model_v2):
    """Enabled, the field is fitted -- and only stage B is allowed to move it."""
    ...

def test_girth_leaf_requires_a_model_that_has_the_field(model_v2):
    """An asset with no spine chain has no girth operator, and must say so."""
    ...

def test_girth_prior_is_part_of_the_objective(model_v2):
    """The prior charges deviation from 1 and kinks along the body."""
    ...

def test_girth_init_from_targets_starts_at_the_measured_ratio(model_v2):
    """A mask that says "10% wider" must start the field near 1.1, not at 1."""
    ...

def test_girth_config_block_and_builder(model_v2):
    ...

def test_girth_summary_block_reports_the_fitted_field(model_v2):
    """The summary must say what the field became, in width-change percent."""
    ...

def test_observed_axis_uses_the_operator_the_fit_was_driven_by(model_v2):
    """A fit driven by no width term does not pay for the scanline's backward.

    The diagnostic re-poses the whole clip and backprops through a width
    operator.  On the shipped defaults -- width term at weight 0, girth leaf
    off -- nothing in the fit used the scanline, and running it here costs
    +0.87 s and +1.22 GB of peak RSS on a 300-frame clip for a number nothing
    was fitted to (girth3/fix1/axis_cost.json).  Which operator ran is part of
    the answer, because the two report different ratios (5.95 vs 8.14 on the
    demo), so it is reported alongside it.
    """
    ...

def test_silhouette_sigma_is_a_proxy_length(model_v2):
    """One softness, expressed once, scaled to whatever raster is used."""
    ...

def test_silhouette_config_block_carries_sigma_in_proxy_px(model_v2):
    ...

def test_hard_iou_diagnostic_scores_the_written_fit(model_v2, tmp_path):
    """Native resolution, every face, and the mask set the term was offered."""
    ...

def test_scanline_is_the_default_model_side_and_band_is_still_reachable(model_v2):
    """`model_side` picks the operator, and the two are measurably different.

    The band operator reads the extent of a hand-picked set of vertices; the
    scanline reads the silhouette's chord.  On real footage the first divided
    by the second was 0.885-0.911 (demo) and 1.114-1.156 (clip 2) -- opposite
    signs, and larger than the girth effect being fitted -- which is why the
    default moved.
    """
    ...

def test_scanline_width_scales_with_a_widened_mesh(model_v2):
    """A body 20% wider laterally reads 20% wider from a top-down view."""
    ...

def test_section_cut_exists_at_every_station_and_ignores_the_fin_gate(model_v2):
    """TASK D1.  The cut is a fixed, per-station, non-empty set of mesh edges.

    The along-body fraction is a REST-frame per-vertex scalar, so which edges a
    station cuts and where along each one the crossing sits are decided once at
    build time -- that is what makes the operator a fixed sparse linear map on
    the posed vertices, and as cheap as the band's gather.
    """
    ...

def test_section_width_scales_with_a_widened_mesh_and_is_the_hard_extent(model_v2):
    """TASK D1.  Exact under a known widening, and exactly the cut's max-min."""
    ...

def test_section_tau_softens_inward_and_zero_is_the_default(model_v2):
    """TASK D1.  A positive tau is strictly inside the hard extent.

    That inward bias is the whole reason the band read 0.79-0.90 of the mask
    chord at the truth: its ``width_tau_px`` is 1.0 PROXY pixel against a trunk
    half-width of 5-9 of them.  The section ships hard (tau 0) for that reason.
    """
    ...

def test_section_is_wired_into_the_objective_and_the_diagnostics(model_v2):
    """TASK D1.  ``model_side: section`` reaches the loss and the dispatch."""
    ...

def test_section_ships_off_but_reachable_from_the_config(model_v2):
    """TASK D1.  The shipped default is still the band, bit for bit."""
    ...

def test_scanline_half_length_comes_from_the_target_when_there_is_one(model_v2):
    """The reach is sized off the MASK, so the hot path measures the model once.

    Sizing it off the model's own band extent would pay for the operator the
    scanline replaces, every step.  The reach only bounds which faces and
    samples take part, so any generous value gives the same width -- which is
    what this checks.
    """
    ...

def test_width_score_exclusions_are_score_only(model_v2):
    """Excluded stations leave the REPORT, never the objective."""
    ...

def test_girth_tie_is_zero_on_an_isotropic_field(model_v2):
    """The unobserved lateral axis follows the observed dorsoventral one.

    Zero, with zero gradient, while the two axes agree; quadratic in their
    difference otherwise; and it is a PRIOR, so a view that does see g_x can
    still pull them apart for the price of it.
    """
    ...

def test_girth_tie_is_part_of_the_objective(model_v2):
    """The tie reaches the total, is weighted, and is off with the leaf off."""
    ...

def test_scanline_and_tie_config_block_and_builder(model_v2):
    ...

def _wl_fitter(model, *, normalize, focal=1000.0, pp=(128.0, 128.0), ref=100.0, tau=0.05):
    """A width-only fitter on the BAND operator (no rasteriser, any camera).

    ``tau`` is the soft extent's smoothing length and is measured in PIXELS, so
    a test that scales every image coordinate scales it too -- otherwise the
    operator itself, not the residual, is what changed.  It is small here so
    that the soft extent is within 0.1% of the hard one and the widths are an
    exact function of the projection.
    """
    ...

def _wl_residual(fitter, verts, joints, target_w, target_len):
    """Mean |residual| of ``_width_loss`` at ``width_sigma_px`` -> quadratic."""
    ...

def test_width_normalize_off_is_bit_identical(model_v2):
    """Default OFF: not one bit of the shipped width term moves.

    Both a whole fit (the objective, the optimiser, the returned pose) and the
    loss value itself, including when the caller hands in the new denominators
    -- an OFF fitter must ignore them rather than quietly using them.
    """
    ...

def test_width_normalize_is_invariant_to_scaling_the_image(model_v2):
    """ON: scale every image coordinate and the residual does not move.

    The camera's focal length and principal point are scaled by ``c`` -- which
    scales every projected coordinate, hence every projected width and length,
    by exactly ``c`` -- and so are the mask's width and length targets.  The
    absolute-pixel residual scales with ``c``; the W/L one is unchanged, which
    is the whole point of the term: depth, focal and letterbox all enter the
    image this way.
    """
    ...

def test_width_normalize_cancels_depth_but_not_pitch(model_v2):
    """ON: a UNIFORM foreshortening cancels; a length-only one does not.

    Two synthetic observations of the very same body:

    * **depth** -- the mask was measured 25% further away, so its width AND its
      length are 1/1.25 of the model's.  The absolute-pixel residual reads that
      as a 20% thinner shark; the W/L residual is ~0, because numerator and
      denominator scaled together.  This is the ambiguity the flag removes.
    * **pitch** -- the mask was measured with the snout tilted out of the image
      plane, so its LENGTH is foreshortened and its width is not.  Here it is
      the other way round: the absolute residual is ~0 and the W/L residual
      moves.  That is the trade this flag makes, and it is why the term must
      not be read as making the girth pitch-free: it swaps a depth ambiguity
      for a pitch one.
    """
    ...

def test_width_norm_data_validates_its_inputs(model_v2):
    ...

def test_width_normalize_reaches_the_config_and_the_driver(model_v2):
    ...
