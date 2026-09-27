"""Tests for silhouette centreline extraction and the canonical bend metric.

The synthetic bodies are built by sweeping a tapering disk along a known
centreline curve: straight over the front 55%, then a constant-curvature arc so
that the head-chord / tail-chord angle of the *generating* curve is the target
bend.  Ground truth is the metric applied to that generating curve, so the test
measures extraction error alone, not the metric's definition.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
from shark_pose.sim2real.mask_centerline import AXIS_CHOICES, ORIENT_BAND, QUALITY_GATE_DEFAULTS, RIG_ARC_SPAN, RIG_ARC_SPAN_PEDUNCLE, SPAN_TAIL_MAX, bend_of_projected_stations, centerline_quality, chord_bend_deg, chord_width, chord_widths_along_polyline, edge_tests, extract_axis, extract_centerline, extract_opening_axis, extract_trunk_axis, trunk_cut_ratio, fin_contamination, mask_edge_contact_px, mask_perimeter_px, orientation_score, peduncle_cut_index, peduncle_index, polyline_edge_distance_px, polyline_span_length_px, resample_polyline, sample_polyline_at_fractions, station_arc_fractions, station_bend, tail_deflection_pct, trim_to_span, ValidRegion
from extract_mask_centerline import WEAK_VOTE_FRAC, add_chord_widths, add_trunk_and_stations, frame_valid_region, load_masks, max_lateral_separation, measure_cut_ratio, sequence_orientation, sequence_quality_stats
_TURN_TO_CHORD = 1.742

def _spine(bend_deg: float, n: int=400, width: int=700, height: int=300) -> np.ndarray:
    """Generating centreline: straight front, constant-curvature back."""
    ...

def _body_mask(bend_deg: float, fins: bool=False, caudal: bool=False, width: int=700, height: int=300):
    """Return (mask, generating centreline) for a shark-like swept body.

    ``fins`` adds a dorsal + pectoral blob; ``caudal`` adds a big fanned tail
    fin at the posterior tip, which is what breaks an outer-tip orientation
    rule (a spread caudal fin is wider than the snout).
    """
    ...

def _ground_truth(pts: np.ndarray) -> float:
    ...

def test_sign_convention_head_along_plus_x():
    """Head along image +x, tail bending toward image +y (down) => positive."""
    ...

def test_straight_polyline_has_zero_bend():
    ...

def test_bend_is_translation_and_rotation_covariant():
    ...

@pytest.mark.parametrize('target', [0.0, 15.0, -15.0])
def test_known_bend_recovered_within_2_deg(target):
    ...

@pytest.mark.parametrize('target', [15.0, -15.0])
def test_tail_deflection_sign_follows_bend(target):
    ...

@pytest.mark.parametrize('target', [0.0, 15.0, -15.0])
def test_fin_blobs_do_not_derail_the_centerline(target):
    """A dorsal + pectoral blob may bias the bend but must not break it.

    Measured bias with these (deliberately oversized) blobs is -1.4 to -2.9 deg,
    hence the 4 deg tolerance; the point of the test is that the line still spans
    the body and stays inside the silhouette.
    """
    ...

def test_second_blob_is_ignored():
    """A detached blob elsewhere in the frame must not join the body."""
    ...

def test_ordering_follows_snout_hint():
    ...

def test_ordering_without_hint_uses_the_inner_width_band():
    ...

def test_orientation_score_is_antisymmetric_and_signs_the_head():
    ...

def test_orientation_score_reads_the_inner_band_only():
    """Only arc 15-40% from each end may move the score."""
    ...

def test_caudal_fin_defeats_the_outer_tip_rule_but_not_the_inner_band():
    """A fanned tail fin is wider than the snout; the inner band is not fooled.

    This is the 2026-09-04 failure mode on real SAM masks (17/150 frames): the
    outer 15% compares rostrum against caudal fin, so it votes tail-first.  The
    15-40% band compares branchial trunk against caudal peduncle instead.
    """
    ...

def _peduncle_terminated_widths(n: int=32) -> np.ndarray:
    """Head-first width profile of a snout-tip -> PEDUNCLE (dt_path) axis.

    Shaped after the median profile the shipped dt_path records measure: a thin
    rostrum, the girth maximum at 0.60 of the arc (the axis stops at the
    peduncle, so what is mid-body on the whole animal lands late here), and a
    thin peduncle at the end.
    """
    ...

def _whole_animal_widths(n: int=32) -> np.ndarray:
    """Head-first width profile of the same body's snout -> caudal-TIP axis.

    The girth peak now sits at 0.25 of the arc and a fanned caudal fin sits at
    the far tip -- the profile ORIENT_BAND was measured on.
    """
    ...

def test_orient_band_inverts_on_a_peduncle_terminated_profile():
    """The band is only meaningful when its MIRRORED window is the peduncle."""
    ...

def _orient_records(n_records: int, dissenting: int=0, thinning: bool=True):
    """``n_records`` head-first records sharing one polyline (gauge all +1).

    The first ``dissenting`` records carry a REVERSED thinning profile, which
    is how a single frame votes against the sequence once the gauge has already
    put every polyline the same way round.
    """
    ...

def test_sequence_orientation_votes_on_the_thinning_profile():
    """The fix: the vote reads ``widths_thinning``, so it is not inverted.

    Scoring the band on the dt_path ``widths`` instead is the 2026-09-05
    regression -- on the shipped clip-2 records that vote is 27 of 65 for, a
    minority, and a hint-less run inverted the whole bend series.
    """
    ...

def test_sequence_orientation_falls_back_to_widths_without_a_thinning_profile():
    """An ``--axis thinning`` run has no ``widths_thinning``; ``widths`` IS it."""
    ...

def test_sequence_orientation_flags_a_weak_vote():
    """A near-tied whole-clip decision must not be silent."""
    ...

def test_orientation_survives_fins_and_caudal_together():
    ...

def test_peduncle_index_finds_the_posterior_width_minimum():
    ...

def test_hint_flips_the_bend_sign():
    """Reversing the polyline reverses the sign of the bend metric."""
    ...

@pytest.mark.parametrize('n_points', [16, 32, 64])
def test_output_shapes_and_arc_length_spacing(n_points):
    ...

def test_widths_track_the_taper():
    ...

def test_empty_and_tiny_masks_return_not_ok():
    ...

def test_non_2d_mask_raises():
    ...

def test_uint8_mask_is_accepted():
    ...

def test_projected_stations_score_like_a_mask_centerline():
    """13 spine stations sampled off the true centreline must score the same."""
    ...

def test_projected_stations_agree_with_extracted_centerline():
    ...

def test_projected_stations_reversal_flips_sign():
    ...

def test_trim_to_span_is_exact_on_a_straight_line():
    ...

def test_trim_to_span_full_span_is_the_identity():
    """Span (0, 1) returns the same curve: exact on a straight polyline, and to
    the input's own chord-vs-arc rounding (<0.01 px here) on a bent one."""
    ...

def test_trim_to_span_resamples_and_validates():
    ...

def test_trimming_removes_the_caudal_fin_from_the_bend():
    """The full-body bend is dominated by the fin; the trunk bend is not."""
    ...

def test_station_arc_fractions_map_into_the_span():
    ...

def test_sample_polyline_at_fractions_hits_the_arc_positions():
    ...

def test_station_bend_is_bend_of_projected_stations_on_the_same_points():
    """The mask side and the fit side must be literally the same function."""
    ...

def test_station_bend_tracks_the_trunk_bend_not_the_fin():
    ...

def test_station_bend_reversal_flips_sign():
    """Same stations, polyline reversed: the mirrored fractions, mirrored sign."""
    ...

def _record_from_mask(mask, span=RIG_ARC_SPAN, spine_fraction=None):
    """Extract one record the way the script does, provenance and all."""
    ...

def _clean_stats(rec):
    """Sequence statistics for which ``rec`` is a perfectly typical frame."""
    ...

def test_mask_edge_contact_counts_the_border_band():
    ...

def test_mask_edge_contact_counts_a_corner_pixel_once():
    ...

def test_mask_edge_contact_margin_is_clamped_to_one_pixel():
    ...

def test_quality_passes_a_clean_typical_frame():
    ...

def test_quality_rejects_a_mask_touching_the_frame_edge():
    """The animal swimming out of frame: same body, shifted until it is cut."""
    ...

def test_quality_edge_test_respects_the_margin():
    ...

@pytest.mark.parametrize('length_rel, expected', [(0.4, False), (0.49, False), (0.55, True), (1.0, True), (1.55, True), (1.8, False)])
def test_quality_gates_the_relative_arc_length(length_rel, expected):
    ...

def test_quality_rejects_a_bend_outlier_but_not_an_unusually_straight_frame():
    ...

def test_quality_thresholds_are_overridable():
    ...

def test_quality_degenerate_sequence_statistics_do_not_crash():
    """MAD 0 must not reject on bend; a zero median length fails on length."""
    ...

def test_quality_reads_an_old_record_without_the_new_fields():
    """Backwards compatibility: a pre-quality record still scores."""
    ...

def test_quality_never_passes_a_not_ok_record():
    ...

def test_quality_rejects_a_non_2d_mask():
    ...

def test_records_carry_the_span_and_station_fractions_they_were_measured_at():
    ...

def test_provenance_is_stamped_even_when_extraction_failed():
    ...

def test_sequence_quality_stats_are_robust_to_the_frames_they_gate():
    """Six normal frames plus two truncated ones: the medians follow the six."""
    ...

def _triangle(vertices, yy, xx):
    """Filled triangle mask from three (x, y) vertices."""
    ...

def _finned_body(bend_deg: float, width: int=700, height: int=340):
    """A shark-shaped body whose true midline is known: swept tapering disks
    plus a TALL dorsal fin, a pectoral blade and a forked heterocercal tail.

    The fins are tapering triangles, as real fins are, which is the whole point:
    their inscribed circles are small, so the distance transform's core ignores
    them while an area-based estimator does not.
    """
    ...

def _deviation_px(polyline, truth, lo: float=0.0, hi: float=1.0) -> float:
    """Largest distance from a stretch of a polyline to the true midline CURVE."""
    ...

@pytest.mark.parametrize('bend', [0.0, 15.0, -15.0])
def test_dt_path_holds_the_midline_where_the_slice_mean_is_pulled_into_the_fins(bend):
    """Both halves of the defect, on a body whose midline is known.

    Measured on this geometry: over the stretch the generating curve covers
    (arc 0.15-1.0 of the extracted axis) the distance-transform path stays
    within 1.8 px of the true midline, while the slice mean of
    `extract_centerline` is dragged 16.7-17.8 px off it in the dorsal-fin
    stretch (arc 0.15-0.6) and 66-100 px off overall, because it follows the
    fork of the caudal fin.  The dt path's first 15% is the ROSTRAL CAP: the
    body is swept disks, so the silhouette reaches 48 px forward of the
    generating curve's first point and the axis is anchored out there
    (`test_dt_path_head_anchor_is_the_silhouette_not_the_hint`), which is why
    the deviation is measured against the curve only where the curve exists.
    """
    ...

def test_dt_path_snaps_an_off_mask_snout_hint_onto_the_silhouette():
    """An off-mask hint lands on the silhouette, and the axis starts on it.

    With `extend_head` off the endpoint IS the snapped hint (this is what the
    axis did before 2026-09-04); with it on the endpoint is the silhouette's
    own forward-most point, which is the subject of
    `test_dt_path_head_anchor_is_the_silhouette_not_the_hint`.  Either way the
    snout endpoint is a mask pixel and it is the polyline's first sample.
    """
    ...

@pytest.mark.parametrize('bend', [0.0, 15.0, -15.0])
def test_dt_path_head_anchor_is_the_silhouette_not_the_hint(bend):
    """The snout endpoint must not move when the HINT moves.

    The 13 station fractions are fractions of the snout -> cut arc, so an
    anchor that slides between frames slides all 13 -- the same contaminant the
    fixed peduncle cut removes at the other end.  The hint is a detector
    keypoint and lands anywhere from the rostrum to the gills, so the axis
    takes only the DIRECTION from it and walks forward onto the outline.

    Measured on this geometry (a swept-disk body whose silhouette reaches
    48.0 px forward of the generating curve's first point): with hints placed
    at 0%, 10% and 20% of the curve, the extracted snout endpoint moves by
    <= 0.02 px and stops 1.5-2.0 px short of the silhouette's forward reach,
    while the un-extended axis's endpoint moves 103 px and stops 53-106 px
    short.
    """
    ...

@pytest.mark.parametrize('bend', [0.0, 15.0])
def test_dt_path_ends_at_the_peduncle_not_at_the_tail(bend):
    ...

def _path_arc(pts) -> float:
    ...

def test_peduncle_cut_index_takes_a_local_minimum_not_the_thinnest_point():
    """The width profile of a real path: trunk, waist, fin fan, fin tip.

    A plain minimum over the back of the line lands on the fin TIP; the cut
    must land on the waist at arc 0.69.
    """
    ...

def test_peduncle_cut_index_falls_back_to_the_waist_when_nothing_is_local_min():
    """A forked, blade-thin tail leaves no local minimum to cut at."""
    ...

def test_peduncle_cut_index_hint_picks_the_nearest_minimum():
    ...

def test_peduncle_cut_index_reports_far_end_when_there_is_no_waist():
    ...

def test_fin_contamination_counts_thin_samples_and_ignores_n_points():
    ...

@pytest.mark.parametrize('bend', [0.0, 15.0])
def test_fin_contamination_is_higher_for_the_slice_mean_axis(bend):
    ...

def test_extract_axis_stamps_the_contract_fields_on_every_axis():
    ...

def test_extract_axis_thinning_is_extract_centerline_untouched():
    ...

def test_extract_axis_rejects_an_unknown_axis():
    ...

def test_trunk_axis_on_an_empty_mask_is_not_ok_but_still_stamped():
    ...

def test_trunk_axis_without_a_hint_uses_the_inner_band_rule():
    ...

def test_opening_axis_removes_the_fins_but_eats_the_ends():
    """The cross-check, and why it is only that.

    Measured here and on the 150 demo masks: opening does lower the fin
    contamination (0.102 against 0.227 for dt_path and 0.273 for the slice
    mean), but an opening cannot preserve a taper, so the axis is much shorter
    than the animal (demo median arc 1292 px against 1569 for dt_path and 2020
    for the slice mean) and its station bend reads 28.1 deg median against
    4.47 -- not the body's beat.
    """
    ...

def test_rig_arc_span_peduncle_matches_the_v2_template():
    """Re-measure the peduncle span from the rig, the way the docstring says.

    Snout = template y max; peduncle = the y of the narrowest mesh
    cross-section behind the second dorsal (minimum of sqrt(lateral x
    dorsoventral extent) of the plane section); the two spine stations are
    read from skeleton.json.

    What this pins is the RECIPE, not a frozen pair of numbers: the span is a
    property of the template, and ``scripts/reshape_template.py`` moves it
    (the shipped, rostrum-extended asset measures 0.0919 / 1.0425 against the
    module constant's pre-reshape 0.0579 / 1.0443 -- a 3.4%-of-body move, which
    is why :data:`RIG_ARC_SPAN_PEDUNCLE` is documented as a pre-reshape
    reference and callers measure the asset in hand).  So this independent
    re-implementation is checked against
    :func:`mask_centerline.measure_rig_arc_span` on the same asset; the two
    differ only by that function's parabolic refinement of the 0.005 search
    grid and by skeleton.json's 4-dp rounding of the joint y's, hence 2e-3.
    """
    ...

def test_station_arc_fractions_accepts_a_span_past_the_polyline_end():
    ...

def test_sampling_past_the_end_clamps_unless_asked_to_extrapolate():
    ...

def test_station_bend_extrapolation_moves_only_the_last_station():
    ...

def test_mask_perimeter_counts_the_outline():
    ...

def test_polyline_edge_distance_takes_the_nearer_end():
    ...

def _trunk_record(mask, snout_hint):
    """One record the way the script builds it, on the fin-free trunk axis."""
    ...

def test_quality_grades_frame_edge_contact_instead_of_flagging_it():
    """A clipped caudal fin keeps its record; a snout out of frame loses it.

    This is the case the boolean edge test got wrong on the demo clip (95 of
    150 whole-animal frames rejected).  With an axis that ends at the peduncle,
    a caudal fin sliced off by the border costs a few percent of the outline
    and leaves both ends of the axis deep inside the frame.
    """
    ...

def test_quality_edge_end_margin_is_overridable_in_pixels():
    ...

def test_quality_rejects_a_mask_that_is_mostly_frame_border():
    """A long run of outline replaced by the frame edge: the animal is cut open.

    The end test is switched off here so that the contact fraction is the only
    thing that can reject: it is the second half of the graded edge test, and
    catches a silhouette sliced lengthwise, where both ends can still sit well
    inside the frame.
    """
    ...

def test_edge_contact_beyond_the_axis_ends_is_not_gated():
    """A caudal fin sliced off by the border is not a cut TRUNK.

    The gated fraction counts only border contact between the axis's two end
    planes.  On the demo clip the unscoped count rejected 48 of 150 records
    whose snout -> peduncle axis sat 100 px or more inside the frame at both
    ends and whose contact was entirely behind the peduncle.
    """
    ...

def test_fixed_cut_ratio_replaces_the_per_frame_peduncle_cut():
    """`cut_ratio` puts the cut where it is told, and says so.

    The 13 stations are fractions of the snout -> cut arc, so a cut that moves
    from frame to frame slides all 13 along the body; the extractor holds it at
    the sequence median instead (measured: cut scatter cv 6.96% -> 0.14% on the
    demo, 7.45% -> 0.16% on clip 2).
    """
    ...

def _write_masks(tmp_path, masks) -> list:
    ...

def test_cut_ratio_pools_only_over_frames_the_edge_gate_keeps(tmp_path):
    """The sequence cut must not be set by the frames the gate throws away.

    The cut is a fraction of the uncut path's arc and every station is a
    fraction of the snout -> cut arc, so a truncated silhouette that votes in
    the median slides all 13 stations on the frames that ARE fitted.  Measured
    on clip 2 (32 of 65 masks truncated): all-frames median 0.789 against
    0.828 over the frames the gate keeps -- 3.85% of the arc, RMS 2.12 deg of
    station bend.  Here the same asymmetry is built by hand: three whole
    bodies and three cut off at the left border.
    """
    ...

def test_cut_ratio_falls_back_to_every_frame_when_the_gate_keeps_none(tmp_path):
    """A clip that is truncated end to end still gets a cut, and says so."""
    ...

def test_max_lateral_separation_ignores_parameterisation_and_sees_a_detour():
    """The script's old-vs-new axis number: 0 for the same curve, high for a
    line that goes somewhere else."""
    ...

def test_script_record_handles_a_span_that_runs_past_the_polyline():
    """add_trunk_and_stations with the peduncle span: the trunk is clamped to
    the line, the stations are extrapolated past its end."""
    ...

def _straight_tapered_body(width: int, height: int, y0: float, x0: int, x1: int, r_head: float=26.0, r_tail: float=7.0) -> np.ndarray:
    """A perfectly straight, tapering body lying along ``y = y0``.

    Straight by construction, so a bend metric that reports anything but ~0 on
    it is reporting the measuring apparatus rather than the animal.
    """
    ...

def _through_the_lens(camera, ideal_mask: np.ndarray) -> np.ndarray:
    """Push a pinhole-frame mask THROUGH a lens: the raw frame a camera sees.

    The inverse of :meth:`CameraIntrinsics.undistort_image`, built from the same
    point map: every distorted pixel samples the ideal image at the position it
    undistorts to.
    """
    ...

def test_undistorting_the_mask_removes_the_lens_bend_from_a_straight_body():
    """The whole point of ``--camera``: a straight animal must read as straight.

    A straight tapering body is placed off the optical axis, pushed through a
    GoPro Wide lens, and extracted twice.  On the raw frame the metric reports
    ~10 deg of bend that is entirely the lens; undistorting the mask first
    brings it back to the ideal frame's own answer.
    """
    ...

def test_undistorted_masks_are_still_binary():
    """NEAREST interpolation only: a mask remapped bilinearly stops being one."""
    ...

def _inset_region(height: int, width: int, inset: int) -> np.ndarray:
    """A stand-in for an undistorted frame's valid area: a rectangle inset on
    the left, i.e. a straight-sided black 'wedge' of known width."""
    ...

def test_full_valid_region_reproduces_the_rectangle_tests_exactly():
    """The ``--camera none`` guarantee, at the level of the primitives."""
    ...

def test_border_band_follows_the_valid_regions_boundary():
    """Contact is counted at the wedge boundary, not at the canvas edge."""
    ...

def test_valid_region_edge_distance_measures_the_curved_boundary():
    """A polyline end deep inside the canvas can be ON the observation's edge."""
    ...

def test_valid_region_shape_must_match_the_frame():
    ...

def test_quality_gate_rejects_a_truncation_the_canvas_rectangle_cannot_see():
    """An animal cut off at the WEDGE boundary is truncated, and must reject.

    Without a valid region the axis's ends sit far from the canvas edge and the
    gate keeps a part-animal; with one it rejects on ``edge_near_end_px``.
    """
    ...

def test_frame_valid_region_is_the_undistorted_canvas():
    """The region the script builds is smaller than the canvas and cached."""
    ...

def test_extractor_stamps_the_camera_and_undistorts(tmp_path):
    """End to end: --camera stamps every record and moves the polylines."""
    ...

def test_chord_width_is_the_full_chord_not_the_inscribed_diameter():
    """A rectangle's chord is its width; the distance transform's is not.

    This is the whole point of the operator.  Twice the distance transform at
    an axis point is the diameter of the largest INSCRIBED circle, so it is
    bounded by the NEARER of the two walls; the chord reaches both.  Off-axis
    in a slab the two answers differ by exactly the offset, and that difference
    is what made "width ratio 1" ask for two different animals.
    """
    ...

def test_chord_width_picks_the_run_nearest_the_station():
    """Two blobs on the ray: the one the station sits in wins, else the closer."""
    ...

def test_chord_width_subpixel_agrees_with_the_scorer_s_nearest_march():
    """``subpixel=False`` IS e2_score's ``_chord_width``; on is within a step."""
    ...

def test_chord_widths_along_polyline_uses_the_local_normal():
    """A tilted slab is measured across, not along, whatever the image axes do."""
    ...

def test_polyline_span_length_is_exact_on_a_straight_polyline():
    """The mask-side cut of the length term, on a curve whose answer is known."""
    ...

def test_extractor_stores_the_chord_beside_the_distance_transform():
    """``add_chord_widths`` adds a ruler; it does not touch the old one."""
    ...
