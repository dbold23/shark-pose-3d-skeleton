"""3D morphometrics layer: geometry, uncertainty bookkeeping and aggregation.

What is actually asserted here, and why:

1. **The volume integrator is exact where it is defined.**  ``divergence_volume``
   is checked against a unit box and a refining icosphere -- and the same test
   shows why it is NOT the primary construction: ``mesh_watertightness`` rejects
   the shipped ``faces.npy``, which has 1954 boundary edges in 142 open loops.
2. **The station integrator is checked against closed form.**  A densely
   sampled elliptic tube and ellipsoid must recover ``pi*a*b*L`` and
   ``4/3*pi*a*b*c``, and a uniform girth field ``g = s`` must scale the girth by
   ``s`` and the volume by ``s**2`` -- the two ground truths the
   synthetic-record test cannot supply, because it compares the extraction to
   another SharkSMPL measurement.  The bin count is swept in the same place:
   the volume is monotone in it, so the band carries a named ``bins`` factor.
3. **Extraction recovers what generated it.**  A profile-v2 synthetic set
   (``scripts/generate_synthetic_data.py --profile v2``, 8 samples, seed 13)
   supplies the ground-truth shape stream; that stream is turned into a v2-rig
   fit record and extracting from it must reproduce the quantities computed
   straight from SharkSMPL at the same GT pose and betas, with every arc at
   least as long as its chord.  The generator's own pose stream is 48-dim (it
   samples the 16-joint v1 skeleton), so the 31-joint v2 pose of each record
   frame is built here from an explicit spine bend -- documented in
   ``_spine_bend_pose`` -- rather than pretending the two rigs share a
   parameterisation.
4. **Scale propagates by dimension.**  Doubling metres-per-model-unit doubles a
   length and multiplies a volume by eight.
5. **An unmet scale is absent, not zero.**  Every absolute row comes back with
   ``value is None`` and a reason.
6. **Aggregation is order-invariant and honours the gates.**
7. **The 2D tool's column names all have a counterpart.**
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.model_3d.shark_smpl import GIRTH_ENVELOPE_MIN_COUNT, SharkSMPL, select_visible_landmarks
from shark_pose.morphometrics import MorphAssets, Record, aggregate_clip, aggregate_individual, bootstrap_median_ci, boxcar_smooth, compose_band, divergence_volume, extract_clip, extract_frame, gate_frames, girth_band_corners, measurement_support, median_mad, mesh_watertightness, proxy_joints_to_native, resolve_girth_observation, resolve_individual_scale, resolve_observed_or_prior, rows_to_csv, rows_to_markdown, scale_propagate, section_profile, support_state, weighted_median
from shark_pose.morphometrics.aggregate import SHARED_BETAS_TOL, fuse_girth
from shark_pose.morphometrics.extract import AREA_RULE, AREA_RULES, GIRTH_BAND_G_X, GIRTH_BAND_G_Z, GIRTH_OBSERVED_LEVEL_ERROR_FRAC, LM_CAUDAL_UPPER_TIP, LM_SNOUT_TIP, PROFILE_BINS_SWEEP, SUPERSEDED_TEMPLATE_VOLUMES, TEMPLATE_VOLUME_CONTOUR_MODEL_U3, TEMPLATE_VOLUME_MODEL_U3, GirthFieldShapeError, template_reference
from shark_pose.morphometrics.schema import CI_KINDS, EXPECTED_RANGES, GIRTH_FIELD_QUANTITIES, PER_STATION_NEVER_OBSERVED, RANGE_ORIENTATION_EXCEPTIONS, QUANTITIES, TRUNK_ONLY_UNMEASURED, TWOD_COLUMNS, quality_grade, range_verdict_applies

@pytest.fixture(scope='module')
def assets() -> MorphAssets:
    ...

def _unit_box() -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned unit cube, outward-oriented, closed."""
    ...

def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit sphere by icosahedron subdivision -- closed and outward-oriented."""
    ...

def _spine_bend_pose(assets: MorphAssets, index: int, n: int) -> np.ndarray:
    """A 31-joint v2 pose: a travelling yaw wave down the 13-joint spine.

    This is the shape of motion the fit's own rails produce (a body wave with a
    fixed amplitude envelope), written explicitly so the test's ground truth is
    a pose, not a fitted guess.  Bending the spine is what makes arc > chord a
    non-trivial assertion.
    """
    ...

def _synthetic_record(annotations: list[dict], scale_metres_per_unit: float | None=None, assets: MorphAssets | None=None, source_name: str='TST20260907_4m_U.mp4') -> Record:
    """Wrap the synthetic GT shape stream in the shape of a v2 fit record."""
    ...

@pytest.fixture(scope='module')
def synthetic_annotations(tmp_path_factory) -> list[dict]:
    """8 profile-v2 samples, seed 13 -- the ground-truth shape stream.

    ``generate_synthetic_data --profile v2`` articulates and projects the
    16-joint v1 skeleton, so its ``pose`` is 48-dim.  What this layer needs from
    it is the GT it can actually use: the per-sample ``betas``, drawn from the
    generator's own seeded shape stream, plus the fact that they vary.  The
    31-joint v2 pose is built separately by ``_spine_bend_pose``.
    """
    ...

def test_divergence_volume_is_exact_on_a_box():
    ...

def test_divergence_volume_converges_on_the_unit_sphere():
    ...

def test_watertightness_rejects_an_open_mesh():
    ...

@requires_v2
def test_shipped_asset_is_not_watertight_so_divergence_is_rejected(assets):
    """The finding that fixes the volume construction, asserted on the asset."""
    ...

@requires_v2
def test_template_reference_reproduces_the_spec_invariants(assets):
    ...

@requires_v2
def test_synthetic_v2_extraction_recovers_its_own_quantities(assets, synthetic_annotations):
    """The GT record's quantities must equal a direct SharkSMPL measurement."""
    ...

@requires_v2
def test_arc_is_never_shorter_than_its_chord(assets, synthetic_annotations):
    ...

@requires_v2
def test_volume_is_positive_and_bounded_by_its_own_envelope(assets, synthetic_annotations):
    ...

def _grid_faces(n_axial: int, n_theta: int) -> np.ndarray:
    """Triangulate an (axial x theta) vertex grid that wraps in theta."""
    ...

def _elliptic_tube(a: float, b: float, length: float, n_axial: int=120, n_theta: int=64):
    """Elliptic tube along +y: vertices, triangles and its own axis polyline."""
    ...

def _ellipsoid(a: float, b: float, c: float, n_axial: int=240, n_theta: int=64):
    """Densely sampled ellipsoid with semi-axes (a, c, b) on (x, y, z)."""
    ...

def test_section_profile_recovers_analytic_volumes():
    """The station integral against closed-form ground truth.

    The synthetic-record test compares the extraction to another SharkSMPL
    measurement, which cannot catch an error shared by both.  These two shapes
    have volumes in closed form -- pi*a*b*L and 4/3*pi*a*b*c -- so they pin the
    integrator itself.  Since D4 the cross-section is an exact plane slice and
    the semi-axes are its extents, so the tube -- whose slice is a polygon with
    a vertex on each semi-axis -- is recovered to machine precision, and the
    only error left on the ellipsoid is the polyhedral chord between axial
    rings, which is one-sided LOW and worth 0.1% at this resolution.
    """
    ...

@requires_v2
def test_uniform_girth_field_scales_the_envelope(assets):
    """g = s must scale the girth by s and the volume by s**2 -- it overshoots.

    An independent ground truth for the girth field, which the synthetic-record
    test never exercises.  Under the superseded vertex-binned envelope the
    recovery was biased +4.7% on the girth and +2.2% on the volume at s = 1.25,
    because the girth gate blended fin vertices back into a sparsely populated
    bin as the field inflated.  The plane slice does not have that failure
    mode: every factor now lands within 0.7% of the analytic one.  TL is
    invariant.
    """
    ...

@requires_v2
def test_clip_extraction_reads_a_fitted_girth_field_in_both_record_shapes(assets, synthetic_annotations):
    """A record that fitted a girth field must be measured WITH that field.

    Records write the field twice and in two different shapes: the per-frame
    ``frames[i].girth`` is the (K, 2) array the model takes, while
    ``summary.girth.fitted`` is a mapping of ``g_x`` / ``g_z`` plus percent
    diagnostics.  Reading the summary as an array is what every record of the
    parked girth pass would have hit, so both shapes are exercised here and
    both must give the same, thinner-than-template body.
    """
    ...

@requires_v2
def test_section_area_is_not_the_elliptical_envelope(assets, synthetic_annotations):
    """pi * a * b is an ENVELOPE, and the band must carry the rule it replaced.

    The semi-axes bound the section, so the ellipse through them is not the
    section's area: on this asset it comes out ABOVE the convex hull of the very
    same contour points, which bounds the area that contour can enclose, at most
    stations.  That is the defect, and it is one-sided -- so the shipped rule is
    the hull, the band's low end is the shoelace over the same points, and the
    envelope is excluded from the band rather than being one end of it.  The
    term must reach every area, volume and mass row, must reach the metric and
    the mass rows built from them, and must NOT reach a length or a perimeter.
    """
    ...

@requires_v2
def test_volume_is_flat_in_the_bin_count_and_the_band_still_says_so(assets, synthetic_annotations):
    """PROFILE_BINS is a QUADRATURE constant since D4; the band must show it.

    The settled construction slices the mesh, so refining the cell count moves
    no envelope and the volume has a plateau: under 3% over 16/32/64/128, where
    the superseded vertex-binned envelope moved by a factor of 1.64 over the
    same sweep.  The band term is kept anyway -- a term that is small should be
    shown to be small -- and lengths must still not pick it up.
    """
    ...

@requires_v2
def test_no_station_is_under_populated_under_the_settled_construction(assets):
    """The gate and the fill rule the D1 spec never wrote down are GONE.

    The vertex-binned envelope had to invent a rule for a bin holding one to
    three body vertices, and 18 of the 64 trunk bins held 1-11 of them.  A
    plane slice cannot be starved that way: every station on this asset cuts at
    least 32 contour points out of the surface, so there is nothing to gate and
    nothing to fill.  The thin per-bin vertex counts still travel, because they
    are the measurement that condemned the old rule.
    """
    ...

def test_scale_propagation_is_linear_on_length_and_cubic_on_volume():
    ...

@requires_v2
def test_metric_values_scale_by_dimension_end_to_end(assets, synthetic_annotations):
    ...

@requires_v2
def test_metric_ci_brackets_the_metric_value_and_carries_the_scale_band(assets, synthetic_annotations):
    """ci_low/ci_high are in `unit`; value_m_ci_* is the band beside value_m.

    The bug this pins: shipping the MODEL-UNIT band beside a METRIC value.  On
    a length that band does not even bracket value_m, and on a girth the metric
    value fell BELOW ci_low -- while the scale's own +/-10-20% never reached a
    single length or area.
    """
    ...

@requires_v2
def test_P3_the_ratio_lane_does_not_move_when_the_label_does(assets, synthetic_annotations):
    """Perturb the length label by +20 % and re-aggregate: every scale-free row
    is BIT-identical and every metric value moves by exactly the label ratio
    raised to its own dimension.

    This is the property the whole per-individual programme rests on -- 111
    individuals can be given a ratio deliverable without a trustworthy absolute
    -- and until now it was only believed.  It is worth testing precisely
    because D1 made the metric BAND a function of the label (May's sigma is a
    length in centimetres, so a longer label is a narrower fraction): the band
    moving while the ratio does not is the whole point, and a leak of the label
    into a scale-free row would be invisible without this.
    """
    ...

@requires_v2
def test_range_verdict_is_withheld_where_the_two_definitions_differ(assets, synthetic_annotations):
    """A rig-geometry substitute is not range-checked against the 2D column."""
    ...

@requires_v2
def test_unmet_scale_yields_null_absolutes_with_a_reason(assets, synthetic_annotations):
    ...

@requires_v2
def test_no_row_ever_claims_a_posterior(assets, synthetic_annotations):
    ...

def test_robust_statistics_are_order_invariant():
    ...

def test_boxcar_smoother_is_zero_phase_and_truncates_symmetrically():
    ...

def test_compose_band_multiplies_independent_factors():
    ...

@requires_v2
def test_gates_reject_undetected_frames_and_report_the_count(assets, synthetic_annotations):
    ...

@requires_v2
def test_residual_gate_rejects_an_outlier_frame(assets, synthetic_annotations):
    ...

@requires_v2
def test_individual_aggregation_is_order_invariant(assets, synthetic_annotations):
    ...

@requires_v2
def test_individual_aggregation_reports_the_single_clip_case_honestly(assets, synthetic_annotations):
    ...

def _two_clips_of_one_shape(assets, synthetic_annotations) -> list[Record]:
    """Two clips of ONE animal: same betas, different poses.

    The fusion tests need the fused betas to EQUAL the per-clip betas, so that
    a fused model row differing from its per-clip rows can only be the
    estimator and not a different shape.
    """
    ...

@requires_v2
def test_fused_model_rows_are_a_pooled_statistic_not_one_frame(assets, synthetic_annotations):
    """A fused model-domain row must be a clip statistic, not a frame's.

    Regression for the defect that put fused ``TL`` 4.5032 model u BELOW both
    of its clips (4.5330 / 4.5524) on FAR18101701, and ``volume_over_TL3``
    2.7 % away from ``volume_model_units3.value_bl`` -- two names for one
    scale-free quantity -- because the value came from ONE ``extract_frame``
    at the heaviest clip's reference frame while ``value_bl`` was the mean of
    the per-clip ratios.
    """
    ...

@requires_v2
def test_fused_model_rows_use_each_frames_own_girth_field(assets, synthetic_annotations):
    """The pooled evaluation must apply the fitted girth field, not g == 1."""
    ...

@requires_v2
def test_clips_of_different_animals_are_never_fused(assets, synthetic_annotations):
    ...

def test_the_joint_fits_declared_individual_beats_the_file_name():
    """A JOINT fit's declaration of whose animal it is beats the file name.

    AN22112105's two windows come from ``AN2211210505_F_11.5.mp4`` and
    ``AN2211210508_F_11.5.mp4``; ``_ENCOUNTER_RE`` reads those 12-digit
    prefixes as two DIFFERENT encounters and splits one joint fit into two
    single-clip aggregates, each of which then refuses ``g_x`` for want of a
    second view that is sitting right beside it.  ``fit_individual.py
    --individual`` names the animal the shared girth field was tied over and
    ``check_meta`` cross-checks it against every clip's ``quality.json``, so
    the declaration is the better-evidenced answer.
    """
    ...

def test_the_file_name_still_names_the_individual_with_no_joint_fit_declaration():
    """The shipped fallback, unchanged: a record that declares nothing is parsed."""
    ...

def test_every_2d_tool_column_has_a_3d_counterpart():
    ...

def test_vocabulary_is_internally_consistent():
    ...

def test_quality_grade_rule():
    ...

@requires_v2
def test_csv_and_markdown_carry_the_vocabulary(assets, synthetic_annotations):
    ...

@requires_v2
def test_extract_frame_needs_no_record(assets):
    """The per-frame entry point works on raw parameters alone."""
    ...

def _oblique_record(annotations, assets, deg: float):
    """The synthetic clip with its aft body swung out of the image plane.

    Spine joint 5 about its first axis is the one rotation in this rig that
    moves the projected longitudinal axis off the image plane by a useful
    amount, so it is what stands in here for clip 2's geometry (43 gated
    frames, all 26-46 deg oblique).  ``deg`` 0 leaves the clip broadside.
    """
    ...

@requires_v2
def test_gate_params_carry_the_obliqueness_angle_not_only_its_count(assets, synthetic_annotations):
    """The angle is reported, not just the number of frames it rejects.

    Without it a reader of a clip table cannot tell a broadside clip from one
    filmed 42 deg off the image plane, which is the whole of what decides
    whether its projected widths and ratios mean anything.
    """
    ...

@requires_v2
def test_projected_width_rows_are_absent_on_a_clip_with_no_broadside_frame(assets, synthetic_annotations):
    """A projected WIDTH over a projected LENGTH is a broadside-only quantity.

    Off broadside the along-body denominator foreshortens and the transverse
    numerator does not, so the ratio climbs with the camera aspect and reads as
    biology.  This is what the 2D tool gates with ``is_perpendicular``.
    """
    ...

@requires_v2
def test_image_rows_on_an_oblique_clip_are_graded_low_and_withhold_value_bl(assets, synthetic_annotations):
    """``value_bl`` is the between-clip column, so it is the one to withhold."""
    ...

def test_quality_grade_takes_the_broadside_fraction():
    ...

@requires_v2
def test_pre_caudal_ratio_is_flagged_inverted_but_keeps_its_range_verdict(assets, synthetic_annotations):
    """The 2D product ships FL/TL (~0.92); this row is TL/FL (~1.085).

    The join is therefore not comparable, but ``EXPECTED_RANGES`` is quoted in
    THIS orientation, so the range verdict still applies -- which is what
    ``RANGE_ORIENTATION_EXCEPTIONS`` says and this test pins.
    """
    ...

@requires_v2
def test_fineness_ratio_ships_the_station_it_is_actually_evaluated_at(assets):
    """argmax(AREA), which is not in general max_girth_position (argmax girth)."""
    ...

def test_generic_window_stems_take_the_parent_directory_as_clip_id(tmp_path):
    """Two corpus windows of one animal are both window.mp4; their tables must not collide."""
    ...

def _girth_verdict(g_x: bool, g_z: bool, **extra) -> dict:
    """A fused girth block, reduced to what the reporting rules read."""
    ...

def test_observed_or_prior_is_computed_from_the_girth_gate():
    """A1 / R1: the label must track the fit's verdict, not the schema string.

    The defect it replaces is in the shipped E1 record, which carries
    ``girth.axes.g_x.observed = true`` and ``volume_m3.observed_or_prior =
    "prior"`` in the same file -- and grades the only observed lateral girth
    the corpus has ever produced ``low``.
    """
    ...

def test_stations_observed_gates_the_station_it_names():
    """A1 / R1, spec 4.3 clause 7: a station nobody saw is not observed."""
    ...

def test_the_clause_7_block_is_read_in_the_shape_its_producer_writes():
    """MERGE SEAM A1 x D2: one key, two branches, and they must agree.

    A1 wrote the consumer (``resolve_observed_or_prior``) against a bare list
    of per-station flags while ``girth["stations_observed"]`` had no producer
    at all; D2 then wrote the producer, and what
    ``aggregate._station_visibility`` publishes there is the whole clause-7
    block.  Read as a list, the ``not_evaluated`` block's four keys were a
    seven-station list of length four, and every girth row of every real
    record raised ``ValueError`` instead of reporting.  Both shapes are read,
    and a shape that is neither is refused rather than assumed.
    """
    ...

def test_quality_grade_follows_the_resolved_label():
    """A1: the grade is a function of the verdict, so `low` stops being fixed."""
    ...

@requires_v2
def test_prior_girth_corners_are_unchanged_by_the_observed_path(assets):
    """A2 / R2: with no observed axis the corners are corner-for-corner today's.

    The prior path is the shipped behaviour on every record in the corpus bar
    one, and the acceptance test for this change is that it does not move.
    """
    ...

@requires_v2
def test_an_observed_axis_is_banded_at_its_own_fitted_field(assets):
    """A2 / spec 5.3: fitted x (1 -+ 8 %), per axis, and only where observed."""
    ...

def _joint_pair(assets, annotations, separation_deg=46.529, tie_weight=0.0):
    """Two clips of one animal from ONE joint fit, with a shared girth field.

    The shape of the records the corpus actually produces: both clips carry the
    same ``girth.joint_fit`` id, the same shared ``(7, 2)`` field, and the
    per-clip width Jacobians the gate scores.

    The two clips are also drawn from two DIFFERENT source videos, recorded
    909 s apart -- the real AN15092101 pair's own geometry (12ft_03.mp4
    21:00:13Z vs 12ft_07.mp4 21:15:22Z).  Gate clause 6 (clip independence)
    refuses a pair of windows cut from one video, so a fixture that left both
    clips on the default ``source_name`` would be testing the girth
    OBSERVATION against a pair the gate is right to refuse.  The encounter
    token ``TST20260907`` is unchanged in both names, so the two clips remain
    one individual, and the ``4m`` length label is unchanged too.
    """
    ...

@requires_v2
def test_the_observed_band_lands_only_on_a_joint_record(assets, synthetic_annotations):
    """A2 end to end: the two passes, and the single-clip dependency spec 5.3 states."""
    ...

@requires_v2
def test_a_joint_record_reports_its_girth_observed_and_stamps_the_floor(assets, synthetic_annotations):
    """A1 + A2 on a whole record: the label, the grade and the stamped words."""
    ...

@requires_v2
def test_a_refused_lateral_axis_keeps_the_prior_label_and_the_prior_band(assets, synthetic_annotations):
    """A1 + A2, the FAR20110101 case: separation under the bar -> g_x PRIOR."""
    ...

@requires_v2
def test_P4_the_published_band_can_be_audited_from_its_own_components(assets, synthetic_annotations):
    """Test P4, as amended: a product per CLIP row, an envelope per INDIVIDUAL row.

    The old form of this test was unsatisfiable on any N >= 2 record: a fused
    row's ``ci_components`` are ONE clip's, copied, and the band is the
    envelope of the per-clip intervals plus the between-clip spread -- so the
    product of the named terms reproduces the SOURCE CLIP's band and not the
    row's.  Both identities are asserted here, from the published numbers alone.
    """
    ...

@requires_v2
def test_a_truncated_clip_never_reports_a_length_or_a_volume(assets, synthetic_annotations):
    """D3 / R10, spec 3.4: the TRUNK_ONLY contract, on a synthetic truncated record."""
    ...

@requires_v2
def test_an_untested_clip_is_untested_not_whole(assets, synthetic_annotations):
    """D3: silence is not a clean lane, and it does not refuse anything either."""
    ...

@requires_v2
def test_a_joint_record_stops_calling_its_tied_shape_a_fallback(assets, synthetic_annotations):
    """D4 / R13: `FALLBACK` beside a `correct_estimator` the fit already used."""
    ...

@requires_v2
def test_a_mixed_individual_builds_its_length_from_the_whole_clips_only(assets, synthetic_annotations):
    """D3 / spec 3.4, the case a two-clip programme actually produces.

    The refusal was enforced on the CLIP row while the individual's fused value
    was pooled over every clip's frames: a TRUNK_ONLY clip's frames entered the
    individual's TL, volume and mass, which were then neither the whole clip's
    number nor a refusal, and the block beside them said `trunk_only`.
    """
    ...

@requires_v2
def test_a_contested_individual_publishes_no_absolute_on_any_of_its_clips(assets, synthetic_annotations):
    """D1 / spec 4.6 clause 4, both levels.

    ``_fuse_scale`` refused the individual while every ``clip_*.json`` beside it
    in the same output directory published a metric scale and every
    ``value_m`` -- two contradictory scales in one directory -- and ``_fuse_row``
    put the clip-fused m3/kg numbers back on the individual as well.
    """
    ...

@requires_v2
def test_two_single_clip_fits_do_not_take_the_two_view_girth_floor(assets, synthetic_annotations):
    """A2 / spec 5.3: the +-8 % floor was measured on the two-view ARM.

    Gating it on the clip COUNT let two independent single-clip fits of one
    animal take it, in a record whose own shape block correctly reports
    FALLBACK -- "these shapes were not jointly fitted" and "this girth carries
    the two-view level floor" in one file.
    """
    ...

@requires_v2
def test_the_girth_band_provenance_describes_the_band_that_was_used(assets, synthetic_annotations):
    """A2: ``girth_band_scope`` asserted the honest_band on a record whose band
    had been rebuilt at the observed floor, three keys from the block saying
    so -- under the older, more familiar key name."""
    ...

@requires_v2
def test_aggregating_an_individual_blind_to_its_clips_verdict_is_refused(assets, synthetic_annotations):
    """The individual recomputed a verdict its own clip rows never saw, and
    could label a row ``observed`` on top of clip rows labelled ``prior``."""
    ...

def _letterboxed_camera(width: int, height: int, res: int=256) -> dict:
    """The camera block ``fit_video`` stamps for a ``width x height`` clip."""
    ...

def _stamp_edge_gate(record, camera: dict, ends_native, margin_frac: float=0.03):
    """Put ``ends_native`` (2 x 2, native px) on every frame's spine ends."""
    ...

def test_the_gate_maps_the_proxy_principal_point_to_the_image_centre_16x9():
    """The defect, stated as one projection.

    ``joints_2d_proxy`` sits on a square letterboxed canvas.  The gate used to
    undo that with the scale alone (``width / proxy_resolution``), which sends
    the proxy principal point of a 1920x1080 frame to y = 960 instead of
    y = 540 -- 420 px = (W - H) / 2 below the image centre, 38.9 % of the frame
    height, enough to put a centred animal past the bottom border.
    """
    ...

def test_the_gate_is_bit_identical_on_a_4x3_frame_with_zero_pad():
    """A 4:3 frame letterboxes with off_y = (W - H)/2 too -- but a SQUARE
    source pads by nothing, and that is the case the old scale-only map got
    right.  Both are asserted: the square frame is bit-identical to the old
    ``x * width / proxy_resolution``, the 4:3 frame is not.
    """
    ...

@requires_v2
def test_a_centred_animal_is_not_edge_truncated_on_a_16x9_clip(assets, synthetic_annotations):
    """End to end through ``gate_frames``: an animal spanning the middle half
    of a 1920x1080 frame must lose zero frames to ``truncation_edge``.  With
    the pad dropped its tail end lands at y = 1160 on a 1080 px frame and every
    frame is discarded.
    """
    ...

def test_support_state_is_the_rule_it_publishes():
    """``ok`` needs BOTH a fraction and an absolute floor; below 0.10 nothing."""
    ...

@requires_v2
def test_measurement_support_counts_frames_measured_of_frames_fitted(assets, synthetic_annotations):
    """The clip block is the gate budget, first class beside the values."""
    ...

@requires_v2
def test_individual_support_is_the_sum_over_clips_and_keeps_the_breakdown(assets, synthetic_annotations):
    """290/300 and 5/300 must not be readable as one 295/600 (S3 B rule 3)."""
    ...

@requires_v2
def test_a_record_carrying_measurement_support_is_json_identical_elsewhere(assets, synthetic_annotations):
    """The block is ADDITIVE: drop it and the record is byte-for-byte the old one.

    The legacy key order is written out here rather than diffed against a
    sibling call, so a future reordering or a quietly changed value anywhere
    else in the record fails this test.
    """
    ...

def _json_default_for_test(obj):
    ...

@requires_v2
def test_insufficient_support_publishes_a_reason_instead_of_a_value(assets, synthetic_annotations, monkeypatch):
    """S3 B rule 1, behind its flag: default OFF leaves every value in place."""
    ...

def test_vocabulary_splits_the_broadside_clause_by_domain():
    """The clause survives exactly where a projection is being gated."""
    ...

@requires_v2
def test_model_space_rows_now_aggregate_over_keep_and_nothing_else_moves(assets, synthetic_annotations, monkeypatch):
    """The change, stated as the only difference it makes.

    A clip whose frames straddle the 20 deg threshold has a broadside subset
    that is a STRICT subset of ``gates.keep``.  On such a clip each of the six
    model-space rows must now equal the median over ``gates.keep`` of its own
    per-frame series, and every other row must be byte-identical to the same
    record aggregated with the old vocabulary.
    """
    ...

@requires_v2
def test_model_space_rows_survive_a_clip_with_no_broadside_frame(assets, synthetic_annotations):
    """The 15-of-280 case: the clause used to DELETE these rows outright."""
    ...
