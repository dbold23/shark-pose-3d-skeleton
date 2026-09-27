"""Tests for the pure helpers in scripts/eval_detector_marginal.py.

Only the geometry / scoring / argument-parsing helpers are covered here; the
inference stages need a checkpoint, a detector and the real frames, so they
are exercised by running the script itself.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from eval_detector_marginal import DIVERGENCE_DEPTH_BAND, DIVERGENCE_FRAME_SCALE, detector_row, extra_divergent_for, extra_divergent_mask, flag_divergent_frames, guard_row, load_extra_divergent_ids, load_subset_ids, markdown_table, offset_distribution, parse_detector_specs, resolve_geometry, score_row, stem_for_sample_id, to_proxy_xy
from shark_pose.sim2real.detector_weights import V4_WEIGHTS
from shark_pose.sim2real.proxy_geometry import image_to_proxy_xy

def test_resolve_geometry_defaults_to_legacy_squash():
    ...

def test_resolve_geometry_rejects_unknown():
    ...

def test_squash_geometry_is_the_anisotropic_scaling():
    ...

def test_letterbox_geometry_matches_shared_helper():
    ...

@pytest.mark.parametrize('geometry', ['squash', 'letterbox'])
def test_geometry_is_invariant_to_uniform_frame_rescale(geometry):
    """Labels stored against 640x360 map the same as detections on 1920x1080."""
    ...

def test_unknown_geometry_raises():
    ...

def test_stem_for_sample_id_strips_prefix_only_when_present():
    ...

def test_load_subset_ids_from_data_dir(tmp_path):
    ...

def test_load_subset_ids_from_index_json(tmp_path):
    ...

def test_load_subset_ids_missing_path(tmp_path):
    ...

def test_parse_detector_specs_defaults_to_v5_then_v4():
    ...

def test_parse_detector_specs_explicit():
    ...

def test_parse_detector_specs_rejects_malformed():
    ...

def _fixture():
    ...

def test_score_row_only_counts_visible_joints():
    ...

def test_score_row_detected_split():
    ...

def test_score_row_handles_nothing_visible():
    ...

def test_detector_row_coverage_and_false_positives():
    ...

def _guard_fixture(n=4, resolution=256):
    """n frames of joints comfortably inside the frame, at a sane root depth."""
    ...

def test_flag_divergent_frames_clear_on_a_sane_fit():
    ...

def test_flag_divergent_frames_catches_a_joint_projected_to_infinity():
    ...

def test_flag_divergent_frames_catches_a_root_on_the_camera_plane():
    ...

def test_flag_divergent_frames_reports_both_reasons_together():
    ...

def test_flag_divergent_frames_catches_non_finite_coordinates():
    ...

def test_flag_divergent_frames_is_blind_to_undetected_and_invisible_joints():
    """Every joint counts: the guard sees the fit, not the labels."""
    ...

def test_flag_divergent_frames_without_depth_uses_the_pixel_test_only():
    ...

def test_flag_divergent_frames_threshold_is_k_half_frames():
    ...

def test_flag_divergent_frames_depth_band_edges():
    ...

def _guard_row_inputs():
    """3 frames x 2 visible joints; frame 1's root is on the camera plane.

    Frame 1's joints stay inside k half-frames on purpose, so the ONLY thing
    flagging it is the depth test -- which is the val89 frame-81 signature and
    keeps the two tests separable.
    """
    ...

def test_guard_row_excludes_the_divergent_frame_and_says_so():
    ...

def test_guard_row_never_edits_the_as_is_row():
    ...

def test_guard_row_with_nothing_divergent_repeats_the_row_exactly():
    """The bit-identity property every unaffected row relies on."""
    ...

def test_guard_row_scores_over_the_union_of_both_arms_divergent_frames():
    """The defect this exists for.

    Arm A breaks on frame 1 only, arm B on frames 1 and 2.  Scored over their
    OWN sets the two means are taken over different observations, and A is
    credited for not being charged for the frame only B broke on.  Under the
    union both exclude {1, 2}.
    """
    ...

def test_guard_row_union_accepts_a_bool_mask_too():
    ...

def test_guard_row_without_extra_divergent_is_unchanged():
    """No ``extra_divergent`` must be the row every published number used."""
    ...

def test_extra_divergent_mask_maps_sample_ids_not_indices():
    ...

def test_extra_divergent_mask_raises_on_an_id_outside_the_subset():
    ...

def test_load_extra_divergent_ids_groups_a_harness_output_by_detector():
    """Pooling v5's and v4's frames would drop v4's frames from the v5 row."""
    ...

def test_offset_distribution_separates_the_converged_body_from_a_runaway():
    ...

def test_guard_row_records_the_offset_distribution_when_given_a_resolution():
    ...

def test_guard_row_carries_the_detected_split_into_the_guarded_row():
    ...

def test_guard_row_pluralises_the_tag():
    ...

def test_render_report_shows_the_guard_block():
    ...

def test_markdown_table_shape():
    ...
