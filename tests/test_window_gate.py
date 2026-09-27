"""Tests for the corpus window trust gate and its view-conditional trunk option.

Contract under test (S1 lane ``oblique_redetect``):
  * with the flag OFF the gate is bit-for-bit the shipped six-clause gate;
  * the trunk branch can only fire on ``view_class == "oblique"``;
  * it can only ADD trust -- it is a DISJUNCTION with the original whole-animal
    clause, and it relaxes no other clause;
  * the trunk bracket itself is a conjunction over three stations with a
    confidence floor, a border inset and an 80 %-of-frames floor.
"""
import numpy as np
import pytest
from shark_pose.corpus.window_gate import GATE_THRESHOLDS, MEASUREMENT_RANKING, TRUNK_GATE, WindowCandidate, WindowStats, admit_then_rank, admit_window, expected_measured_frames, gate_failures, trunk_frame_ok, trunk_bracket_fraction
from shark_pose.corpus.window_gate import FrameStats, best_span, subwindow_scan, window_relative_clip_frames

def stats(**kw):
    ...

def test_clean_window_passes():
    ...

@pytest.mark.parametrize('kw,tag', [(dict(n_detected=3), 'too_few_detections'), (dict(fit_rms_over_L=0.16), 'incoherent_landmarks'), (dict(fit_rms_over_L=None), 'incoherent_landmarks'), (dict(L_px_native_median=249.0), 'too_small_in_frame'), (dict(frac_both_ends=0.49), 'animal_not_wholly_in_frame'), (dict(frac_multi_det=0.35), 'competing_detection'), (dict(mean_visible_kp=9.9), 'too_few_keypoints')])
def test_each_clause_fires(kw, tag):
    ...

def test_flag_off_is_identical_for_oblique():
    """A truncated oblique window still fails with the flag off."""
    ...

def test_trunk_gate_admits_truncated_oblique():
    ...

def test_trunk_gate_is_a_disjunction_not_a_replacement():
    """Pre-registration 3: a window already carried by frac_both_ends keeps its
    trust even when its trunk bracket is incomplete."""
    ...

def test_bracket_threshold_is_eighty_percent():
    """The one revision permitted by pre-registration 6, taken through
    falsifier 5: 0.50 left the admitted set separable at 0.734, 0.80 at 0.603."""
    ...

def test_residual_stays_whole_body_by_default():
    """Measured: the trunk-only residual is worse (0.199 vs 0.163 median over
    258 rejected oblique windows), so it must not be substituted by default."""
    ...

@pytest.mark.parametrize('view', ['broadside', 'intermediate'])
def test_trunk_gate_never_touches_other_views(view):
    """Pre-registration 2: no broadside/intermediate row may move."""
    ...

def test_trunk_gate_requires_the_bracket():
    ...

@pytest.mark.parametrize('kw,tag', [(dict(n_detected=3), 'too_few_detections'), (dict(fit_rms_over_L=0.16), 'incoherent_landmarks'), (dict(L_px_native_median=249.0), 'too_small_in_frame'), (dict(frac_multi_det=0.35), 'competing_detection'), (dict(mean_visible_kp=9.9), 'too_few_keypoints')])
def test_other_clauses_survive_the_trunk_branch(kw, tag):
    ...

def test_trunk_residual_option_rescues_a_blown_up_whole_body_fit():
    """Opt-in only: with trunk_residual=True a whole-body residual blown up by a
    missing snout is read from the trunk-only fit instead."""
    ...

def test_trunk_residual_option_never_fires_on_a_full_window():
    """A window carried by frac_both_ends is FULL, so its whole-body residual
    still governs even with trunk_residual=True."""
    ...

def test_already_trusted_oblique_stays_trusted():
    """Pre-registration 3: the change may only add."""
    ...
W, H = (640.0, 360.0)

def kp_all_inside():
    ...

def test_bracket_true_when_all_stations_present():
    ...

@pytest.mark.parametrize('station', [(4, 6), (8, 9), (11,)])
def test_bracket_false_when_a_station_is_missing(station):
    ...

def test_paired_station_survives_one_member():
    ...

def test_bracket_false_when_a_station_sits_in_the_border():
    ...

def test_bracket_border_is_two_percent_of_the_diagonal():
    ...

def test_bracket_fraction_over_frames():
    ...

def test_thresholds_are_the_scan_thresholds():
    ...
from shark_pose.corpus.window_gate import SUBWINDOW, FrameStats, aggregate_frames, score_window, subwindow_pass, subwindow_scan

def good_frame(**kw):
    ...

def bad_frame(**kw):
    """Far/frontal approach: small, incoherent, ends out of frame."""
    ...

def test_subwindow_defaults_are_the_preregistered_ones():
    ...

def test_aggregate_frames_reproduces_the_window_medians():
    ...

def test_flag_off_is_the_whole_window_gate():
    """Default OFF must be the shipped gate, pass and fail alike."""
    ...

def test_subwindow_recovers_a_window_whose_medians_fail():
    """30 unusable frames + a sustained 20-frame lateral pass."""
    ...

def test_subwindow_is_sustained_not_lucky():
    """A single 5-frame flash passes ANY-BLOCK but not the sustained rule."""
    ...

def test_sustain_fraction_is_the_decision_boundary():
    ...

def test_min_runs_blocks_a_short_window():
    ...

def test_subwindow_can_only_add_trust():
    """A window the whole-window gate passes is never taken away by the flag."""
    ...

def test_subwindow_does_not_relax_any_clause():
    """Every block is individually too small: no amount of contiguity helps."""
    ...

def test_subwindow_composes_with_the_trunk_gate_and_stays_oblique_only():
    """Trunk-bracketed but snout/caudal out of frame, sustained over 20 frames."""
    ...

def test_best_stats_is_a_passing_block_when_one_exists():
    ...

def test_empty_window_is_rejected():
    ...

def cand(window_id='w', **kw):
    ...

def test_admit_then_rank_is_default_off_and_off_is_the_identity():
    ...

def test_admission_is_the_conjunction_of_the_two_measured_thresholds():
    ...

def test_expected_measured_frames_is_the_product_of_the_measured_rates():
    ...

def test_resolution_enters_linearly_and_is_capped():
    ...

def test_the_view_bonus_is_what_makes_the_pair_observe_two_axes():
    """A second oblique window is worth 0.6; the first is worth full score."""
    ...

def test_a_rejected_window_is_not_ranked_and_is_reported_as_rejected():
    ...

def test_an_admitted_window_outranks_a_trunk_admissible_one_at_a_higher_score():
    """TRUNK_ONLY may never quote a length, so it never leads the queue."""
    ...

def test_the_ranking_is_deterministic_under_input_permutation():
    ...

def fs(ok=True):
    """A scored frame that passes (or fails) the six clauses on its own."""
    ...

def test_best_span_is_source_frame_numbers_not_scored_indices():
    """C1: best_index indexes the SCORED list; a fitter needs source frames."""
    ...

def test_best_span_picks_the_longest_passing_run_and_expands_it():
    ...

def test_best_span_falls_back_to_the_whole_window_when_scored_frames_are_few():
    """Every window in this corpus: n_scored 3-6 against run_len 5."""
    ...

def test_subwindow_scan_emits_the_span_only_when_it_is_given_frame_numbers():
    ...

def test_window_relative_clip_frames_is_what_fit_individual_parses():
    """C2: source frames -> window-relative, half-open, clipped."""
    ...

def test_the_clip_frames_flag_is_refused_when_it_would_gain_nothing():
    ...
