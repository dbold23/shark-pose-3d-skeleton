"""Tests for the corpus view (yaw) cue and its depth-free replacement.

Contract under test (S1 lane ``yaw_defect``):
  * the legacy map is a deterministic affine function of ``dvL`` and reproduces
    the shipped ``obl_deg_est`` / ``view_class`` exactly;
  * with the flag OFF (the default) nothing in this module consults the fin
    spans -- ``window_cue`` publishes the legacy cue and ``classify`` is
    ``classify_dvL`` bit-for-bit;
  * the legacy cue MOVES when only the body depth moves and the yaw does not;
    the fin-referenced cue does NOT;
  * the fin-referenced cue tracks a synthetic pure yaw by the cosine law;
  * the replacement refuses to classify without thresholds on its own scale.
"""
import numpy as np
import pytest
from shark_pose.corpus.view_yaw import CLASS_THRESHOLDS_DVL, DEFAULT_CONFIG, IDX, KAPPA_CS_OVER_DH, LEGACY_ANCHORS, LEGACY_SLOPE, PROPORTIONS, SKELETON_ORDER, ViewYawConfig, classify, classify_dvL, frame_cue, legacy_obl_deg, rank_preserving_thresholds, window_cue

def test_legacy_map_reproduces_the_two_anchors():
    ...

def test_legacy_map_is_strictly_increasing_in_dvL():
    """Which is why Spearman(obl_deg_est, dvL_median) is 1.0 up to rounding."""
    ...

@pytest.mark.parametrize('dvL,expected', [(0.2, 'broadside'), (0.534, 'broadside'), (0.535, 'intermediate'), (0.685, 'intermediate'), (0.686, 'oblique'), (0.9, 'oblique')])
def test_shipped_class_thresholds(dvL, expected):
    ...

def _straight_shark(depth_over_L=0.18, dorsal_h=0.09, caudal_span=0.26, L=400.0, yaw_deg=0.0):
    """A rigid broadside shark yawed by ``yaw_deg`` about the dorsoventral axis.

    The yaw foreshortens the along-axis coordinate by cos(yaw) and leaves every
    transverse (dorsoventral) coordinate alone -- the exact optics both cues
    claim to read.
    """
    ...

def test_frame_cue_recovers_the_planted_geometry():
    ...

def test_the_defect_legacy_cue_moves_on_fatness_alone():
    """Same yaw (zero), fatter animal: the legacy cue moves, the new one does not."""
    ...

def test_fin_cue_follows_the_cosine_law_under_pure_yaw():
    """cue = L cos(yaw) / T, so arccos(cue / cue_broadside) returns the yaw.

    The recovery is exact up to the caudal pair's own 0.0468 TL along-axis
    offset, which the raw chord carries; that leaves at most 1.5 deg of bias
    out to a 60 deg yaw.
    """
    ...

def test_dorsal_fin_fallback_matches_the_caudal_span_scale():
    """An animal whose two fin chords sit at exactly KAPPA reads the same cue
    whether or not its caudal tips were detected."""
    ...

def test_default_config_is_the_legacy_estimator():
    ...

def test_window_cue_publishes_the_legacy_cue_by_default():
    ...

def test_classify_off_is_classify_dvL_bit_for_bit():
    ...

def test_classify_on_refuses_without_thresholds_on_its_own_scale():
    ...

def test_classify_on_inverts_the_comparisons():
    ...

def test_unknown_estimator_is_refused():
    ...

def test_rank_preserving_thresholds_reproduce_the_class_counts():
    ...

def test_skeleton_order_is_the_16_joint_sequence():
    ...

def test_frame_cue_refuses_a_wrong_shaped_array():
    ...

def test_class_thresholds_are_the_shipped_numbers():
    ...
