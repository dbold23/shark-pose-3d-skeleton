"""Tests for shark_pose.sim2real.egomotion (plan items P2 / P3).

Four properties the report demands of this estimator, plus serialisation:

1. **one sign convention** -- ``R_cw`` is camera-from-world, world = the camera frame of
   the first valid frame; the Kabsch fit returns the SCENE rotation
   ``dR = R_cw(k+1) R_cw(k)^T`` and the reported rate is camera-in-world
   ``-rotvec(dR)``.  Checked on a synthetic rotated point cloud where the truth is known.
2. **injection recovery** -- a known rotation pushed through the identical chain must come
   back with gain 0.99-1.01.
3. **monotonicity in information** -- a zero-information series must NOT pass the gate.
   A featureless clip (no trackable background at all) is refused at stage 1, and a
   white-noise rate series buried under its own injection floor is refused at stage 2,
   even though B5 alone would happily pass it.
4. **serialisation round-trip** -- the record survives JSON with every number intact.
"""
import json
import math
import numpy as np
import pytest
from shark_pose.sim2real import egomotion as ego

def _synthetic_pair(R_cw0, R_cw1, K, n=400, seed=0):
    """Project a static world cloud into two camera poses."""
    ...

def test_sign_convention_on_synthetic_point_cloud():
    ...

@pytest.mark.parametrize('rate_deg_s', list(ego.SELF_TEST_RATES_DEG_S))
def test_injection_recovery_gain_on_synthetic_tracks(rate_deg_s):
    ...

def _featureless_clip(n=8, h=96, w=128):
    """Uniform frames: LK has no gradient anywhere, so there is nothing to track."""
    ...

def test_featureless_clip_is_refused_at_stage_1():
    ...

def test_white_noise_track_is_refused_at_stage_2():
    """A rate series carrying no information about the camera must not pass the gate.

    The scene here looks perfectly rigid and the estimate is perfectly repeatable -- so
    stage 1 passes -- but every axis is buried under the injection floor, so stage 2
    refuses.  This is the ordering the report demands: B5 is anti-monotone in estimator
    quality (white noise passes it), so it may only ever be reached after B1 and B2.
    """
    ...

def test_gate_b5_three_outcomes():
    ...

def test_b4_is_a_diagnostic_and_never_a_gate_condition():
    ...

def test_egotrack_json_round_trip(tmp_path):
    ...

def test_defaults_and_presets_are_the_measured_operating_point():
    ...

def test_named_floor_never_falls_below_the_per_pair_repeatability():
    """B3's numerator is a PER-PAIR RMS, so its floor may not be a probe-median limit.

    The demo's shape: an injection min-detectable well above its repeatability floor is
    used as named.  clip2's shape: a min-detectable that only exists in the median over
    probes, below the per-pair repeatability, is discarded for that floor.
    """
    ...

def test_b4_windows_and_trim_match_the_reference_definition():
    ...
