"""Tests for the S8 R3 dorsal-fin trailing-edge matcher on synthetic known-truth fins.

Identity lives only in trailing-edge notches (shark_pose.archive.synth_fins); a
sighting is a rendered body+fin mask under a random view with noisy keypoints.
Thresholds were set from the seeds below with margin; they guard against
regressions, they are not accuracy claims for real fins.
"""
from __future__ import annotations
import numpy as np
import pytest
from shark_pose.archive.fin_edge import encounter_similarity, fin_contour, fin_mask_from_body, match_score, profile_from_mask, trailing_edge
from shark_pose.archive.synth_fins import random_identity, render

def _rank1(n_notches, *, n=40, frames=3, base=160, seed=0):
    ...

def test_trailing_edge_never_contains_anterior_base():
    ...

def test_mirror_view_gives_the_same_profile():
    ...

def test_notched_fins_are_identified_well_above_chance():
    ...

def test_unmarked_fins_carry_no_identity():
    """Smooth fins differ only in sweep, which the detail removes: rank 1 must stay near chance."""
    ...

def test_more_frames_per_encounter_do_not_hurt():
    ...

def test_unmarked_fin_is_less_distinct_than_a_notched_one():
    ...

def test_vectorised_matrices_equal_the_loop():
    ...
