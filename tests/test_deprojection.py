"""S5 lane `broadside_clause`: the de-projected companions of the
broadside-only projected rows.  Pure numpy -- no assets, no fit record.
"""
from types import SimpleNamespace
import numpy as np
import pytest
from shark_pose.morphometrics.aggregate import bootstrap_median_ci, median_mad
from shark_pose.morphometrics.deprojection import DEPROJECTED_QUANTITIES, MIN_OOP_COS, build_deprojected_rows, deprojected_series
from shark_pose.morphometrics.schema import QUANTITIES

def _per_frame(yaw_deg, tl_model=4.0, dh_model=0.6, px_per_unit=100.0):
    """A clip whose body yaws through ``yaw_deg`` while nothing about the
    animal changes.  The total-length chord lies along the body, so it
    foreshortens by cos(yaw); the dorsal fin stands perpendicular to it and
    does not foreshorten at all.
    """
    ...

def test_deprojection_makes_the_projected_length_yaw_invariant():
    """The whole point: the projected row varies by 3.6x across the yaw range
    the corpus actually contains, the de-projected row by nothing."""
    ...

def test_a_chord_perpendicular_to_the_body_is_not_foreshortened_by_yaw():
    """DH is why gating the MODEL-space rows on the body axis is the wrong
    test: the dorsal chord's own out-of-plane cosine is 1 at every yaw, so its
    projected row is already yaw-invariant and de-projection is a no-op."""
    ...

def test_frames_below_the_cosine_floor_are_dropped_not_amplified():
    ...

def test_rows_are_taken_over_all_gated_frames_and_count_only_finite_ones():
    ...

def test_the_companions_are_not_in_the_shipped_vocabulary():
    """Flag default OFF means the shipped record is untouched: none of these
    names may appear in QUANTITIES, which is what aggregate_clip iterates."""
    ...
