"""The girth field: a per-station cross-section scale on the v2 SharkSMPL mesh.

The v2 PCA shape basis has no girth mode, so the fit cannot make the animal
fatter or thinner where the silhouette says it should.  ``forward(girth=...)``
adds an explicit one: K=7 control stations along the body carry (g_x, g_z),
Catmull-Rom interpolated to every vertex by its own along-body fraction and
applied to the REST mesh, before LBS, as a scale of the vertex's lateral and
dorsoventral offsets from the rest spine axis.

What these tests pin down, all measured on ``data/model_assets_v2``:

* ``girth=None`` and ``girth=1`` are BIT-IDENTICAL to the model built without a
  girth field at all -- every existing caller is unaffected.
* a uniform g_x = 1.2 widens the body wall by exactly 1.2 at every station and
  moves the pectoral, first-dorsal and caudal tips by exactly 0.
* the small fins that lie INSIDE the body's radial envelope (pelvic 13.4%,
  second dorsal 4.1%, anal 0.4% of their lateral offset, under an isotropic
  g = 1.2 -- the two median fins' numbers are a midline artefact, see the test)
  ride out with the body wall they attach to.  A purely radial gate
  cannot separate them from the trunk, and letting them ride is the better of
  the two failure modes: they translate with the wall instead of detaching
  from it.  The fins the silhouette is actually made of -- pectoral, first
  dorsal, caudal -- are exactly untouched.
* girth changes girth, not length: the y coordinate of every vertex is
  bit-identical, and so is every joint.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.model_3d import GIRTH_GATE_END, GIRTH_GATE_START, GIRTH_NUM_STATIONS, GIRTH_STATION_FRACTIONS, SharkSMPL, catmull_rom_basis, girth_from_width_ratio, girth_prior_loss

@pytest.fixture(scope='module')
def model() -> SharkSMPL:
    ...

@pytest.fixture(scope='module')
def rest_params(model: SharkSMPL):
    ...

def _tip_vertex(model: SharkSMPL, joint_name: str) -> int:
    """Template vertex nearest a named rig joint."""
    ...

def test_v2_assets_provide_a_girth_field(model):
    ...

def test_identity_girth_is_bit_identical(model, rest_params):
    ...

def test_no_girth_path_matches_a_model_without_the_field(model, rest_params):
    """Every existing caller (girth=None) is bit-for-bit unaffected."""
    ...

def test_model_without_spine_chain_refuses_girth():
    ...

def _widen(model, pose, betas, gx=1.0, gz=1.0):
    ...

def test_uniform_gx_widens_the_trunk_by_20_percent(model, rest_params):
    ...

def test_protruding_fins_are_not_scaled(model, rest_params):
    """Pectoral / first dorsal / caudal tips: exactly 0 displacement."""
    ...

def test_fins_inside_the_body_envelope_ride_with_the_wall(model, rest_params):
    """Measured, not asserted away: the small fins the radial rule cannot see.

    Re-measured on the shipped template (2026-09-06, TASK F).  The gate is
    unchanged in kind at all three tips -- the second dorsal and the anal fin
    take the wall scale in full (gate exactly 1.0, u 1.176 and 0.893), the
    pelvic tip is outside it (gate 0.666) -- and the ratios are back at the
    values the PRE-reshape template gave, 0.089 and 0.043, because the shipped
    reshape is dorsoventral only: it moves no x and no y, so the spine stations
    the ratio is measured against are v1's.  (TASK E's 0.041 / 0.004 came from
    the rostrum extension re-parameterising those stations by ~0.0013 in
    ``axis_x``; that half no longer ships.)  For a fin on the midline this
    ratio is ill-conditioned anyway: the tip's lateral move is
    ``0.2 * (x - axis_x)`` and both terms are ~0.006, so it measures the gap
    between the blade and the model's own midline, not how the blade rides.
    ``pelvic_L_tip``, the one tip with a real lateral offset (|x| = 0.234), is
    the stable row: 0.130.
    """
    ...

def test_gate_is_smooth_and_bounded(model):
    ...

def test_girth_changes_girth_not_length_or_joints(model, rest_params):
    ...

def test_flank_landmarks_follow_the_surface(model, rest_params):
    ...

def test_girth_composes_with_betas(model):
    ...

def test_single_station_bump_is_local_and_smooth(model, rest_params):
    """A bump at one station stays inside that station's Catmull-Rom support."""
    ...

def test_batched_and_unbatched_girth_agree(model):
    ...

def test_bad_girth_shape_raises(model):
    ...

def test_gradients_are_finite_and_nonzero(model):
    ...

def test_catmull_rom_basis_properties():
    ...

def test_girth_prior_loss():
    ...

def test_girth_prior_smoothness_is_measured_against_a_per_station_mean():
    """A ``(K, 2)`` mean carries its own profile into the smoothness term.

    The docstring promised "smooth like the prior says a shark is, not flat";
    the code differenced the RAW field, so the term fought any non-constant
    mean. A scalar or ``(2,)`` mean is constant along the body and must keep
    the untouched expression, bit for bit.
    """
    ...

def test_girth_from_width_ratio():
    ...

def test_girth_from_width_ratio_round_trips_through_the_model(model, rest_params):
    """Measured widths -> initial g -> the model actually widens by that much."""
    ...
