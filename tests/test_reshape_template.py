"""Tests for scripts/reshape_template.py -- the v2 rest-template reshape.

The reshape is the one edit in this repo that changes the MESH the whole v2
path is built on, so what these tests pin is: it is exactly a no-op when its
parameters say "do nothing", it moves what it claims to move by the amount it
claims, and it moves nothing else -- in particular no fin tip that the
silhouette is made of.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

def test_peduncle_factor_reproduces_the_measured_knot_table():
    """The shipped profile IS D's table, knot for knot (report.md sec. 4).

    The sixth knot is not D's: it returns the profile to 1.0 at y = -1.90 so
    that the thinning stops inside the caudal fin's root instead of running to
    the end of the mesh.  D measured the BODY's depth at the peduncle; the fin
    is outside what that number constrains.
    """
    ...

def test_peduncle_factor_is_flat_outside_the_knots():
    ...

def test_peduncle_factor_is_bounded_and_falls_then_returns():
    """0.80 is the floor; the profile leaves 1.0 once and comes back once."""
    ...

def test_peduncle_factor_is_c1_at_every_knot():
    """THE CREASE FIX.  A slope jump in a(y) is a normal jump in the surface.

    ``np.interp`` (the profile that shipped on 2026-09-06) is C0: its second
    difference is a delta at each knot and diverges as the sample step shrinks.
    The smoothstep profile's slope VANISHES at every knot, so the second
    difference stays bounded.  Measured on a 1e-5 grid: max |d2a/dy2| 78.0 for
    the smoothstep profile against 1.3e5 for np.interp, and the latter grows
    like 1/h.
    """
    ...

def test_smooth_on_mesh_holds_a_constant_and_respects_the_pin():
    """The smoother must not invent a gradient, and pinned rows never move."""
    ...

@needs_assets
def test_smooth_on_mesh_keeps_the_fin_blades_at_full_fade():
    """Pinning at 1 is what keeps every fin TIP moving exactly 0."""
    ...

def test_rostrum_map_is_identity_behind_the_ramp():
    ...

def test_rostrum_map_moves_the_snout_by_exactly_the_requested_extension():
    ...

def test_rostrum_map_is_monotone():
    ...

def test_rostrum_jacobian_matches_a_finite_difference():
    ...

def test_rostrum_stretch_factor_rejects_an_anchor_ahead_of_the_snout():
    ...

def test_check_source_template_refuses_a_different_mesh():
    ...

@needs_assets
def test_check_source_template_accepts_the_v1_template():
    ...

@needs_assets
def test_identity_params_leave_the_mesh_bit_identical():
    """dy = 0 and every factor 1 must return the input BIT for BIT."""
    ...

@needs_assets
def test_apply_rostrum_and_apply_peduncle_are_bitwise_no_ops_at_identity():
    ...

@pytest.fixture(scope='module')
def source():
    ...

def test_rostrum_moves_the_snout_tip_by_exactly_dy(source):
    ...

def test_nothing_behind_the_ramp_moves_longitudinally(source):
    ...

def test_the_mid_body_does_not_move_at_all(source):
    """Between the ramp and the peduncle the reshape is exactly the identity."""
    ...

def test_the_reshape_never_moves_x(source):
    ...

def test_the_peduncle_thinning_never_moves_y(source):
    """Stage 2 is a cross-section scale: it changes girth, not length."""
    ...

def test_peduncle_thins_the_section_depth_by_the_stated_factor(source):
    """At the measured peduncle the BODY's dorsoventral depth x 0.80."""
    ...

def test_peduncle_leaves_the_trunk_section_untouched(source):
    """The trunk's own depth is unchanged to within a twentieth of a percent.

    Not bit-identical, and that is a property of the SECTION, not of the mesh:
    every trunk VERTEX is bit-identical (see
    ``test_the_mid_body_does_not_move_at_all``), but a long face reaching back
    from the trunk into the thinned peduncle has one endpoint that moves, so
    the plane's intersection with it moves a little.  Measured worst case over
    these four planes: 0.35% at y = -0.80, 0.05% at y = -0.50, exactly 0 ahead
    of the pelvic fin.
    """
    ...

def test_the_caudal_fin_tips_do_not_move(source):
    """The blades that make the silhouette sit outside the gate: exactly 0."""
    ...

def test_the_gate_is_the_shipped_one(source):
    """build_gate must return the operator SharkSMPL builds, unchanged."""
    ...

def test_extra_points_do_not_perturb_the_vertices_gate(source):
    """Appending extra points must not move the body-radius estimate."""
    ...

def test_build_gate_rejects_a_mismatched_fin_weight(source):
    ...

def test_section_extents_measures_a_known_box():
    """A 2 x 4 x 1 box: the section at any interior y is 2 wide, 1 deep."""
    ...

def test_profile_uses_the_planes_it_is_given():
    ...

@needs_assets
def test_build_thin_gate_is_one_field_and_pins_the_blades(source):
    """The shipped stage-2 field has a single implementation.

    ``scripts/build_spine_rig.py`` builds stage 2 through
    :func:`reshape_template.build_thin_gate`, and so does :class:`Reshape` when
    it is handed faces and a blade fade.  What the function owes: the fade's
    pinned rows take EXACTLY zero thinning, and the field is not the raw radial
    gate (smoothing it is the crease fix).
    """
    ...

@needs_assets
def test_reshape_thins_by_the_shipped_field_only_when_it_is_given_one(source):
    """Reshape without faces uses the RAW gate -- a different mesh, on purpose.

    That is the divergence the CLI cannot paper over: no fin segmentation, no
    blade fade, no smoothing, so no shipped template.
    """
    ...

@needs_assets
def test_reshape_refuses_half_of_the_shipped_field(source):
    ...

def test_the_cli_cannot_be_asked_to_write_a_template(tmp_path):
    """``--out`` is gone: this path cannot reproduce the shipped mesh."""
    ...

@needs_assets
def test_the_cli_writes_nothing_and_says_so(capsys):
    """It prints the profile of a PRE-RESHAPE asset and leaves no files."""
    ...

@needs_assets
def test_the_cli_refuses_an_asset_that_is_already_reshaped():
    ...

def test_smooth_on_mesh_tolerates_an_isolated_last_vertex():
    """A degree-0 LAST vertex must not crash the neighbour reduction.

    np.add.reduceat raises when a segment start equals len(idx); the shipped
    template has 20 isolated vertices but none is the highest index, so the
    guard is pinned by this two-triangle mesh whose last vertex is unreferenced.
    """
    ...
