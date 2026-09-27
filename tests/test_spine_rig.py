"""Tests for the v2 midline spine rig (data/model_assets_v2)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.model_3d import spine_skeleton as ss
from shark_pose.model_3d.skinning import lbs

@pytest.fixture(scope='module')
def rig():
    ...

@pytest.fixture(scope='module')
def manifest():
    ...

def _pose(rig, rotations: dict[int, np.ndarray]):
    """LBS the template with per-joint local rotations (identity elsewhere)."""
    ...

def _rot_z(deg: float) -> np.ndarray:
    ...

def _rot_axis(axis: np.ndarray, deg: float) -> np.ndarray:
    ...

def test_joint_set_shape():
    ...

def test_spine_names_match_sevengill_semantics():
    ...

def test_parents_are_a_single_rooted_tree():
    ...

def test_spine_chain_is_walkable_both_ways_from_the_root():
    ...

def test_fin_tips_hang_off_fin_roots_which_hang_off_the_spine():
    ...

def test_bone_pairs_count():
    ...

def test_build_parents_rejects_a_fin_root():
    ...

def test_asset_shapes(rig):
    ...

def test_v2_geometry_is_v1_reshaped_not_copied():
    """v2's template is v1's RESHAPED (scripts/reshape_template.py), not a copy.

    The v2 mesh has been given the measured rostrum extension and peduncle
    thinning, so it is deliberately NOT byte-identical to v1 any more.  What is
    still pinned is the topology and the parts the reshape does not touch: the
    faces, the vertex count, and every vertex between the stretch ramp and the
    peduncle knots -- the whole trunk -- which must be bit-identical.
    """
    ...

def _reshape_bounds():
    """(ramp start, aft-most peduncle knot) of the shipped reshape parameters."""
    ...

def test_v2_landmark_asset_keeps_v1s_keys_and_moves_the_corrected_rows():
    ...

def test_the_shipped_template_carries_the_measured_reshape(rig):
    """The defect the shipped reshape is for is gone, by the amount D targeted.

    PEDUNCLE.  The narrowest body section, sqrt(lateral x dorsoventral), thins
    by sqrt(1.0 x 0.80) = 0.894 -- lateral unchanged, depth x 0.80.  Measured
    on the asset: 0.25037 before, and the value below after.  It is a little
    above the ideal 0.22394 because the thinning gate is smoothed over the mesh
    (the crease fix), which softens it where the anal and second-dorsal blades
    come close to the ventral wall.
    """
    ...

def test_the_rostrum_half_is_off_and_that_is_the_measured_decision(rig):
    """extension_frac_y_extent is 0.0, so no landmark and no y moves.

    D's diagnosis of a short rostrum stands (report.md sec. 4), but the fix was
    withdrawn on 2026-09-06 by TASK F on its own numbers: at half the extension
    that shipped, clip 2's from-scratch residual went 30.5354 -> 31.6826 px and
    its hard IoU 0.7108 -> 0.7011, and the val89 harness's v2 row median went
    8.6017 -> 8.7816 px, all outside the decision rule's bounds (clip 2 +1.0 px
    / -0.005 IoU, harness +0.1 px on mean AND median).  The demo's 41.1028 px /
    0.6962 IoU was the only row it bought.  The parameter is kept, with its
    provenance, so that turning it back on is one edit and a rebuild.
    """
    ...

def test_the_reshape_moves_no_landmark_but_the_corrected_rows(rig):
    """Every head landmark stays put.

    D measured on BOTH clips that eye_center, gill_slit and first_dorsal_base
    land within 0.8% of the body span of their detections, so the reshape must
    not slide them.  With the rostrum half off (extension 0.0) that holds for
    snout_tip too: the only rows that move are the ones calibrate_landmarks.py
    stamps as corrected, plus caudal_notch, which rides the thinned wall.
    """
    ...

def _rebuild(tmp_path, *extra):
    """Run the shipped builder into a scratch directory and return it."""
    ...

def test_the_shipped_parameters_regenerate_the_shipped_asset(tmp_path):
    """The asset in the tree IS what RESHAPE_PARAMS + this script produce.

    Every array, bit for bit -- otherwise the checked-in parameters are not the
    provenance of the checked-in mesh and no measurement made on it is
    reproducible.
    """
    ...

def test_no_reshape_reproduces_the_v1_geometry_bit_for_bit(tmp_path):
    """--no-reshape is a provable no-op: IDENTITY_PARAMS write nothing.

    The rest of the build (rig, skinning, landmark rebuild) still runs, so this
    also pins that the reshape is the ONLY thing between v1's mesh and v2's.
    """
    ...

def _edge_dihedrals(verts, faces):
    """Angle between the two incident face normals, for every manifold edge."""
    ...

def test_the_thinning_does_not_crease_the_surface():
    """THE CREASE TEST.  No edge may gain 30 deg of dihedral to the reshape.

    A step in the displacement field between neighbouring vertices is a step in
    the surface normal, i.e. a visible crease.  The template that shipped on
    2026-09-06 had 19 edges crossing from below 60 deg to above it, the worst
    (1483-1878) going 26.19 -> 88.08 deg, because a(y) was piecewise linear,
    clamped aft of its last knot into the caudal fin, and multiplied by a gate
    that goes 1 -> 0 across a single edge there.  With the C1 profile, the
    -1.90 knot and the mesh-smoothed gate the count is 7 and the worst gain is
    24.84 deg (1608-1610, 6.02 -> 30.85).
    """
    ...

def test_every_fin_tip_moves_exactly_zero(manifest):
    """The blade fade is pinned at 1 over each blade's outer half."""
    ...

def test_manifest_stamps_the_measured_arc_span(manifest):
    """The extractor and the fitter measure the span off the asset, so the
    asset says what it measures."""
    ...

def test_weight_rows_sum_to_one(rig):
    ...

def test_every_joint_drives_some_vertex(rig):
    ...

def test_spine_stations_lie_on_the_midline_and_run_snout_to_tail(rig):
    ...

def test_spine_chain_is_nearly_straight(rig):
    ...

def test_fin_tips_come_from_the_landmark_asset(rig):
    ...

def test_fin_roots_sit_between_the_midline_and_the_tip(rig):
    """A root must be proximal: nearer the spine than its own tip is."""
    ...

def test_rest_pose_reproduces_the_template(rig):
    ...

def test_bone_lengths_are_preserved_under_forward_kinematics(rig):
    ...

def test_tail_bend_leaves_the_head_untouched(rig):
    """Rotating spine_08..12 must not move a single head vertex, and must not
    tear the flank it does move."""
    ...

def test_tail_bend_does_not_tear_the_mesh(rig):
    ...

def test_cranium_rotation_moves_only_the_head(rig):
    ...

def test_head_stays_rigid_when_the_cranium_rotates(rig):
    """Vertices the cranium alone drives must move as one rigid body."""
    ...

def test_fin_root_rotation_stays_local_to_that_fin(rig):
    ...

def test_manifest_records_passing_checks(manifest):
    ...

def test_manifest_cross_checks_against_the_digitallife_armature(manifest):
    ...

def test_skeleton_json_matches_the_module(rig):
    ...

def test_v1_rig_is_untouched():
    """The 16-joint rig must keep working beside the new one."""
    ...

@pytest.fixture(scope='module')
def prior():
    ...

def _default_params(prior, **overrides):
    ...

def _posed_joints(rig, prior, params, t=0.0, clamp=True):
    ...

def _tail_shift(rig, prior, params):
    """Displacement of the last vertebral station away from the rest pose."""
    ...

@needs_prior
def test_prior_pitch_is_dorsoventral_not_a_roll(rig, prior):
    """H1: the prior's angles are decomposed in the joint-local bone frame, so
    they have to be rotated into the model frame before they are skinned.
    Without that, ``pitch`` turns about the body's long axis and the tail
    barely moves at all."""
    ...

@needs_prior
def test_prior_turn_is_lateral(rig, prior):
    ...

@needs_prior
def test_prior_roll_hardly_moves_the_tail(rig, prior):
    """A roll spins the body about its own long axis, so the midline stations
    stay where they are."""
    ...

def _cruise_stations(rig, prior, n=120):
    """Vertebral stations over one cruise period at amplitude 1.0."""
    ...

def _chord_bend_deg(points_2d):
    """Signed angle between the front-half and back-half chords of a midline."""
    ...

@needs_prior
def test_cruise_at_full_amplitude_keeps_the_head_rigid(rig, prior):
    """M1: the anterior stations are parented backwards, so the sevengill's
    curvature has to change sign there.  Either way the head itself must not
    deform: its three joints stay inside their 3 deg limits and the cranium
    barely moves relative to the branchial bone."""
    ...

def _angle_between(a, b) -> float:
    ...

@needs_prior
def test_cruise_chord_bend_matches_the_sevengill_calibration(rig, prior):
    """Head-chord vs tail-chord angle, the metric the SAM centrelines are scored
    with: the sevengill's cruise at amplitude 1.0 sits at ~16 deg median."""
    ...

@needs_prior
@pytest.mark.skipif(not SEVENGILL.exists(), reason='sevengill rig asset missing')
def test_cruise_midline_matches_the_sevengill_shape(rig, prior):
    """The posed midline is the sevengill's own cruise midline (up to the rigid
    motion the free global orientation absorbs).  Running the anterior wave with
    the sevengill's raw sign folds the front of the body the wrong way and blows
    this up."""
    ...

def _rigid_residual(a: np.ndarray, b: np.ndarray) -> float:
    """Largest point error after the best rigid (rotation + shift) fit of a onto b."""
    ...
