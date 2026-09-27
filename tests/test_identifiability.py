"""Identifiability: what the reprojection residual can and cannot see (P8).

Plan item **P8** of ``outputs/demo_2026-09-03/work/egomotion/report.md``
section 4:

    P8  Identifiability unit tests -- **claim_1 and claim_2 only** -- plus the
        standing prohibitions in the module docstring.  claim_3 withdrawn.

and its acceptance line in ``design/recommendation.md``:

    claim_1 max reprojection difference < 1e-9 px over 200 random camera
    rotations, **built against the pipeline's own PerspectiveCamera.project and
    root composition** so the test can actually fail; claim_2 sideslip
    invariance < 1e-9 deg.

**claim_1** (``design/camera_leaves_identifiability.json``, shipped
1.1369e-13 px; re-derived 1.36e-12 px in ``synth2/revision_measurements.json >
C9_identifiability``): rotate the camera by ``R`` and the animal by ``R^-1``
and every projected pixel is unchanged.  The camera/animal split is a **gauge
freedom of the data term**, so no residual-based test can ever confirm that an
ego compensation is correct -- which is why P4 ships default-off and its loss is
validated numerically instead (``tests/test_record_contract.py``).

**claim_2**: sideslip is invariant to camera *attitude*; the camera's *rate*
enters only through ``omega_c x x_c``.  Consequence, and the reason the
non-holonomic prior is not a camera-attitude observation: a swimming prior can
constrain the camera rate and can say nothing at all about where the camera is
pointing.

**claim_3 is deliberately absent.**  It is an algebraic identity of a 3x3 skew
matrix (2000-case control: max rank 2, min |cos| 1.0), so it cannot fail for any
input, and it is false for the 31-joint map it was quoted to justify (rank 3,
sv [28.84, 28.15, 6.64], cond 4.34).  Do not re-add it.

These tests run in **float64** end to end.  The model's forward pass is float32,
so the body points are read once and the rotations, composition and projection
-- the thing under test -- are done in double; the float32 corroboration is
asserted separately at its own, much looser, precision.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.core.camera import PerspectiveCamera
from shark_pose.core.transforms import axis_angle_to_rotation_matrix
from shark_pose.encoder.smplify_fitting import compose_axis_angle
from shark_pose.model_3d.shark_smpl import SharkSMPL
from shark_pose.sim2real import egomotion as em
from shark_pose.sim2real import world_frame as wf
N_CASES = 200
CLAIM1_PX = 1e-09
CLAIM2_DEG = 1e-09

def _random_rotations(n: int, seed: int=20260907, max_deg: float=30.0) -> np.ndarray:
    """(n, 3) axis-angle vectors: uniform axis, angle uniform up to ``max_deg``.

    The degeneracy itself holds for **any** rotation, but the comparison has to
    be made in pixels and ``PerspectiveCamera.project`` clamps ``z`` at 1e-8:
    a rotation that swings the animal behind the camera turns both arms into
    ~1e10 px coordinates, where an absolute pixel difference means nothing.
    30 deg keeps every joint well in front of the camera (asserted in the test)
    and is far beyond any camera motion this footage contains -- the demo
    integrates 0.616 deg over its whole window.
    """
    ...

def _camera() -> PerspectiveCamera:
    """The pipeline's own pinhole, at the demo fit's proxy focal."""
    ...

def _project(cam: PerspectiveCamera, pts: np.ndarray) -> np.ndarray:
    """PerspectiveCamera.project in float64, (N, 3) -> (N, 2)."""
    ...

@pytest.fixture(scope='module')
def body_points():
    """``(joints, pivot)``: the v2 model's joints with the root rotation removed.

    Read once, in float32 (the model's own dtype), then held in double: both
    arms of every test below use the SAME points, so the model's precision
    cancels exactly and what is measured is the rotation/composition/projection
    chain.  ``pivot`` is the root joint's position -- the point the chain
    rotates about, which the absorbed translation has to respect.
    """
    ...

def test_claim_1_a_joint_camera_animal_rotation_leaves_every_pixel_unchanged(body_points):
    """Rotate the camera by R, the animal by R^-1: the projection is identical.

    Arm A is the physical statement -- the camera turns by ``R`` about its own
    centre and the world-static animal's camera-frame points become ``R X_cam``.
    Arm B is what the FIT would have to do to explain the same image without a
    camera: absorb ``R`` into ``root_orient`` (through the pipeline's own
    ``compose_axis_angle``) and into the translation.  The two agree to
    ``CLAIM1_PX``, so the data term cannot tell them apart, and a wrong
    composition order in arm B would show up here immediately.
    """
    ...

def test_claim_1_holds_through_the_records_own_world_camera_split(body_points):
    """The gauge freedom is exactly the split the record writes.

    ``world_frame.body_attitude`` turns ``(pose[root_joint], R_cw)`` into
    ``R_wb``.  Re-composing ``R_cw @ R_wb`` and projecting must return the same
    pixels for **every** ``R_cw`` -- that is the claim_1 degeneracy stated in
    the record's own variables, and it is what makes ``R_cw`` unconstrained by
    the fit.  A transposed ``body_attitude`` fails this test.
    """
    ...

def test_claim_1_corroborated_through_the_models_own_float32_forward(body_points):
    """The same statement end to end through SharkSMPL, at float32 precision.

    The double-precision tests above are the ones with the 1e-9 bar; this one
    exists so the claim is not merely a statement about numpy: it drives the
    real forward pass -- LBS, blend shapes and all -- whose float32 arithmetic
    puts the floor near 1e-3 px on a 256 px proxy.  Do not tighten the bound --
    tighten the dtype first.

    The absorbed translation is ``R (p + t) - p``, not ``R t``, because the
    model rotates the chain about the root joint ``p``.  With ``R t`` the error
    is 15.7 px, not 1e-3: this test is what pins that convention.
    """
    ...

def _trajectory(T: int=60, seed: int=7):
    """A smooth camera-frame trajectory and body long axis."""
    ...

def test_claim_2_sideslip_is_invariant_to_camera_attitude():
    """Turning the camera moves every vector; the ANGLE between them does not.

    This is the whole reason a swimming prior is not a camera-attitude
    observation: attitude must come from gravity or a horizon, neither of which
    exists on this footage (report.md F1).
    """
    ...

def test_claim_2_the_camera_rate_enters_only_as_omega_cross_x():
    """The ``omega x x`` correction is the discrete de-rotation, to first order.

    The exact statement is the de-rotated trajectory: with a camera turning at
    ``omega``, the animal's world path is ``R_cw(k).T x_c(k)`` and its sideslip
    follows from differencing THAT.  ``sideslip_deg(..., camera_rate_deg_s=)``
    is the continuous form of the same correction, so it must (i) reduce the
    error against the de-rotated truth by orders of magnitude and (ii) shrink
    with the time step.  Feeding the rate in the OPPOSITE sense -- report.md
    section 3(b)'s ``xdot_c - omega_c x x_c``, whose ``omega_c`` is the scene
    sense rather than this module's camera-in-world one -- makes the error
    *worse* than no correction at all, which is the whole reason the convention
    is pinned in the docstring.
    """
    ...

def test_prohibition_splithalf_over_sqrt2_is_never_the_floor_b3_judges_on():
    """``splithalf/sqrt(2)`` is reported for comparison and never used.

    Constructed so the two disagree in verdict: every axis passes against the
    split-half ratio and fails against the named floor.  B3 must return FAIL.
    """
    ...

def test_prohibition_an_unmeasured_condition_does_not_open_the_gate():
    """Stage 1 UNMEASURED with stages 2 and 3 passing must still refuse."""
    ...

def test_prohibition_identity_R_cw_at_an_invalid_pair_is_never_a_measurement():
    """A rejected pair integrates as identity ONLY to keep the frame index.

    The frame it produces must be flagged invalid, and the accumulated rotation
    must not jump: identity-filling is an index convention, and a consumer that
    reads it as "the camera was still" is the error P4's drop-the-pair rule
    exists to prevent.
    """
    ...

def test_prohibition_the_estimator_module_cannot_run_or_modify_a_fit():
    """Diagnosis and refusal only: no torch, no fitter, no fit driver.

    A static check on the source, because an accidental import here would let a
    future edit reach into the objective the gate is supposed to stand outside
    of.
    """
    ...
