"""The v3 anatomical shape basis and the shape-dependent rest skeleton.

``data/model_assets_v3`` is ``data/model_assets_v2`` with two things replaced:

1. the synthetic PCA (``shape_pca_components.npy``) becomes EIGHT named
   parametric anatomical modes, each calibrated so one unit of the parameter
   changes its own published length by exactly 1 % of TL, and each deflated
   against the girth field so shape and girth do not model the same thing;
2. a ``joint_regressor.npy`` plus ``skeleton.json["joint_source"] ==
   "regressor"``, so the rest skeleton is regressed from the SHAPED vertices and
   the joints stop being beta-blind.

What these tests pin, in order: the regressor reproduces the shipped rest
skeleton; its rows are well-formed; v1 and v2 stay on the explicit-joint path
they are on today, bit for bit; a v3 forward at ``betas = 0`` agrees with v2 to
float32; each mode moves exactly the anatomy it claims and preserves TL; the
basis is orthogonal to the girth field; the joints finally follow the betas; and
the per-mode shape prior is a no-op unless the asset asks for it.

Everything here is CPU-only and runs in a few seconds.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.core.constants import NUM_BETAS
from shark_pose.encoder.smplify_fitting import SpineSMPLify
from shark_pose.model_3d.shark_smpl import JOINT_REGRESSOR_REPRO_TOL, SharkSMPL
FLOAT32_FLOOR = 4e-06

@pytest.fixture(scope='module')
def model_v2() -> SharkSMPL:
    ...

@pytest.fixture(scope='module')
def model_v3() -> SharkSMPL:
    ...

@pytest.fixture(scope='module')
def shape_modes() -> dict:
    ...

@pytest.fixture(scope='module')
def basis() -> dict:
    """The v3 asset, as plain float64 arrays."""
    ...

def _poses(model: SharkSMPL, n: int=6, seed: int=0) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of non-trivial poses and translations, deterministically."""
    ...

def test_regressor_reproduces_rest_skeleton(basis):
    """J @ template must give back the rig's own measured bone positions.

    Spec is 1e-3 model units (0.022 % of the 4.54-unit body); the float32
    matrix lands at ~1.2e-06, inside it by ~840x.
    """
    ...

def test_regressor_rows_wellformed(basis):
    """Every row is an affine average of vertices, and at most two extrapolate.

    A row that sums to something other than 1 is not a position; a row with a
    large L1 is extrapolating, which is how a rebuild that quietly changed the
    support would show up. The two allowed exceptions are the right-flank
    pectoral and pelvic tips, whose shipped positions are exact mirrors of the
    LEFT landmarks and so sit outside the asymmetric mesh's own right blades.
    """
    ...

def test_regressor_tracks_the_analytic_joint_field(basis):
    """The regressed joints follow each mode the way the mode's own field does.

    ``shape_mode_joints.npy`` is the analytic displacement of the rig's rest
    joints under one unit of each mode; ``J @ S_k`` is what the model will
    actually do. They must agree, or the skeleton would drift away from the
    mesh it is supposed to describe.
    """
    ...

def test_v1_and_v2_keep_explicit_joints():
    """Neither shipped asset opts in, and their rest joints are still constant.

    v1 SHIPS a ``joint_regressor.npy`` that reproduces its own bone positions to
    only ~2e-02, so the switch must not key off the file's presence.
    """
    ...

def test_v2_joints_are_beta_blind_bit_for_bit(model_v2):
    """v2's rest skeleton is the constant buffer, whatever the betas are.

    This is the OLD behaviour, pinned: at a zero pose the posed joints are
    exactly ``explicit_joint_positions``, for any beta.
    """
    ...

def test_v3_forward_at_betas_zero_matches_v2(model_v2, model_v3):
    """The v3 asset is inert at ``betas = 0``: same mesh, same landmarks.

    At the REST pose the vertices and landmarks are bit-identical (LBS with
    identity rotations never reads the rest joints). Under real poses the two
    paths part by the float32 floor and no more -- ``J @ V`` and
    ``bone_positions`` are two float32 reductions of the same geometry.
    """
    ...

def test_load_refuses_an_inconsistent_regressor():
    """Opting in with a regressor that does not reproduce the asset's own rest
    skeleton must raise, and say by how much.

    v1's shipped regressor misses v1's own bone positions by ~2e-02, i.e. half a
    per cent of body length; silently switching to it would move every joint.
    """
    ...

def test_load_checks_the_regressor_against_the_mean_shape():
    """The guard must check what ``forward`` regresses from.

    ``forward`` takes ``joint_regressor @ shape_space(betas)``, and at
    ``betas = 0`` that is the MEAN SHAPE, not ``template_vertices``. They are
    the same array in every shipped asset, so an asset whose mean shape had been
    re-centred would sail through a template-based check and then place every
    rest joint somewhere else. Build exactly that asset and require a raise.
    """
    ...

def test_joints_follow_betas(model_v2, model_v3):
    """The regression guard for the bug this asset exists to fix.

    On v2 the joint Jacobian wrt the betas is EXACTLY zero on every component,
    so the midline / bend / chain-length terms have no shape gradient at all.
    On v3 every named mode moves the skeleton.
    """
    ...

def test_per_sample_rest_skeleton(model_v3):
    """Two different beta rows give two different rest skeletons, and the
    gradient reaches the leaf from a single joint."""
    ...

def test_each_mode_moves_its_own_named_quantity(basis, shape_modes):
    """One unit of a mode changes its own published length by exactly 1 % of TL.

    Measured on the JOINT FIELD THE MODEL READS -- ``joint_regressor @ S``, not
    the analytic ``shape_mode_joints.npy`` -- because that is what the fitter
    sees. The modes are calibrated on this derivative at build time, so this is
    the check that the calibration survived the girth deflation and the float32
    round trip into the asset.
    """
    ...

def test_realised_named_derivative_through_the_model(model_v3, shape_modes):
    """The unit means what it says once the WHOLE model has been through.

    P2's regression: the build used to calibrate the two partition modes with a
    joint endpoint on the analytic joint field, which the model never reads, so
    ``trunk_share`` realised 0.963 and ``peduncle_share`` 0.900 of their own
    named unit. Forward at ``betas = e_k`` and read the answer off the output.
    """
    ...

def test_each_mode_moves_only_the_landmarks_it_claims(basis, shape_modes):
    """The recorded signature is the whole signature.

    ``shape_modes.json`` names, per mode, the landmarks that move more than
    5 mm per unit. Anything moving that is NOT on the list is a leak -- a fin
    gate that spilled onto the body wall, or a partition ramp that reached a
    segment it was supposed to leave alone.
    """
    ...

def test_partition_modes_preserve_total_length(basis):
    """Absolute length is not in this basis and cannot be smuggled into it.

    Every mode is TL-preserving by construction; the residue is the smoothstep
    ramps and the fact that ``caudal_upper_tip`` sits at p = 0.9968 rather than
    at the mesh's aft-most vertex.
    """
    ...

def test_basis_is_orthogonal_to_the_girth_field(basis):
    """Shape and girth must not model the same deformation.

    The girth field already carries per-station width and depth and is
    identifiable from mask widths; the shape basis is deflated against its
    14-dimensional span at build time, so the overlap is numerically zero
    rather than merely small.
    """
    ...

def test_padding_components_are_inert(basis, shape_modes):
    """The basis has 8 modes and NUM_BETAS is 10; the slack must be zero.

    A zero component has no data gradient, so the prior simply holds it at its
    initial value -- safer than changing the shape dimension under every SPIN
    checkpoint.
    """
    ...

def test_shape_modes_are_surfaced_on_the_model(model_v3, shape_modes):
    """A fitted beta has to be readable, which means the model must know the
    names, the units and the literature the widths came from."""
    ...

def _prior_only(model, betas, betas_init, **kw) -> float:
    """``shape_prior_weight * prior(betas)``, isolated by differencing two
    fitters that agree on everything except that weight."""
    ...

def test_v2_shape_prior_is_the_shipped_isotropic_expression(model_v2):
    """Without a ``shape_modes.json`` the prior is unchanged, bit for bit."""
    ...

def test_v3_shape_prior_is_per_mode(model_v3, shape_modes):
    """With one, each mode is measured in its OWN literature sd.

    The components are in named units (one unit = 1 % of TL on that mode's
    published length), so an isotropic prior would penalise ``trunk_share``
    (sd 4 % of TL) sixteen times harder than it deserves and ``pectoral_length``
    (sd 0.83) not nearly hard enough.
    """
    ...

def test_explicit_shape_prior_sds_override_the_asset(model_v3):
    """An explicit argument wins, and a malformed one is refused."""
    ...

def test_pre_dorsal_is_a_definition_difference_not_a_bent_template():
    """The model's pre-dorsal ratio reads 0.46 TL against a published 0.37.

    The build measures the fin's own base on the mesh and records it: the
    ``first_dorsal_base`` landmark is the base MIDPOINT (the annotator's
    ``dorsal_base_front`` and ``dorsal_base_back`` are merged into it), while
    the publication measures to the fin's ORIGIN. Measured to the origin the
    template is inside one published sd, so the 9 pp gap is a definition, and
    ``dorsal_position``'s ``template_deficit_units`` must not be read as a
    geometry error.
    """
    ...

def test_girth_field_still_reads_the_rest_reference(model_v2, model_v3):
    """The girth operator stays built on ``explicit_joint_positions`` at
    ``betas = 0``, so which vertex belongs to which station -- and which vertex
    is fin -- is a definition, not something that moves with the fitted shape.
    """
    ...

@pytest.mark.skipif(not B2_ASSEMBLE.exists(), reason='needs the B2 eval sources')
def test_ship_rule_verdicts_are_computed_not_written():
    """Every clause verdict in ``eval/report.json`` comes from its own numbers.

    A hard-coded ``"verdict": "PASS"`` survives a re-run on new inputs and
    reports a result nobody measured. The assembler is allowed exactly one
    place that produces the strings -- the ``verdict()`` helper -- and the
    clause bodies must reach it through an expression.
    """
    ...
