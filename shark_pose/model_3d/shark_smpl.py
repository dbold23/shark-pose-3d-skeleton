"""
SharkSMPL: Parametric 3D shark model analogous to SMPL for humans.

Parameters:
    theta: (batch, J*3) pose — J joints x 3 axis-angle
    beta:  (batch, 10) shape — PCA coefficients
    t:     (batch, 3)  translation in camera frame

The joint count and kinematic tree come from the ASSET DIRECTORY: a directory
carrying ``skeleton.json`` (the v2 31-joint midline spine rig) is loaded with
that rig's joints and parents; without it the model falls back to the 16-joint
defaults in ``shark_pose.core.skeleton`` (v1), byte-for-byte as before. The 16
OBSERVATION landmarks are independent of the rig -- they are surface points on
the shared template -- so ``SharkSMPLOutput.landmarks`` is always (B, 16, 3).

Forward pass:
    beta -> shape offsets -> rest vertices
    girth -> per-station cross-section scaling of those rest vertices (optional)
    theta -> rotation matrices -> pose blend shapes
    LBS(rest_verts + pose_offsets, rotations, weights) -> posed mesh
    J_regressor @ posed_verts -> J skeleton joints
    D_regressor @ posed_verts -> 91 surface landmarks
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from ..core.constants import CAMERA_DIM, NUM_BETAS, NUM_JOINTS, NUM_SURFACE_LANDMARKS
from ..core.skeleton import PARENT_INDICES
from ..core.transforms import axis_angle_to_rotation_matrix
from .pose_space import PoseBlendShapes
from .shape_space import ShapeSpace
from .skinning import lbs

class SharkSMPLOutput:
    """Container for SharkSMPL forward pass outputs."""

    def __init__(self, vertices: Tensor, joints: Tensor, surface_landmarks: Tensor, rotation_matrices: Tensor, shaped_vertices: Tensor, landmarks: Tensor | None=None, landmarks_mirror: Tensor | None=None):
        ...

def select_visible_landmarks(landmarks: Tensor, landmarks_mirror: Tensor, bilateral: Tensor) -> Tensor:
    """Pick, per bilateral landmark, the instance facing the camera.

    A shark filmed from its left side shows its left eye/gill/pectoral; the
    right-flank instances of those landmarks are hidden behind the body. The
    camera looks down +z, so the instance with the SMALLER camera-frame z is
    the near (visible) one.

    The selection is a hard, DETACHED mask: which side is visible is a discrete
    observation about the current pose, not something the optimizer should be
    able to differentiate through (a soft blend would let the fit average two
    anatomically distinct points).

    Args:
        landmarks: (B, 16, 3) left-instance landmarks in camera frame.
        landmarks_mirror: (B, 16, 3) right-instance landmarks in camera frame.
        bilateral: (16,) mask, true where the landmark is anatomically paired.
            Midline landmarks always take ``landmarks``.

    Returns:
        (B, 16, 3) visible-side landmarks.
    """
    ...
GIRTH_NUM_STATIONS = 7
GIRTH_GATE_START = 1.2
GIRTH_GATE_END = 1.6
GIRTH_ENVELOPE_BINS = 32
GIRTH_ENVELOPE_QUANTILE = 0.95
GIRTH_ENVELOPE_SMOOTH_BINS = 1.0
GIRTH_ENVELOPE_MIN_COUNT = 4
GIRTH_FIN_WEIGHT_MAX = 0.2
JOINT_REGRESSOR_REPRO_TOL = 0.001

def _interp1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation of ``fp`` at ``x`` (``xp`` ascending, flat ends)."""
    ...

def catmull_rom_basis(fractions: Tensor, num_stations: int=GIRTH_NUM_STATIONS) -> Tensor:
    """Catmull-Rom interpolation weights from uniform stations to arbitrary x.

    The girth field is interpolated with a uniform Catmull-Rom spline, which is
    LINEAR in the control values: the whole interpolation therefore collapses
    to a constant (M, K) matrix that can be precomputed once and reused every
    forward pass.  End tangents are clamped (the end control point is repeated),
    so the spline is C^1 inside and never over/undershoots off the ends.

    Args:
        fractions: (M,) along-body fractions in [0, 1].
        num_stations: K, number of uniformly spaced control stations.

    Returns:
        (M, K) weights; each row sums to 1 and reproduces the control values
        exactly at the station fractions.
    """
    ...

def girth_prior_loss(girth: Tensor, smooth_weight: float=1.0, mean: Optional[Tensor]=None) -> Tensor:
    """Prior pulling the girth field to a MEAN and keeping it smooth along the body.

    Args:
        girth: (..., K, 2) per-station (g_x, g_z) scales.
        smooth_weight: weight of the along-body first-difference term.
        mean: **TASK F1, default None.**  Where the field is pulled TO.
            ``None`` is the shipped behaviour -- the unscaled template, i.e.
            1.0 on both axes -- and takes a separate code path so a fit that
            does not set it is bit-identical to one built before this argument
            existed.  Anything else is broadcast against ``girth``: a scalar, a
            ``(2,)`` pair (one mean per AXIS, which is what a literature girth
            ratio gives -- the template's cross-section is a constant of the
            asset, so "the published animal is 7% wider and 27% deeper than
            this template" is two numbers), or a full ``(K, 2)`` field.

            It moves the DEVIATION term only.  The smoothness term is a
            first difference and a constant mean cancels in it exactly; a
            per-station mean does not, and then the smoothness is measured
            against the mean's own profile, which is the intended reading
            ("smooth like the prior says a shark is", not "flat").

    Returns:
        scalar ``mean((g - mu)^2) + smooth_weight * mean((d[k+1] - d[k])^2)``
        where ``d = g`` for a constant mean (``None``, a scalar or a ``(2,)``
        pair -- the first difference of a constant is zero, so the two forms
        agree) and ``d = g - mu`` for a per-station ``(K, 2)`` mean.
    """
    ...

def girth_from_width_ratio(fractions, ratios, num_stations: int=GIRTH_NUM_STATIONS, model_over_target: bool=False, clamp: tuple[float, float]=(0.5, 2.0), kernel_sigma: float | None=None) -> Tensor:
    """Turn measured per-station silhouette WIDTH RATIOS into a starting girth.

    Args:
        fractions: (M,) along-body fractions of the measurements, in [0, 1].
        ratios: (M,) width ratios.  By default these are TARGET / MODEL, i.e.
            the factor by which the model must widen.  The width scorers in
            this repo report MODEL / MASK instead -- pass
            ``model_over_target=True`` and the ratios are inverted here.
        num_stations: K stations of the returned field.
        model_over_target: invert the ratios before use (see above).
        clamp: (lo, hi) bounds on the returned scales.
        kernel_sigma: Gaussian width, in fraction units, of the kernel that
            resamples the measurements onto the stations.  Defaults to half the
            station spacing, which averages the ~32 arc samples of a centreline
            record onto the K stations instead of point-sampling them.

    Returns:
        (K, 2) girth field; g_x and g_z are both set to the measured ratio.
        A silhouette width does NOT single out the lateral axis: it is
        measured normal to the projected spine, so it reads whichever
        cross-section axis lies in the IMAGE PLANE -- ``g_z`` on a broadside
        flank view (the lateral axis points at the camera there), ``g_x``
        top-down.  Which one that is depends on the clip, not on this
        function, so setting both to the measured ratio is the least-committal
        starting point; ``fit_video.girth_observed_axis`` reports which axis
        the clip actually constrained.
    """
    ...

def build_girth_operator(vertices: Tensor, axis_points: Tensor, fin_weight: Tensor | None=None, num_stations: int=GIRTH_NUM_STATIONS) -> dict[str, Tensor]:
    """Precompute the constant per-vertex quantities the girth field needs.

    Everything here depends only on the REST template and the rest spine
    polyline, so it is computed once at construction time.  The forward pass is
    then a single (N, K) x (K, 2) matmul plus a per-vertex scale.

    Args:
        vertices: (N, 3) rest template vertices.
        axis_points: (S, 3) rest spine polyline, snout -> tail.
        fin_weight: (N,) per-vertex "how much of this vertex belongs to a fin"
            in [0, 1] (the summed LBS weight of the non-spine joints).  Used
            ONLY to keep fin vertices out of the body-radius estimate; None
            means every vertex is treated as body.
        num_stations: K control stations.

    Returns:
        dict with
            ``axis_point`` (N, 3) nearest point on the rest spine polyline,
            ``fraction``   (N,)   its arc fraction along that polyline,
            ``basis``      (N, K) Catmull-Rom weights, rows summing to 1,
            ``gate``       (N,)   1 on the body, 0 on the fins, smooth between,
            ``body_radius``(N,)   the local body radius used by the gate.
    """
    ...

class SharkSMPL(nn.Module):
    """Parametric 3D shark body model.

    Matches the SMPL interface so that SPIN/SMPLify code ports directly.
    """

    def __init__(self, template_vertices: Tensor, faces: Tensor, joint_regressor: Tensor, lbs_weights: Tensor, shape_space: ShapeSpace, pose_blend_shapes: PoseBlendShapes, surface_landmark_regressor: Tensor | None=None, explicit_joint_positions: Tensor | None=None, landmark_regressor: Tensor | None=None, landmark_regressor_mirror: Tensor | None=None, landmark_bilateral: Tensor | None=None, parents: Sequence[int] | None=None, joint_names: Sequence[str] | None=None, girth_spine_joints: Sequence[int] | None=None, num_girth_stations: int=GIRTH_NUM_STATIONS, shape_dependent_joints: bool=False, shape_modes: dict | None=None):
        """
        Args:
            template_vertices: (N, 3) canonical T-pose mesh vertices
            faces: (F, 3) face indices
            joint_regressor: (J, N) matrix mapping vertices to joint locations. May
                be None when ``explicit_joint_positions`` supplies the rest joints
                (the v2 rig ships measured bone positions and no regressor).
            lbs_weights: (N, J) per-vertex skinning weights. Its column count
                DEFINES the rig's joint count.
            shape_space: PCA shape model
            pose_blend_shapes: pose-dependent corrective blend shapes
            surface_landmark_regressor: (91, N) optional dense surface landmark regressor
            explicit_joint_positions: (J, 3) optional exact joint positions from the rig.
                When provided, these are used as rest-pose joint locations instead of
                computing them via joint_regressor @ shaped_verts.
            landmark_regressor: (16, N) optional OBSERVATION landmark regressor —
                calibrated mesh-surface positions of the 16 annotated keypoints
                (left-flank instance for paired landmarks). Rows that are entirely
                zero mark INTERIOR landmarks with no surface definition (the body
                midpoints, second_dorsal_base); those fall back to the posed joint.
            landmark_regressor_mirror: (16, N) the right-flank instance, i.e. the
                same regressor mirrored across the sagittal plane.
            landmark_bilateral: (16,) mask, true where the landmark is anatomically
                paired (eye, gill, pectoral base/tip, pelvic fin tip).
            parents: parent index per joint (-1 for the root). Defaults to the
                16-joint ``PARENT_INDICES``; the v2 rig passes its own 31-entry tree.
            joint_names: optional names, one per joint, for reporting.
            girth_spine_joints: indices of the joints forming the midline spine
                chain, snout -> tail. When given (together with
                ``explicit_joint_positions``) the model gains the GIRTH field:
                their rest positions are the axis the cross-section is scaled
                about, and the remaining joints are taken to be fin joints,
                whose LBS weights mark the vertices excluded from the local
                body-radius estimate. Without it ``forward(girth=...)`` raises.
            num_girth_stations: K control stations of the girth field.
            shape_dependent_joints: when True the REST SKELETON is regressed from
                the SHAPED vertices (``joint_regressor @ shaped_verts``) instead
                of read from ``explicit_joint_positions``, so the joints -- and
                with them the midline, bend and chain-length terms -- follow the
                betas. Requires ``joint_regressor``; at load time
                ``joint_regressor @ shape_space.mean_shape`` -- the beta = 0
                vertices ``forward`` regresses from -- is checked against
                ``explicit_joint_positions``, because a regressor that does not
                reproduce its own asset's rest skeleton would silently move
                every joint. v1 and v2 leave it False and are byte-for-byte
                unchanged; the v3 asset declares ``"joint_source": "regressor"``
                in ``skeleton.json``.
            shape_modes: the asset's ``shape_modes.json``, when it ships one --
                the named anatomical basis: per-beta name, unit, prior sd and
                literature range. Surfaced as ``shape_mode_names`` /
                ``shape_mode_prior_sds`` for the fitter's per-mode shape prior.
                None (v1, v2) leaves both None and every prior isotropic.
        """
        ...

    @property
    def has_girth(self) -> bool:
        """True when the asset supplied a spine chain, so girth is available."""
        ...

    def girth_identity(self, batch_size: int=1) -> Tensor:
        """(B, K, 2) girth field that leaves the mesh bit-identical."""
        ...

    def apply_girth(self, vertices: Tensor, girth: Tensor) -> Tensor:
        """Scale the rest mesh's cross-section by a per-station girth field.

        For every vertex, the lateral (x) and dorsoventral (z) offsets from its
        nearest point on the rest spine axis are multiplied by the interpolated
        (g_x, g_z); the longitudinal (y) coordinate never moves, so the girth
        field changes girth and not length. The scaling is faded out on the
        fins by ``girth_gate``.

        Written as ``v + offset * (scale - 1)`` rather than
        ``axis + offset * scale`` so that ``girth == 1`` returns the input
        BIT-IDENTICALLY (the latter would round-trip through the axis point).

        Args:
            vertices: (B, N, 3) rest vertices.
            girth: (B, K, 2) or (K, 2) per-station (g_x, g_z).

        Returns:
            (B, N, 3) rest vertices with the girth field applied.
        """
        ...

    @property
    def has_landmarks(self) -> bool:
        """True when the calibrated observation-landmark asset was loaded."""
        ...

    def visible_landmarks(self, output: 'SharkSMPLOutput') -> Tensor:
        """Convenience wrapper: visible-side observation landmarks for an output."""
        ...

    def forward(self, pose: Tensor, betas: Tensor, translation: Tensor | None=None, girth: Tensor | None=None) -> SharkSMPLOutput:
        """Forward pass: parameters -> posed mesh + joints + landmarks.

        Args:
            pose: (batch, J*3) axis-angle per joint, or (batch, J, 3)
            betas: (batch, 10) shape coefficients
            translation: (batch, 3) global translation. If None, no translation applied.
            girth: (batch, K, 2) or (K, 2) per-station (g_x, g_z) cross-section
                scales, applied to the REST mesh before LBS. None (the default)
                skips the deformation entirely; an all-ones field is a no-op
                bit-for-bit. Joints are NOT affected -- they sit on the axis the
                scaling is measured from -- but the surface landmarks are, which
                is the point: the flank landmarks follow the body wall.

        Returns:
            SharkSMPLOutput with posed vertices, joints, surface landmarks, rotations.
        """
        ...

    @classmethod
    def from_assets(cls, assets_dir: str | Path) -> 'SharkSMPL':
        """Load SharkSMPL from a directory of numpy/mesh assets.

        Expected files in assets_dir:
            - shark_template_mesh.obj (or .npy for vertices)
            - joint_regressor.npy (J, N)  [optional when bone_positions.npy exists]
            - lbs_weights.npy (N, J)
            - shape_pca_components.npy (K, N, 3)
            - mean_shape.npy (N, 3)  [optional, defaults to template verts]
            - pose_blend_shapes.npy (J*9, N, 3)  [optional]
            - surface_landmark_regressor.npy (91, N)  [optional]
            - faces.npy (F, 3)
            - skeleton.json  [optional] the rig's ``joint_names``/``parents``;
              present in data/model_assets_v2 (31-joint midline spine rig),
              absent in data/model_assets (v1, 16 joints). Its optional
              ``joint_source`` key selects where the REST joints come from:
              ``"regressor"`` (v3) regresses them from the shaped vertices, so
              the skeleton follows the betas; anything else, or absent (v1, v2),
              keeps ``bone_positions.npy``.
            - shape_modes.json  [optional] the named anatomical shape basis:
              per-beta name, unit, prior sd and literature range (v3).

        Args:
            assets_dir: path to model assets directory
        """
        ...

    @classmethod
    def create_placeholder(cls, num_verts: int=1000, num_faces: int=1800) -> 'SharkSMPL':
        """Create a placeholder model for testing (random mesh, no real anatomy).

        Args:
            num_verts: number of vertices
            num_faces: number of faces
        """
        ...
