"""
SharkSMPLify: Optimization-based fitting of SharkSMPL to 2D keypoints.

Provides two optimization strategies:
    1. Neural Descent (HUND): An RNN-based learned optimizer that takes
       the current parameter vector and its gradient as input and outputs
       an update step. This is the primary method.
    2. Classical fallback: Adam or L-BFGS optimization for cases where
       the neural optimizer is unavailable.

Fitting proceeds in two hierarchical stages:
    Stage A (pose-first): Optimize pose parameters while keeping shape
        near the regressor initialization. 60 optimization steps.
    Stage B (shape refinement): Jointly refine pose and shape. 40 steps.

Uses Geman-McClure robust error function for 2D reprojection to handle
outlier keypoint detections gracefully.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Tuple
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..core.constants import CAMERA_DIM, NUM_JOINTS, NUM_BETAS, POSE_DIM, SHAPE_DIM, PARAM_DIM
from ..core.camera import PerspectiveCamera
from ..core.transforms import axis_angle_to_rotation_matrix
from ..losses.joint_limits import JointLimitLoss
from ..losses.soft_silhouette import BASE_FOCAL, BASE_RESOLUTION, DEFAULT_SIGMA_PROXY, SCANLINE_SIGMA_PX, SCANLINE_STEP_PX, sigma_at, soft_iou, soft_scanline_width, soft_silhouette
from ..model_3d.shark_smpl import GIRTH_NUM_STATIONS, SharkSMPL, girth_from_width_ratio, girth_prior_loss, select_visible_landmarks

def geman_mcclure(residuals: Tensor, sigma: float=1.0) -> Tensor:
    """Geman-McClure robust error: rho(r) = r^2 / (sigma^2 + r^2).

    Provides soft rejection of outliers. As |r| -> inf, rho -> 1 (bounded).
    This is strictly better than L2 for noisy keypoint detections.

    Args:
        residuals: Tensor of any shape containing residual values.
        sigma: Scale parameter controlling the outlier transition point.
            Smaller sigma = more aggressive outlier rejection.

    Returns:
        Robust loss values with same shape as input.
    """
    ...

class NeuralDescentRNN(nn.Module):
    """GRU-based learned optimizer for parameter updates.

    Inspired by HUND (Learned Gradient Descent for Inverse Problems):
    takes the current parameter vector and the gradient of the fitting
    objective as input and produces an update step.

    The GRU maintains hidden state across optimization steps, allowing
    the optimizer to learn momentum-like behavior and adaptive step sizes.
    """

    def __init__(self, param_dim: int=PARAM_DIM, hidden_dim: int=256, num_gru_layers: int=2):
        """
        Args:
            param_dim: Dimensionality of the parameter vector (61).
            hidden_dim: GRU hidden state size.
            num_gru_layers: Number of stacked GRU layers.
        """
        ...

    def forward(self, params: Tensor, gradient: Tensor, hidden: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        """Compute one optimization step.

        Args:
            params: (B, param_dim) current parameter vector.
            gradient: (B, param_dim) gradient of loss w.r.t. params.
            hidden: (num_layers, B, hidden_dim) GRU hidden state.
                If None, zero-initialized.

        Returns:
            update: (B, param_dim) parameter update step.
            hidden: Updated GRU hidden state.
        """
        ...

class SharkSMPLify:
    """Optimization-based fitting of SharkSMPL to 2D keypoint observations.

    Two-stage hierarchical fitting:
        Stage A: Pose optimization (60 steps) with shape regularized to init.
        Stage B: Joint pose + shape refinement (40 steps).

    Supports both neural descent (learned optimizer) and classical Adam/L-BFGS.
    """

    def __init__(self, shark_model: SharkSMPL, neural_optimizer: Optional[NeuralDescentRNN]=None, focal_length: float=1000.0, image_size: tuple[int, int]=(256, 256), stage_a_steps: int=60, stage_b_steps: int=40, gm_sigma: float=40.0, shape_prior_weight: float=1.0, pose_prior_weight: float=1.0, lr: float=0.01, use_neural_descent: bool=True, temporal_weight: float=0.0, temporal_loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]]=None, use_landmarks: bool=False, joint_limit_weight: float=0.0):
        """
        Args:
            shark_model: SharkSMPL body model instance.
            neural_optimizer: Trained NeuralDescentRNN. If None and
                use_neural_descent=True, one will be created (untrained).
            focal_length: Camera focal length in pixels.
            image_size: (H, W) image resolution.
            stage_a_steps: Number of steps for Stage A (pose-first).
            stage_b_steps: Number of steps for Stage B (shape refinement).
            gm_sigma: Geman-McClure sigma for robust reprojection error.
            shape_prior_weight: Weight for shape regularization.
            pose_prior_weight: Weight for pose prior regularization.
            lr: Learning rate for classical optimizer fallback.
            use_neural_descent: If True, use NeuralDescentRNN; else Adam.
            temporal_weight: If > 0, the batch dimension of a fit() call is
                treated as TIME (per-clip test-time optimization) and
                ``temporal_loss_fn`` couples the frames. Default 0.0 leaves
                the single-frame behaviour byte-for-byte unchanged.
            temporal_loss_fn: Callable ``(pose (T,48), joints_3d (T,16,3))
                -> scalar`` returning a temporal-smoothness loss (it extracts
                the root joint itself). Only invoked when
                ``temporal_weight > 0``. Build one with ``make_temporal_loss_fn``.
            use_landmarks: Drive the 2D data term with the calibrated OBSERVATION
                landmarks (the visible flank per bilateral landmark) instead of the
                kinematic joints. The rig's pivots sit off the midline and away from
                the anatomical points a human annotator clicks, so fitting joints to
                human labels forces implausible rotations. Requires
                ``keypoint_landmarks.npz`` in the model assets; without that asset
                the landmarks equal the joints and this flag is a no-op. Default
                False leaves every existing caller unchanged.
            joint_limit_weight: Weight on the anatomical joint-angle hinge penalty
                (``JointLimitLoss``). 0.0 disables it.
        """
        ...

    def fit(self, init_pose: Tensor, init_shape: Tensor, init_camera: Tensor, keypoints_2d: Tensor, keypoint_conf: Tensor) -> Dict[str, Tensor]:
        """Fit SharkSMPL to 2D keypoint observations.

        Args:
            init_pose: (B, 48) initial axis-angle pose (from regressor).
            init_shape: (B, 10) initial shape betas (from regressor).
            init_camera: (B, 3) initial camera translation (from regressor).
            keypoints_2d: (B, 16, 2) detected 2D keypoint locations.
            keypoint_conf: (B, 16) keypoint confidence scores in [0, 1].

        Returns:
            Dictionary with optimized parameters:
                pose: (B, 48) optimized axis-angle pose
                shape: (B, 10) optimized shape betas
                camera: (B, 3) optimized camera translation
                reprojection_error: (B,) final mean reprojection error
        """
        ...

    def _fit_neural(self, init_pose: Tensor, init_shape: Tensor, init_camera: Tensor, keypoints_2d: Tensor, keypoint_conf: Tensor) -> Dict[str, Tensor]:
        """Fit using the NeuralDescentRNN learned optimizer."""
        ...

    def _fit_classical(self, init_pose: Tensor, init_shape: Tensor, init_camera: Tensor, keypoints_2d: Tensor, keypoint_conf: Tensor) -> Dict[str, Tensor]:
        """Fit using classical Adam optimization."""
        ...

    def _compute_fitting_loss(self, pose: Tensor, shape: Tensor, camera: Tensor, keypoints_2d: Tensor, keypoint_conf: Tensor, shape_init: Tensor, shape_weight: float, pose_init: Optional[Tensor]=None) -> Tensor:
        """Compute the total fitting objective.

        Components:
            1. Geman-McClure robust 2D reprojection error (confidence-weighted)
            2. Shape regularization: ||beta - beta_init||^2
            3. Pose regularization: ||pose - pose_init||^2

        The pose term regularizes toward ``pose_init`` (the incoming estimate),
        NOT toward the zero/rest pose. Under-observed footage (only ~5/16 joints
        visible) cannot constrain the full pose, so a toward-zero prior collapses
        the shark to rest and *raises* reprojection error on the visible joints
        (measured: 66 -> 114px on real data). Regularizing toward the estimate
        holds the unobserved joints in place while the data term fits the visible
        ones. Falls back to toward-zero only if no init is supplied.
        """
        ...

    def _project_observations(self, smpl_out) -> Tensor:
        """Project whatever the 2D observations correspond to, to pixels.

        With ``use_landmarks`` that is the visible-flank observation landmarks;
        otherwise the kinematic joints (the historical behaviour). Falls back to
        the joints automatically when the model carries no landmark asset, since
        ``SharkSMPLOutput.landmarks`` then aliases ``joints``.
        """
        ...

    def _reprojection_error(self, pose: Tensor, shape: Tensor, camera: Tensor, keypoints_2d: Tensor, keypoint_conf: Tensor) -> Tensor:
        """Compute mean confidence-weighted reprojection error per sample.

        Returns:
            error: (B,) mean pixel error per sample.
        """
        ...

def _axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
    """(..., 3) axis-angle -> (..., 4) quaternion (w, x, y, z).

    Uses the sinc series near zero so the identity rotation -- the starting
    point of every fit -- has a finite, correct gradient.
    """
    ...

def _quaternion_to_axis_angle(q: Tensor) -> Tensor:
    """(..., 4) quaternion (w, x, y, z) -> (..., 3) axis-angle.

    The generic matrix log goes through ``acos`` and has an infinite gradient at
    the identity; this form is the same series trick as
    ``spine_prior.ypr_to_axis_angle`` and stays smooth there.
    """
    ...

def compose_axis_angle(outer: Tensor, inner: Tensor) -> Tensor:
    """Axis-angle of ``R(outer) @ R(inner)``, differentiable at the identity."""
    ...

def retarget_translation(root_orient: Tensor, translation: Tensor, pivot_from: Tensor, pivot_to: Tensor) -> Tensor:
    """Move a global translation from one rig's root pivot to another's.

    Both rigs rotate the body about their own root joint, so a v1 fit's
    ``(R, t)`` places a template point ``x`` at ``R(x - p1) + p1 + t``.  Feeding
    the same ``t`` to a rig whose root sits at ``p2`` would shift the whole
    shark by ``(I - R)(p1 - p2)``; this returns the ``t2`` that reproduces the
    v1 rigid transform exactly:  ``t2 = t + (I - R)(p1 - p2)``.

    Args:
        root_orient: (B, 3) global axis-angle.
        translation: (B, 3) translation that goes with ``pivot_from``.
        pivot_from: (3,) rest position of the source rig's root joint.
        pivot_to: (3,) rest position of the target rig's root joint.
    """
    ...
BEND_RESAMPLE_POINTS = 32
BEND_FRONT_FRAC = 0.4
BEND_BACK_FRAC = 0.4
WIDTH_FIN_RADIUS_FACTOR = 1.3
WIDTH_TAU_PX = 1.0
WIDTH_SECTION_TAU_PX = 0.0
WIDTH_SECTION_MIN_EDGES = 8
WIDTH_SECTION_NUDGE_FRAC = 0.02
WIDTH_SECTION_NUDGE_MAX = 25
WIDTH_SCANLINE_HALF_SCALE = 2.5

def _soft_extent(offsets: Tensor, mask: Tensor, tau: float) -> Tensor:
    """Differentiable ``max - min`` of ``offsets`` over the last axis.

    Both ends are softmax-weighted extrema -- ``sum(softmax(x/tau) * x)`` and
    its mirror.  That is the gradient of ``tau * logsumexp(x / tau)`` applied to
    ``x``, so it is the same construction, but it can never report a body wider
    than its own widest vertex: it is exact for a degenerate set, exact in the
    limit ``tau -> 0``, and otherwise strictly INSIDE the hard extent.
    ``tau * logsumexp`` itself is not -- its bias is OUTWARD and grows as
    ``tau * log(n)`` with the band's point count, which reaches 1016 on this
    template.  Measured on the rest mesh at ``tau`` 1 px, over the 13 stations
    of a broadside and a top-down view: this estimator sits 0.1-1.8 px inside
    the hard extent, ``tau * logsumexp`` 0.0-2.8 px outside it.  A width the mesh does not have is the worse error of the two,
    and the only one that can be paid for by shrinking the animal.

    ``mask`` is True on the real entries of the padded band; padding is pushed
    to -inf in both softmaxes, so it can neither win an extreme nor take weight.
    """
    ...

def _resample_polyline_torch(points: Tensor, n_points: int) -> Tensor:
    """Arc-length resample (M, K, 2) ordered polylines to (M, n_points, 2).

    The torch twin of ``mask_centerline.resample_polyline`` (which is
    ``np.interp`` on the cumulative chord length).  Differentiable in the input
    points: the bracketing indices are piecewise constant in them, while the
    blend weights and the bracketing points carry the gradient.
    """
    ...

def _chord_direction_torch(points: Tensor) -> Tensor:
    """Total-least-squares direction of (M, n, 2) point sets, signed first->last.

    ``mask_centerline._chord_direction`` takes the first right singular vector
    of the centred points; in 2D that eigenvector has the closed form
    ``theta = atan2(2 Sxy, Sxx - Syy) / 2``, which is what is used here because
    ``torch.linalg.svd``'s gradient is undefined when the two singular values
    coincide (a straight chord is exactly that case).  Same vector, defined
    everywhere.
    """
    ...

def projected_station_bend_deg(points_2d: Tensor, n_points: int=BEND_RESAMPLE_POINTS, front_frac: float=BEND_FRONT_FRAC, back_frac: float=BEND_BACK_FRAC) -> Tensor:
    """Signed head-chord to tail-chord angle of projected stations, in degrees.

    ``points_2d`` is (M, K, 2) image pixels ordered snout -> tail (the 13 v2
    spine stations).  The chain is arc-length resampled to ``n_points`` and the
    angle is taken between the total-least-squares directions of its first
    ``front_frac`` and last ``back_frac``, positive when the tail chord is
    rotated clockwise on screen (image y DOWN).

    This is the differentiable twin of
    ``sim2real.mask_centerline.bend_of_projected_stations``: same resampling,
    same chords, same sign convention, and ``tests/test_model_v2.py`` pins the
    two together to better than 0.05 deg.  A fit and a mask centreline are
    therefore compared by one measurement, and the difference between them is
    body curvature rather than a difference of instrument.
    """
    ...
SWIM_UP_PROXY = (0.0, -1.0, 0.0)
SWIM_SMOOTHING_WINDOW_S = 0.5

def _parse_root_temporal_weight(w: Optional[float | Sequence[float]]) -> Tuple[Optional[float], Optional[Tuple[float, float, float]]]:
    """``(scalar_weight, per_axis_weights)`` from the leaf's two accepted forms.

    ``None`` -> ``(None, None)``, the shipped coupling.  A scalar -> that
    scalar and no axis weights.  A 3-sequence whose entries are all EQUAL is
    the scalar case and is reduced to it, so an "isotropic" triple runs the
    untouched geodesic path and is bit-identical to the scalar; only a
    genuinely anisotropic triple returns axis weights.
    """
    ...

def _log_rotvec(S: Tensor) -> Tensor:
    """(..., 3) rotation vector of the rotation stack ``S``, radians.

    The same angle :meth:`SpineSMPLify._geodesic_sq` reads -- ``atan2(|sin|,
    cos)`` off the skew part and the trace -- so ``|_log_rotvec(S)| ** 2``
    equals ``_geodesic_sq(I, S)`` to floating point.  At coincidence the scale
    factor ``theta / |sin theta|`` is taken from its series ``1 + n^2 / 6``
    rather than evaluated as 0/0, which keeps the gradient finite;
    ``core.transforms.rotation_matrix_to_axis_angle`` cannot, because it clamps
    ``sin`` and sits on ``acos``'s plateau there.
    """
    ...

def _swim_boxcar_matrix(n: int, k: int, device, dtype) -> Tensor:
    """(n, n) P7 zero-phase boxcar with the ``shrink_symmetric`` edge rule.

    As a MATRIX, so the low-pass the swimming terms see is differentiable and
    is exactly ``world_frame.boxcar_zero_phase`` -- the operator every reported
    drift statistic is measured with -- rather than a look-alike.
    """
    ...

@dataclass
class ClipContext:
    """Everything one clip contributes to an objective, computed once.

    TASK B1.  This is a *pure extraction* of the block :meth:`SpineSMPLify.fit`
    used to run before its amplitude-well loop -- the observation casts, the
    midline / bend / width / length correspondences, the time base and the
    silhouette batchers.  It exists so that a JOINT fit over the several clips
    of one individual can hold N of them at once while
    :class:`~shark_pose.encoder.joint_fitting.JointSpineFit` drives one
    optimiser over the shared girth field and betas.

    ``fitter`` is the clip's OWN :class:`SpineSMPLify`: fps, focal length,
    principal point and the ego track differ inside a single individual (the
    FAR20110101 pair is 59.94 fps against 47.96, the AN15092101 pair
    1920x1440 against 1920x1080), so there is one fitter per clip and no
    mutable per-clip state is ever swapped on a shared one.
    """
    fitter: 'SpineSMPLify'
    n_frames: int
    device: object
    dtype: object
    init_root_orient: Tensor
    init_translation: Tensor
    times: Tensor
    betas_init: Tensor
    keypoints_2d: Tensor
    keypoint_conf: Tensor
    midline: object = None
    bend: object = None
    width: object = None
    length: object = None
    next_sil_batch: object = None
    score_sil_batch: object = None
    clip_id: Optional[str] = None

    def loss_args(self, girth):
        """The positional tail :meth:`SpineSMPLify._loss` takes after ``raw``.

        Ordered exactly as ``fit()`` orders it, so a call site can splat this
        and get the shipped statement sequence.
        """
        ...

class SpineSMPLify:
    """Fit the v2 spine rig to 2D observation landmarks over a clip.

    Optimises, per clip:

    * ``SpinePoseParams`` (25 numbers; the wave -- amplitude, FREQUENCY and
      phase -- is shared by the whole clip, everything else is per frame),
    * the global root rotation (3) and translation (3), per frame,
    * the shape betas (10), shared by the clip and only weakly free,
    * optionally (``girth_enabled``) the GIRTH field (K x 2 per-station
      cross-section scales, K = 7), shared by the clip and free in stage B
      only.  The betas cannot make the animal fatter -- the shipped PCA basis
      has no girth mode -- so without this leaf the only way the fit can pay a
      width error is by moving the body in depth.  With it off, ``girth=None``
      reaches :class:`SharkSMPL` and the fit is bit-identical to one built
      before the leaf existed.

    The objective is the same Geman-McClure visible-side landmark term as
    :class:`SharkSMPLify`, plus the prior's negative log density
    (``SpinePrior.prior_loss``), a hinge on angles outside the per-joint limits
    measured from the sevengill animations, and temporal coupling that keeps
    ``turn(t)``, the root rotation and the translation smooth across the clip.

    Set ``share_clip_params=False`` to fit UNRELATED frames (an evaluation set):
    every parameter then becomes per frame, the time base collapses to zero and
    the temporal terms are skipped.  With ``t = 0`` the wave degenerates to its
    phase, so the frequency is frozen at its init there rather than optimised.

    In clip mode the wave's three clip-wide numbers are free leaves,
    warm-started by a coarse (amplitude x frequency x phase) grid search over
    :data:`DEFAULT_AMP_GRID` x :data:`DEFAULT_FREQ_GRID_HZ` (see
    :meth:`_init_wave`): the objective is periodic in phase, multi-modal in
    frequency, and the mixture amplitude prior has two wells (glide ~0.05,
    swimming ~0.55) with a barrier Adam never crosses, so a gradient fit alone
    would only ever find the well it started in.  Stage A + B is then re-run
    from the best start in EACH amplitude well and the lower converged
    objective wins (:attr:`amp_well_scores` records both), so the reported
    amplitude is the MAP under the mixture prior rather than an artefact of
    ``default_amp``.  Fitting one clip over the WHOLE video with a global time
    base is what makes the search meaningful -- a 24-frame window is 0.4 s and
    cannot resolve a 2-3 s beat.

    Two optional terms bring the MASK, not just the detector, into the fit.
    ``midline_weight > 0`` plus per-frame centrelines handed to :meth:`fit`
    matches the 13 projected spine stations to the mask centreline in arc-length
    order (:meth:`_midline_loss`); it runs in both stages because the 16
    observation landmarks are near-midline and anterior, so they leave the
    body's roll -- and therefore the plane the tail sweeps in -- almost free.
    ``upright_weight > 0`` adds a smooth hinge holding the body's dorsal axis
    within ``upright_margin_deg`` of camera up (:meth:`_upright_loss`), which is
    how sharks swim except when banking; its ``"broadside"`` variant instead
    asks the camera to see the flank, which on the demo clip rolls the body
    48.6 deg median off vertical for +1.09 px residual and -0.08 IoU.  Both are
    exactly inert at weight 0.

    ``bend_weight > 0`` plus a per-frame ``bend_series`` handed to :meth:`fit`
    adds a third mask observation, and the only one that is a TIME series: the
    signed chord bend of the 13 projected stations against the same bend
    measured on the mask centreline (:meth:`_bend_loss`).  It is one scalar per
    frame, invariant to where the body sits in the image and how big it is, so
    unlike the midline term it constrains the wave's frequency, phase and sign
    without pulling on the translation.  It runs in whichever stages
    ``bend_stages`` names (both by default) and, being part of :meth:`_loss`
    unconditionally, is part of the coarse warm-start grid and of the converged
    objective the amplitude wells are ranked on.  Inert at weight 0.

    ``width_weight > 0`` plus a per-frame ``width_series`` adds the girth
    analogue of that term, and the only observation in the objective that can
    see how THICK the animal is: the projected body width at each of the 13
    stations against the width the mask measured at the same station
    (:meth:`_width_loss`).  ``width_model_side`` picks how the MODEL's half of
    that comparison is measured -- ``"scanline"`` by default
    (:meth:`station_scanline_widths_px`: the soft silhouette's coverage
    integrated along the station's normal, which IS the chord the mask target
    is read with, measured to agree with a chord through the model's own hard
    render to within 2% at every trunk station on both clips -- a per-station
    MEDIAN over frames, and measured at the ROW-I optimum: across all ten of
    girth round 3's fitted optima the worst trunk station median reaches 2.6%
    (demo) and 3.3% (clip 2), and per FRAME the worst trunk station-frame is
    1.28x (demo) and 1.66x (clip 2) with the chord clear of the frame edge,
    ``girth3/review2/operator_optima.json``), or ``"band"``
    (:meth:`station_widths_px`: the extent of the station band's vertices along
    that normal, no faces and no rasteriser, measured 9-16% off that chord in
    OPPOSITE directions on the two clips, which is why it is no longer the
    default).  It runs
    in whichever stages ``width_stages`` names (``"B"`` by default, where the
    betas are free) and is likewise always part of the converged objective the
    amplitude wells are ranked on.  Which vertices count as body rather than
    fin is not decided here: it is ``SharkSMPL.girth_gate``, the same gate the
    3-D girth field uses, so the thickness this term fits and the thickness a
    fit reports are measured off the same vertices -- a question the scanline
    does not ask at all, because it reads the silhouette rather than a chosen
    set of vertices, exactly as the mask's chord does.  Inert at weight 0,
    which is the shipped default.  Both sides of the ratio are now ONE
    operator: the target is the mask CHORD (``mask_centerline.chord_width``,
    what the offline girth score marches) and the model side integrates the
    same chord over the model's own silhouette.

    ``length_weight > 0`` plus a per-frame ``length_series`` is the companion
    observation: the projected arc length of the 13-station chain
    (:meth:`station_chain_length_px`) against the arc length the mask
    centreline covers over the SAME span (:meth:`_length_loss`).

    WHAT IT WAS MEASURED TO DO, on the 2026-09-05 rows
    (``outputs/demo_2026-09-03/work/girth2/report.md``): it pins the APPARENT
    length -- model/mask 1.0562 -> 1.0005 on clip 2 and 0.9811 -> 0.9994 on the
    demo -- and it does so through TRANSLATION and POSE.  On the v1 and v2
    assets it has no path into the shape at all:
    :class:`~shark_pose.model_3d.shark_smpl.SharkSMPL` reads its rest joints
    from a constant buffer there, so the 3-D 13-station chain is 3.784945 long
    for every beta vector tested and after a backward through
    :meth:`station_chain_length_px` neither ``betas.grad`` nor ``girth.grad``
    exists, while translation and pose carry the whole gradient.  It is
    therefore an apparent-length observation, never a metric-length biometric.

    An asset that declares ``skeleton.json["joint_source"] == "regressor"``
    (v3) regresses the rest joints from the SHAPED vertices instead, and this
    term -- like :meth:`_midline_loss` and :meth:`_bend_loss` -- then does carry
    a beta gradient.  It is still an APPARENT length, so it trades against depth
    exactly as the landmarks do; what it buys is the along-body PARTITION, which
    does not cancel against depth.

    WHAT IT WAS MEASURED NOT TO DO: give back the depth the width term takes.
    Clip 2's median root depth is 5.1863 with no width term, 5.4256 with width
    + girth, and 5.4292 once this term is added -- the depth payment survives
    it untouched, while the width ratio at the fitter's own stations gets
    worse over the same step (0.8775 -> 0.8669).  A constraint on DEPTH is
    still missing.

    It runs in whichever stages ``length_stages`` names (``"B"`` by default;
    ``"B"`` and ``"AB"`` were measured against each other in
    ``outputs/demo_2026-09-03/work/girth2/fix1`` and differ by at most 0.09 px
    of residual and 0.006 of IoU) and is always part of the converged
    objective the amplitude wells are ranked on.  Inert at
    weight 0, the shipped default.

    An optional soft-silhouette term (``silhouette_weight > 0`` plus per-frame
    masks handed to :meth:`fit`) adds ``1 - soft IoU`` between the posed mesh's
    differentiable coverage map and a segmentation mask.  It runs in stage B
    only, on a rotating slice of the masked frames per step, so its cost and
    memory are set by ``silhouette_frames_per_step`` rather than by the clip
    length.  Because the term is part of what the fit minimises it is also part
    of what the wells are compared on: the converged score of every well is
    taken on ONE FIXED slice of masks (the sampler's first slice), so the
    comparison is not contaminated by which frames the rotating sampler happened
    to land on, and ``amp_well_scores["silhouette_term"]`` reports the weighted
    contribution separately.  The term rasterises through the fitter's OWN
    camera -- ``focal_length`` and ``principal_point`` are forwarded to
    ``soft_silhouette`` -- so a calibrated real camera (see
    ``sim2real.camera_calib``) drives the landmarks and the mask through one
    pinhole; only the resolution convention (:data:`BASE_RESOLUTION`) is
    required, and a mismatch is rejected in ``__init__``.
    """
    tracer = None
    trace_terms = None

    def __init__(self, shark_model: SharkSMPL, prior: Optional['SpinePrior']=None, focal_length: float=1000.0, image_size: tuple[int, int]=(256, 256), principal_point: Optional[Tuple[float, float]]=None, stage_a_steps: int=200, stage_b_steps: int=300, gm_sigma: float=40.0, lr: float=0.05, prior_weight: float=1.0, shape_prior_weight: float=1.0, shape_prior_scale: Optional[Sequence[float]]=None, shape_freeze_components: Sequence[int] | str=(), shape_prior_sds: Optional[Sequence[float]]=None, beta_lr_scale: float=0.1, shape_lr_in_sd: bool=False, joint_limit_weight: float=20.0, temporal_weight: float=0.0, root_temporal_weight: Optional[float | Sequence[float]]=None, translation_temporal_weight: Optional[float]=None, translation_temporal_target: str='root', swim_lateral_weight: float=0.0, swim_headfirst_weight: float=0.0, swim_speed_weight: float=0.0, swim_heading_weight: float=0.0, swim_speed_mu_bl_s: float=0.55, swim_speed_sigma_log: float=0.6, swim_heading_cap_deg_s: float=30.0, swim_body_length: float=4.3509, share_clip_params: bool=True, fps: float=30.0, default_amp: Optional[float]=None, amp_grid: Optional[Sequence[float]]=None, amp_restarts: bool=True, freq_init_hz: Optional[float]=None, freq_grid_hz: Optional[Sequence[float]]=None, freq_phase_samples: int=16, prior_overrides: Optional[dict]=None, silhouette_weight: float=0.0, silhouette_res: int=96, silhouette_sigma_px: float=DEFAULT_SIGMA_PROXY, silhouette_face_stride: int=2, silhouette_frames_per_step: int=25, silhouette_sigma_px_final: Optional[float]=None, silhouette_anneal_frac: float=1.0, midline_weight: float=0.0, midline_sigma_px: float=12.0, midline_use_widths: bool=False, midline_chamfer: bool=False, midline_arc_span: Optional[Tuple[float, float]]=None, upright_weight: float=0.0, upright_margin_deg: float=25.0, upright_mode: str='dorsal', bend_weight: float=0.0, bend_sigma_deg: float=4.0, bend_stages: str='AB', width_weight: float=0.0, width_sigma_px: float=3.0, width_stages: str='B', width_stations: object='trunk', width_tau_px: float=WIDTH_TAU_PX, width_fin_rule: str='girth', width_fin_radius_factor: float=WIDTH_FIN_RADIUS_FACTOR, width_model_side: str='scanline', width_section_tau_px: float=WIDTH_SECTION_TAU_PX, width_scanline_sigma_px: float=SCANLINE_SIGMA_PX, width_scanline_step_px: float=SCANLINE_STEP_PX, width_scanline_half_scale: float=WIDTH_SCANLINE_HALF_SCALE, width_scanline_face_stride: int=1, width_score_exclude_stations: Sequence[int]=(), width_align_stations: bool=False, width_align_grad: bool=False, width_normalize: bool=False, width_norm_ref_px: float=100.0, length_weight: float=0.0, length_sigma_px: float=6.0, length_stages: str='B', girth_enabled: bool=False, girth_lr_scale: float=0.5, girth_prior_weight: float=1.0, girth_smooth_weight: float=5.0, girth_tie_weight: float=1.0, girth_init: str='ones', girth_prior_mean: object=None, girth_delay_frac: float=0.0):
        """
        Args:
            shark_model: a :class:`SharkSMPL` loaded from ``data/model_assets_v2``
                (31 joints, ``skeleton.json`` present).
            prior: the swimming prior. Defaults to ``SpinePrior.load()`` reordered
                onto ``shark_model.joint_names``.
            focal_length: the fit's camera focal length, in proxy pixels. The
                default 1000 is the pipeline's fixed pinhole (a 14.6 deg
                horizontal field of view at 256 px), which is what the SPIN
                warm start was trained at; a calibrated camera passes
                ``camera_calib.CameraIntrinsics.proxy_focal()`` instead, and
                the caller must then rescale the warm-start depth by
                ``focal / 1000`` so the body starts at roughly the same image
                size -- exactly the same only for a fronto-parallel body; a
                solid one gains real parallax at the wider camera (1.41x the
                projected extent at the demo geometry, against 0.12x with no
                rescale).
            image_size: (H, W) proxy resolution.
            principal_point: ``(cx, cy)`` in proxy pixels. ``None`` (default) is
                the image centre, which is where the letterbox puts a camera
                whose principal point is the frame centre. An undistorted real
                camera's is not exactly there, so it is passed explicitly --
                otherwise the fit absorbs the offset into the translation.
            stage_a_steps: steps for stage A (global placement: root rotation,
                translation, wave amplitude/phase, turn).
            stage_b_steps: steps for stage B (everything, plus the fins, the free
                per-segment residual and the betas).
            gm_sigma: Geman-McClure sigma, in proxy pixels.
            prior_weight: weight on ``SpinePrior.prior_loss`` (0 at the defaults).
            shape_prior_weight: L2 pull of the betas toward their initial value.
            shape_prior_scale: **TASK F1, default None.**  ``NUM_BETAS``
                non-negative multipliers making that pull per-component, i.e.
                component ``k`` is penalised at
                ``shape_prior_weight * shape_prior_scale[k]``.  ``None`` runs
                the shipped isotropic expression verbatim.  See
                :meth:`_parse_shape_prior_scale`.
            shape_freeze_components: **TASK F1, default ().**  Shape components
                held at ``init_betas`` for the whole fit, by zeroing their
                gradient on the leaf.  An empty sequence registers no hook.
                See :meth:`_parse_shape_freeze`.
            shape_prior_sds: per-mode prior widths, one per beta, in the units
                the shape basis is scaled in. The prior becomes
                ``w * sum_k scale_k * ((beta_k - beta_init_k) / sd_k)^2``, i.e.
                one sd of any mode costs exactly ``w * scale_k``. ``None`` --
                the default, and what the v1/v2 assets get -- leaves the
                isotropic ``w * ||beta - beta_init||^2`` expression untouched,
                bit for bit.  When the model's asset ships a ``shape_modes.json``
                (v3), its sds are picked up automatically; the basis is then in
                NAMED units (one unit = 1% of TL on that mode's published
                length), so an isotropic prior would penalise a mode whose
                literature sd is 4% of TL sixteen times too hard.
            beta_lr_scale: the betas' learning rate as a fraction of ``lr``; shape
                is nearly unobservable from 16 landmarks in one clip, so they are
                deliberately slow.
            shape_lr_in_sd: **TASK F1, default False.**  Take the betas' Adam
                step in units of each mode's own PRIOR SD instead of in the
                basis's raw units, by optimising ``z`` with
                ``beta_k = sd_k * z_k``.

                Why it exists.  Adam's step size is ``lr`` per step almost
                independently of the gradient, so ``lr * beta_lr_scale *
                stage_b_steps`` is a hard CEILING on how far a component can
                travel in one fit -- 0.05 * 0.1 * 300 = **1.5 raw units**.  On
                the v1/v2 PCA basis one unit IS one sd, so that ceiling is 1.5
                sd and nothing notices.  On the v3 named basis a unit is 1 % of
                TL and one sd of ``trunk_share`` is 4.0 of them, so the same
                setting can reach only **0.375 sd** of that mode -- and 0.375
                sd is, to three digits, where every released-prior fit in
                TASK F1's sweep stopped (0.346-0.369) and where the tier-2
                recovery of a 1 sd truth stopped (0.300).  The shape was pinned
                by the OPTIMISER, not by the prior.

                ``False`` leaves the leaf in raw units and is bit-identical;
                ``True`` needs the model's ``shape_modes.json`` (a basis with
                no declared sds has nothing to rescale by and raises).
            joint_limit_weight: weight on the out-of-limit hinge.
            temporal_weight: weight on the temporal coupling (0 disables it).
            root_temporal_weight: absolute weight on the ROOT half of the
                temporal term (the frame-to-frame smoother on the global
                rotation), replacing ``temporal_weight`` for that half alone.
                ``None`` -- the default and every fit made before TASK F --
                leaves the root half coupled to ``temporal_weight`` and runs
                the shipped expression verbatim.  Setting either this or
                ``translation_temporal_weight`` switches the term to the SPLIT
                form, in which the root half is the isotropic **geodesic**
                d1/d2 (``_temporal_halves``) rather than the heading-dependent
                axis-angle one; the manoeuvre rails and the wave keep
                ``temporal_weight``, so a root smoother can be stiffened
                without stiffening the tail.  A split term runs even at
                ``temporal_weight = 0``.

                A **3-sequence** ``(w_x, w_y, w_z)`` weights the three CAMERA
                axes -- ``(pitch, yaw, roll)``, the axes
                ``clip_metric.root_rate_block`` reports -- apart, on the LEFT
                increment ``log(R_{k+1} R_k^T)``, which is the only increment
                whose components lie on camera axes when ``root_orient`` is the
                body->camera rotation
                (``outputs/demo_2026-09-03/work/kinematic/aniso/aniso.md``
                section 1 proves it).  Three EQUAL entries dispatch back into
                the isotropic geodesic path, bit for bit, so a scalar and its
                own triple are the same objective; only a genuinely
                anisotropic triple takes the per-axis branch, whose ``d2`` is
                the second difference of the log coordinates rather than the
                geodesic distance between consecutive increments (they agree to
                O(theta^3), measured 3-7e-05 on these clips).
            translation_temporal_weight: the same, for the translation half.
            translation_temporal_target: what the translation half smooths in
                the split form -- ``"root"`` (today's target, the translation
                leaf), ``"centroid"`` (the posed body's joint centroid, the
                only target whose gradient reaches ``root_orient`` and so the
                only one that charges a rotation/translation swap) or
                ``"both"`` (their mean).  Ignored while the term is coupled.
            swim_lateral_weight: TASK F kinematic prior, **default off**.
                Weight on the non-holonomic term: the squared LATERAL and
                DORSAL components of the 0.5 s low-passed root velocity, in
                (body lengths / s)^2.  The low-passed velocity, not the raw
                one, because the defect it addresses is a drift and the
                frame-rate jitter belongs to the root smoother above
                (``outputs/demo_2026-09-03/work/kinematic/swim/swim.md``).
            swim_headfirst_weight: weight on the one-sided along-body term,
                ``relu(-v_long)^2`` in (BL/s)^2 -- zero while the animal swims
                forwards, so it is a rail against the fit reversing rather than
                a penalty on speed.
            swim_speed_weight: weight on the log-normal cruise-speed prior on
                the low-passed speed, ``(log(|v|/L) - log(mu))^2 / 2 sigma^2``.
            swim_heading_weight: weight on the one-sided heading-rate cap,
                ``relu(|psi_dot| - cap)^2 / cap^2`` on the low-passed heading
                rate about the world up PROXY (camera -y; not gravity).
            swim_speed_mu_bl_s: median of the speed prior, body lengths / s.
            swim_speed_sigma_log: its log sigma (0.6 puts 0.3 and 1.0 BL/s at
                +-1 sigma).
            swim_heading_cap_deg_s: the heading-rate cap, deg/s.
            swim_body_length: the ruler the swimming terms are expressed in,
                model units.  The default 4.3509 is the 31-joint rest span, so
                a weight transfers between clips of different animals.
            share_clip_params: see the class docstring.
            fps: frame rate used to build the clip's time base.
            default_amp: cruise amplitude the leaves are created at. Only
                decides the fit when the coarse search is off (an empty
                ``amp_grid``, or ``share_clip_params=False``); otherwise the
                grid overwrites it. Defaults to the prior's swimming mode.
            amp_grid: candidate cruise amplitudes for the coarse init search
                (clip mode only). ``None`` uses :data:`DEFAULT_AMP_GRID`; an
                empty sequence disables the amplitude axis and leaves the fit
                starting at ``default_amp``.
            amp_restarts: re-run both stages from the best coarse start in each
                well of the mixture amplitude prior (glide and swimming) and
                keep the lower converged objective. Costs one extra fit and is
                the only thing that makes the amplitude data-selected rather
                than init-selected, because Adam cannot cross the barrier
                between the wells. Ignored when the coarse search is off.
            freq_init_hz: initial tail-beat frequency. Defaults to the prior's
                mode (0.45 Hz), not the sevengill animation's 0.9 Hz.
            freq_grid_hz: candidate frequencies for the coarse init search run
                before stage A (clip mode only). ``None`` uses
                :data:`DEFAULT_FREQ_GRID_HZ`; an empty sequence disables it.
            freq_phase_samples: phases tried per grid candidate (1-D search).
            prior_overrides: amplitude / frequency prior shape, forwarded to
                :meth:`SpinePrior.configure_priors`.
            silhouette_weight: weight on the soft-silhouette term (0 disables it
                entirely -- no mask is even looked at).  The term is
                ``1 - soft IoU`` between the posed mesh's differentiable
                coverage map and a segmentation mask, averaged over the frames
                scored in that step, and it is added in stage B only: it is a
                shape/limb-placement cue, useless before the body is roughly
                where the landmarks say it is.
            silhouette_res: square resolution the coverage map is rasterised at.
                The masks handed to :meth:`fit` must already be letterboxed to
                it (``proxy_geometry.letterbox_image``).
            silhouette_sigma_px: rasteriser softness in PROXY pixels, scaled to
                the raster by :func:`~shark_pose.losses.soft_silhouette.sigma_at`
                so the soft coverage area tracks the hard silhouette's at every
                resolution instead of only at one.  A fixed raster sigma of 1
                covers 2.12x the hard area at res 64 and 1.47x at res 128, and
                a term aiming at twice the animal's area inflates the animal.
            silhouette_face_stride: keep every n-th face when rasterising.  2
                halves the cost; a decimated closed mesh shrinks slightly, which
                a silhouette *term* tolerates (it is a pull, not a measurement)
                but a reported IoU does not -- score with ``hard_silhouette``.
            silhouette_frames_per_step: how many masked frames are scored per
                optimiser step.  Every step sees a different slice, cycling
                through the masked frames, so the whole clip is covered over a
                stage while peak memory stays that of one slice.
            silhouette_sigma_px_final: TASK F1 -- the ROLL observation.
                ``None`` (the default) keeps ``silhouette_sigma_px`` fixed for
                the whole fit, which is bit-for-bit what every fit before
                2026-09-08 did.  A positive value anneals the rasteriser's
                softness GEOMETRICALLY from ``silhouette_sigma_px`` down to it
                over the first ``silhouette_anneal_frac`` of stage B, then
                holds.  Why it exists, measured on the synthetic roll-40 view
                (``outputs/demo_2026-09-03/work/roll/roll_jacobian/``): the
                mask carries the roll -- hard IoU is 0.990 at the rendered
                39.6 deg and 0.85 at 20 deg of error -- but at the shipped
                0.75 proxy-px blur the SOFT term's roll minimum sits 15 deg off
                truth and its curvature there is NEGATIVE (-2.3e-05), i.e. the
                truth is a local maximum.  At 0.2 the minimum is exactly at
                truth, the curvature is +4.45e-04 and ``g_z`` absorbs 35 % of
                it instead of more than all of it.  Resolution is NOT the
                lever (96 / 192 / 384 / 768 give the same landscape to within
                0.005): sharpness is.  The anneal is not decoration -- a
                0.2 px blur has a gradient reach of a fraction of a raster
                pixel, so a fit that starts there sees nothing.
            silhouette_anneal_frac: fraction of stage B the anneal above is
                spread over (1.0 = the whole of it).  Ignored while
                ``silhouette_sigma_px_final`` is ``None``.
            midline_weight: weight on the projected-midline term (0 disables it
                and no centreline is even looked at).  The term compares the 13
                projected SPINE STATIONS against a mask centreline handed to
                :meth:`fit`; see :meth:`_midline_loss`.  It runs in stage A and
                stage B both -- 13 points per frame is cheaper than the landmark
                term, and the bend plane is a global-placement question, so it
                has to be live while the root is being placed.
            midline_sigma_px: Geman-McClure sigma of the midline term, in proxy
                pixels.  Smaller than ``gm_sigma`` because a centreline is a
                geometric construction, not a detector output: the distances it
                should tolerate are extraction bias, not missed keypoints.
            midline_use_widths: weight each station by the centreline's local
                half-width (normalised to mean 1, clamped to [0.25, 4]), so the
                thick trunk -- where the slice-mean midline is well determined
                -- counts for more than the thin peduncle.  Off by default: the
                widths are an extraction by-product and the weighting has not
                been shown to help.
            midline_chamfer: replace the ordered arc-length correspondence with
                the nearest point on the centreline polyline.  A fallback for
                centrelines whose ENDS are unreliable (a truncated mask), at the
                cost of being blind to sliding along the body -- an ordered
                correspondence is what carries the bend.  Off by default.
            midline_arc_span: ``(head_frac, tail_frac)``, where the cranium and
                the last caudal station sit along the MASK centreline's arc
                length.  The centreline runs snout tip -> caudal tip while the
                stations run cranium -> caudal_axis_2, so the station fractions
                have to be mapped into that span.  ``None`` measures it from the
                rig's own rest geometry (see :meth:`_default_arc_span`).
            upright_weight: weight on the upright-body prior (0 disables it).
                Sharks swim dorsal-fin-up and bank only in turns, so the body's
                vertical axis is penalised once it leaves a cone around the
                camera's up direction; the penalty is a smooth hinge, exactly
                zero and zero-gradient inside the cone.
            upright_margin_deg: half-angle of that cone, in degrees.  25 deg
                leaves ordinary banking free.
            upright_mode: ``"dorsal"`` (default) hinges the body's dorsal axis
                against camera up.  ``"broadside"`` instead hinges the body's
                LATERAL axis out of the image plane, i.e. it asks the camera to
                see the flank; it is a much stronger claim about the camera and
                is not on by default.  Measured on the demo clip (weight 50):
                it rolls the body 48.6 deg median off vertical and costs
                +1.09 px residual and -0.08 hard mesh-vs-SAM IoU, i.e. it buys
                a broadside view by breaking the upright one.
            bend_weight: weight on the bend-series term (0 disables it and no
                bend series is even looked at).  See :meth:`_bend_loss`: the
                projected stations' own chord bend against the mask
                centreline's, per frame, in degrees.
            bend_sigma_deg: Geman-McClure sigma of that term, in degrees.  The
                mask's STATION bend on the demo clip is 3.63 deg median with a
                30.7 deg peak-to-peak swing (-8.2 to +22.5 deg), so a few
                degrees is the scale of "the same beat, slightly differently
                measured".
            bend_stages: which stages the term runs in -- ``"A"``, ``"B"`` or
                ``"AB"`` (default).  Timing is a global-placement question, so
                stage A is where it earns its keep; the well-selection score is
                always taken with the term on, whatever this says, because it
                is part of the objective the fit minimises.
            width_weight: weight on the width-series term (0 disables it and no
                width target is even looked at).  See :meth:`_width_loss` and
                :meth:`station_widths_px`: the projected body width at each of
                the 13 spine stations against the width the mask measured
                there.  This is the observation that can see GIRTH; nothing
                else in the objective can.
            width_sigma_px: Geman-McClure sigma of that term, in proxy pixels.
            width_stages: which stages it runs in -- ``"A"``, ``"B"`` (default)
                or ``"AB"``.  Girth is a shape question and the betas are only
                free in stage B, so B is the default; the well-selection score
                always includes the term, as with the bend series.
            width_stations: which stations are compared -- ``"trunk"``
                (default: stations 3-9, the seven vertebral stations between
                the branchial region and the precaudal one), ``"all"`` or an
                explicit sequence of station indices in ``[0, 13)``.  The
                default follows the config, and the config follows the
                evidence: at stations 10-11 this term's own measurement and the
                chord that scores it disagree by ~30%.
            width_tau_px: temperature of the soft extent, in proxy pixels
                (:func:`_soft_extent`).
            width_fin_rule: how fin vertices are kept out of a station's band
                -- ``"girth"`` (default: G1's own gate, see
                :data:`WIDTH_FIN_RULES`) or ``"radius_median"``.  ``"girth"``
                falls back to ``"radius_median"`` on a model that carries no
                girth operator, and says so in
                :attr:`width_fin_rule_used`.
            length_weight: weight on the length-series term (0 disables it and
                no length target is even looked at).  See :meth:`_length_loss`
                and :meth:`station_chain_length_px`: the projected arc length
                of the 13-station chain against the arc length the mask
                centreline covers over the SAME span.  The one observation that
                is about how LONG the animal looks, which is what the width
                series trades against when it is left alone (measured: clip 2
                grew 4-7% in length while thinning).
            length_sigma_px: Geman-McClure sigma of that term, in proxy pixels.
            length_stages: which stages it runs in -- ``"A"``, ``"B"``
                (default) or ``"AB"``; the well-selection score always includes
                the term, as with the bend and width series.
            girth_enabled: give the clip a shared (K, 2) GIRTH leaf -- the
                per-station cross-section scales of
                :meth:`~shark_pose.model_3d.SharkSMPL.apply_girth` -- and
                optimise it in stage B.  ``False`` (the default) passes
                ``girth=None`` to every model forward, so the fit is
                bit-identical to one built before the leaf existed.  Requires a
                model with a girth operator (``data/model_assets_v2``).
            girth_lr_scale: the girth leaf's learning rate as a fraction of
                ``lr``.  Girth is shape, and shape is a clip-wide constant read
                off a noisy per-frame outline, so it is stepped more slowly than
                the pose -- but not as slowly as the betas
                (``beta_lr_scale``), which are effectively frozen.
            girth_prior_weight: weight of
                :func:`~shark_pose.model_3d.girth_prior_loss`, the pull toward
                the unscaled body.  The girth field has no data of its own
                beyond the width series and the silhouette, so without a prior
                a station the mask says nothing about would drift.
            girth_tie_weight: weight of the isotropy tie
                ``mean((g_x - g_z)^2)``, which pulls the LATERAL girth axis
                toward the DORSOVENTRAL one -- SYMMETRICALLY, so it moves the
                observed axis too: measured over rows ii (tie 1.0) and iv (tie
                0.0), ``g_z`` shifts +0.4-1.8% per station on the demo and -2.2
                to +7.2% on clip 2 (see :meth:`_girth_tie_loss`).  Measured, not
                assumed: on these
                broadside clips the image-plane width IS the dorsoventral axis
                (measured on the shipped fits, the width term is 5.95x/3.13x
                more sensitive to g_z than to g_x), so g_x has almost no
                observation of its own and wandered 0.74-1.55 between rows
                while g_z reproduced to +/-3%.  The tie makes the fit say
                "as thick across as it is deep unless the data show otherwise"
                instead of reporting an unobserved number as shape.  Inert
                whenever the girth leaf is off.
            girth_smooth_weight: weight of that prior's along-body
                first-difference term, relative to its deviation term.  A real
                shark's cross-section varies smoothly with arc length; 5 makes
                a kink between neighbouring stations five times as expensive as
                the same total deviation spread evenly.
            girth_init: ``"ones"`` (the identity field), ``"prior_mean"``
                (TASK F1: start at ``girth_prior_mean``) or ``"from_targets"``
                -- initialise from the mask's own widths, by measuring the warm
                start's station widths, taking mask/model per station and
                resampling them onto the girth stations with
                :func:`~shark_pose.model_3d.girth_from_width_ratio`.  Falls
                back to ones (with a warning) when the fit has no width series.
            girth_prior_mean: **TASK F1, default None.**  Where
                :func:`~shark_pose.model_3d.girth_prior_loss` pulls the field
                TO -- a scalar, a ``(mu_x, mu_z)`` axis pair or a full (K, 2)
                field.  ``None`` is the unscaled template (1.0), the shipped
                behaviour.  It also rescales the isotropy tie (see
                :meth:`_girth_tie_loss`).  See :meth:`_parse_girth_prior_mean`.
            width_fin_radius_factor: radius factor for the ``"radius_median"``
                rule (:data:`WIDTH_FIN_RADIUS_FACTOR`).
            width_model_side: which operator measures the MODEL's width --
                ``"scanline"`` (default) or ``"band"``; see
                :data:`WIDTH_MODEL_SIDES` and
                :meth:`station_scanline_widths_px`.  This is the half of the
                ratio the previous round left disagreeing with its own score.
            width_scanline_sigma_px: softness of the scanline, in proxy pixels.
                NOT the rasteriser's ``silhouette_sigma_px``: a scanline
                integral crosses the soft rim twice and pays its whole outward
                bias, where an area integral does not, and the default
                (:data:`~shark_pose.losses.soft_silhouette.SCANLINE_SIGMA_PX`)
                is the largest value measured to keep every trunk station
                within 3% of a chord through the model's own hard render on
                both clips.  Pass ``silhouette_sigma_px`` to tie the two
                together and read 11-26% wide.
            width_scanline_step_px: sample spacing along the line, proxy pixels.
            width_scanline_half_scale: how far the line reaches either side of
                the station, as a multiple of the expected width there
                (:data:`WIDTH_SCANLINE_HALF_SCALE`).
            width_scanline_face_stride: keep every n-th face when scanning.
                Decimating a closed mesh shrinks its silhouette, so this is for
                speed only and 1 is the honest setting.
            width_align_stations: **TASK F1, default False.**  Read the mask's
                width target at the arc fraction the MODEL's own projected
                station falls on, instead of at the record's
                ``station_arc_fractions``.  Needs ``width_polyline`` /
                ``width_profile`` in :meth:`fit`; see :meth:`_width_align_data`
                for what it fixes and what it was measured to be worth.
            width_align_grad: **TASK F1, default False.**  Let the ALIGNMENT
                carry gradient: the mask chord is read at the model station's
                own arc fraction WITH the graph attached, so a shape mode that
                slides a station along the body is charged for the width it
                slides onto.  The two discrete choices (polyline segment,
                profile interval) stay under ``no_grad`` -- there is no
                gradient through an ``argmin``.  ``False`` is the shipped
                correspondence-only read, bit for bit.
            width_normalize: **S6 lane wl_term, default False.**  Compare
                LENGTH-NORMALISED widths instead of absolute proxy pixels: the
                residual becomes
                ``ref * (W_model / L_model - W_mask / L_mask)``, where each
                length is the SAME frame's projected trunk span -- the model's
                station chain (:meth:`station_chain_length_px`, cut by the same
                ``length_seg_weights`` the length series uses) and the mask's
                own centreline arc between the first and last station fraction
                (``fit_video.record_span_length``).  Needs
                ``width_length_series`` in :meth:`fit`.

                WHY.  The shipped residual is a claim about the IMAGE, so a
                frame 10% further away, or a body 10% foreshortened, is 10%
                narrower on screen and the term accepts that as a thinner
                shark; measured on 16 completed pass pairs, the fitted girth
                follows the mask's width/length aspect with slope 0.28
                (``outputs/demo_2026-09-03/work/individuals/s5/girth_scale``).
                Dividing both sides by their own in-frame length cancels any
                factor that scales the whole image equally -- depth, focal,
                letterbox -- so what is left is a SHAPE ratio.  It does NOT
                cancel pitch: a length-only foreshortening still moves the
                ratio, which is the trade this flag makes (depth ambiguity out,
                pitch ambiguity in).  Keep ``length_series.weight > 0`` with it:
                the ratio can be paid by growing longer as readily as by
                growing thinner, and only the length term charges for that.
            width_norm_ref_px: the reference length, in proxy pixels, the
                dimensionless W/L discrepancy is multiplied by so that
                ``width_sigma_px`` keeps a pixel meaning -- the residual is
                "the width error this ratio would make on a body
                ``width_norm_ref_px`` long".  A CONSTANT, never read off the
                data, so the residual is invariant to scaling every image
                coordinate.  Ignored when ``width_normalize`` is False.
            girth_delay_frac: **TASK F1, default 0.0.**  Fraction of stage B
                during which the girth field is held at its init while shape
                is free -- the girth-vs-shape disambiguation rule.  Girth and
                shape are not independently observable from a width.  Measured
                width Jacobian norms, px per unit
                (``shape_observe/grad/audit_{demo48,clip2_48}_v3.json``): demo
                ``g_x`` 441.1 and ``g_z`` 20.5 against 0.030-4.51 for the
                eight named modes (98x-14500x and 4.6x-677x); clip 2 ``g_x``
                236.1 and ``g_z`` 73.7 against 0.034-3.15 (75x-7030x,
                23x-2190x).  Cosine up to 0.81 against ``g_x``.  So with both
                leaves free from step one the width error is always paid by
                girth.  0.0 runs stage B as the single Adam it always was.
            width_score_exclude_stations: station indices dropped from the
                REPORTED width agreement and from nothing else -- the objective
                still compares whatever ``width_stations`` selects.  It exists
                because clip 2's head stations read 0.31-0.73 of the mask chord
                in every row of the previous round: the rostrum is short and
                the fin roots are misplaced in the MESH, so scoring girth there
                measures the mesh, not the girth.
        """
        ...

    def _default_arc_span(self) -> tuple[float, float]:
        """Where the first and last spine station sit along the body's length.

        The mask centreline spans the whole animal, snout tip to caudal tip; the
        stations stop short at both ends (the cranium sits behind the snout, and
        the caudal fin trails past ``caudal_axis_2``).  Measured on the rest
        template along the model's longitudinal axis (+y = snout), which is what
        an arc-length fraction reduces to for a straight body.  It is a property
        of the TEMPLATE, so no number is quoted here: the shipped
        ``data/model_assets_v2`` gives 0.078039 and 0.885481, its pre-reshape
        predecessor gave 0.048889 and 0.881144, and
        :func:`shark_pose.sim2real.mask_centerline.measure_rig_arc_span` is the
        same recipe applied to an asset directory instead of a live model.
        Falls back to the full span when the model carries no template vertices
        (a hand-made rig in a test).
        """
        ...

    def _station_arc_fractions(self, arc_span: Optional[Tuple[float, float]]) -> Tensor:
        """(13,) arc-length fractions of the stations along the MASK centreline.

        The prior's ``spine_fraction`` places the stations along the SPINE
        (cranium = 0, caudal_axis_2 = 1); the rest-pose cumulative bone length
        is the fallback when the asset has no such table.  Those fractions are
        then mapped affinely into ``arc_span`` -- the piece of the centreline the
        spine actually covers.
        """
        ...

    @staticmethod
    def _arc_interp_weights(polyline: Tensor, fractions: Tensor):
        """Bracketing indices and blend weights for arc-length fractions.

        ``polyline`` is (M, P, 2); ``fractions`` is (K,) in [0, 1].  Returns
        ``(lo, hi, w)`` each (M, K), so the same correspondence can be applied
        to the points and to the per-point widths.
        """
        ...

    def _midline_data(self, centerlines: Optional[Tensor], centerline_widths: Optional[Tensor], centerline_frames: Optional[Sequence[int]], n_frames: int, device, dtype):
        """Validate the centrelines and pre-compute the station correspondences.

        The targets are constants, so the arc-length resampling happens once per
        :meth:`fit` rather than once per optimiser step.  Returns
        ``(frame_idx, targets, weights, polyline_or_None)`` or ``None`` when the
        term is off or there is nothing to compare against.
        """
        ...

    def _midline_loss(self, smpl_out, midline) -> Tensor:
        """Robust distance from the projected spine stations to the centreline.

        The 16 observation landmarks sit close to the midline and near the head,
        so they pin the body's AXIS but barely constrain roll about it -- which
        is what decides how much of the tail's lateral sweep survives projection.
        A mask centreline is a whole-body curve: matching the 13 stations to it
        in order makes the fitted bend answer to the image.

        Correspondence is by arc length (station fraction -> centreline
        fraction, see :meth:`_station_arc_fractions`), which is what carries the
        bend; ``midline_chamfer`` swaps it for nearest-point, which does not.
        """
        ...

    def _bend_data(self, bend_series: Optional[Sequence[float]], bend_frames: Optional[Sequence[int]], n_frames: int, device, dtype):
        """Validate the per-frame mask bend series.  ``None`` when the term is off.

        Returns ``(frame_idx, target_deg)``; both are constants, so nothing here
        has to be repeated per optimiser step.
        """
        ...

    def _bend_loss(self, smpl_out, bend) -> Tensor:
        """Robust error between the fitted and the mask's chord bend, per frame.

        The midline term above matches the stations to the centreline's
        POSITION, which the root translation and depth can pay for; this one
        compares a single shape number per frame -- the signed angle between the
        head chord and the tail chord of the projected station chain
        (:func:`projected_station_bend_deg`, numerically the mask's own
        ``bend_of_projected_stations``) -- so it can only be paid for by bending
        the body at the right time and in the right direction.  That is exactly
        the quantity the demo clip disagrees on: over the 5 s clip the mask's
        projected station bend completes one beat and the landmark-only fit's
        two -- 0.200 Hz against 0.400 Hz, both measured on the PROJECTED series
        by timing/p2_metrics.py's FFT (that fit's fitted 3-D wave parameter is
        0.364 Hz, which the data do not identify and which is not what is
        compared here) -- and at matched frequency the two series are
        anti-phase.

        The two depth branches of that sweep -- tail toward the camera vs away
        -- are NOT invisible to the rest of the objective, and an earlier
        version of this docstring said they were.  Measured on the demo clip
        (timing/probe_freq_sign.json: the wave frozen at phase* and at
        phase* + pi, with the WAVE frequency pinned -- 0.2 Hz and 0.364 Hz
        there are pinned 3-D wave parameters, not projected beats): the mirror
        costs +0.672 px of landmark residual and -0.037 median hard IoU at the
        0.2 Hz pin, +0.589 px and -0.035 at the 0.364 Hz pin -- 2-3.5x the
        entire spread either cue shows across the whole 0.15-0.6 Hz scan.  So both cues rank the branches, weakly but
        consistently.  What this term adds is that they rank them the WRONG way
        round against the mask: at 0.2 Hz the branch the landmarks and IoU
        prefer correlates -0.30 with the mask centreline's own station bend and
        the mirror +0.41.  The term makes the image-plane curvature SIGN an
        explicit observation instead of a by-product, which is what settles the
        disagreement; the shipped fit is not the mirrored branch (at the
        shipped defaults it pays +0.179 px and -0.004 IoU against the
        landmark-only baseline, not +0.6 px / -0.035).

        One scalar per frame is also what the per-frame ``turn`` can produce,
        so the term does not identify the WAVE on its own: with the coarse
        grid switched off and the landmarks silent, a fit handed a 0.2 Hz bend
        series reproduces it to 0.8 deg while leaving the frequency at the
        prior's 0.45 Hz and paying for the shape with ``turn`` (measured,
        2026-09-04).  What identifies the beat is this term inside the coarse
        (amplitude x frequency x phase) grid, where only the wave varies: the
        same clip is then recovered to within 8% of 0.2 Hz, in phase rather
        than mirrored, from a grid that picks 0.4 Hz without it.
        """
        ...

    def _resolve_width_stations(self, spec) -> Tensor:
        """(S,) station indices the width term compares, from the config spec.

        ``"all"`` is every station; ``"trunk"`` is 3-9 inclusive -- the seven
        vertebral stations between the branchial region and the precaudal one,
        i.e. the part of the animal whose silhouette width is body wall rather
        than fin root or caudal keel.  Anything else must be a sequence of
        integer station indices.
        """
        ...

    @staticmethod
    def _parse_shape_prior_scale(spec) -> Optional[Tensor]:
        """(10,) per-component multiplier on the shape prior, or ``None``.

        TASK F1.  ``None`` -- the default -- means the prior is the shipped
        isotropic ``w * ||beta - beta_init||^2`` and is evaluated by the
        original expression, so nothing about a fit that leaves this alone
        changes by a bit.

        A sequence of ``NUM_BETAS`` non-negative numbers scales the prior
        component by component, i.e. component ``k`` is penalised at
        ``shape_prior_weight * scale[k]``.  It exists because the shipped
        basis is not one family of modes: components 0-3 are structured
        (affine R^2 0.88-0.999 -- a length mode, a dorsoventral depth mode and
        two cross-section/fin modes) while 4-9 are the generating script's
        isotropic ``randn`` floor (affine R^2 <= 0.038, <= 0.54% of any
        anatomy per unit).  A single scalar weight therefore cannot relax the
        prior on the modes that mean something without relaxing it on the
        noise, where 97.6% of the norm goes the moment it is loosened
        (``outputs/demo_2026-09-03/work/identify/shape/report.md`` sec. 3).
        """
        ...

    @staticmethod
    def _parse_shape_freeze(spec, mode_names: Optional[Sequence[str]]=None) -> Tuple[int, ...]:
        """Sorted, de-duplicated shape components held at their init value.

        TASK F1.  An empty spec -- the default -- registers no hook at all, so
        the fit is bit-identical to one built before this existed.

        Freezing is done on the LEAF, by zeroing those entries of the betas'
        gradient, so it applies to every term at once (the data term, the
        prior, and anything added later) and Adam never accumulates a moment
        for a frozen component.  It is not the same as an infinite prior
        weight, which would still let a component move while the optimiser
        fought it.

        ``"auto"`` resolves the list AGAINST THE ASSET instead of hard-coding
        it, which is the whole point: the shipped literal ``[4, 5, 6, 7, 8, 9]``
        was calibrated on the v2 basis, where 4-9 are the generating script's
        isotropic randn floor -- and on the v3 anatomical basis those same six
        indices are ``dorsal_position``, ``dorsal_height``, ``caudal_span``,
        ``pectoral_length`` and the two inert pads, so the literal silently
        froze FOUR NAMED MODES at zero.  Under ``"auto"`` the rule is the one
        the literal was meant to express -- freeze what carries no named
        anatomy:

        * a basis that declares its modes (``shape_modes.json``, v3) freezes
          the components past the last named mode and any whose name marks it
          inert, i.e. exactly the pads;
        * a basis that declares nothing (v1/v2) freezes
          :data:`SHAPE_NOISE_FLOOR`, which reproduces the shipped literal bit
          for bit, so a v2-asset fit written ``"auto"`` is the fit that
          shipped.

        A LITERAL list is still accepted -- and still does exactly what it
        says -- but when it lands on a mode the asset has NAMED (and not
        marked inert) this warns, because that is the silent failure above:
        the shipped default ``[4, 5, 6, 7, 8, 9]`` on ``model_assets_v3``
        freezes ``dorsal_position``, ``dorsal_height``, ``caudal_span`` and
        ``pectoral_length`` -- half the named basis -- and the only other
        signal is ``summary.shape.freeze_resolved`` after the fit has run.
        """
        ...

    def _parse_girth_prior_mean(self, spec) -> Optional[Tensor]:
        """(K, 2) girth prior mean, or ``None`` for the shipped 1.0.

        TASK F1.  Accepts a scalar, a ``(2,)`` axis pair ``(mu_x, mu_z)``, or a
        full ``(K, 2)`` field, and always returns the broadcast ``(K, 2)`` form
        so the value a fit used can be written out verbatim.  ``None`` keeps
        :func:`~shark_pose.model_3d.girth_prior_loss` on its original
        expression.

        WHAT IT IS FOR.  The girth field's only observation is the width
        series, and on both reference clips that series sees ONE of the two
        axes: the image-plane extent is the dorsoventral one on a broadside
        view, and the lateral axis points at the camera (measured, d(width)/dg
        ratios 9.78x on the demo and 4.81x on clip 2).  The unobserved axis
        goes wherever its prior puts it, so the prior's mean is not a
        formality -- it IS the reported lateral cross-section.  Leaving it at
        1.0 reports the TEMPLATE, whose girth/TL is 0.5154 on the demo against
        a published 0.55-0.62: 6.3% below the band's floor.

        That deficit is measured under the SETTLED cross-section construction
        (D4's exact plane slice, D4b's convex-hull area rule).  An earlier
        "~14% thinner" and the pair ``[1.0735, 1.267]`` derived from it came
        off the superseded vertex-binned envelope and are VOID; the re-derived
        pair is ``[1.0573, 1.2301]``, and
        ``outputs/.../identify/girth/prior.json`` computes it from the shipped
        morphometrics records rather than from a copied constant.
        """
        ...

    def _resolve_score_exclusions(self, spec) -> Tensor:
        """(E,) station indices excluded from the REPORTED width agreement.

        Score-only: nothing here reaches the objective.  A station lands in
        here when the two sides of the ratio stop describing the same anatomy
        -- clip 2's head stations 0-2, where the model's rostrum is short and
        the mask's chord crosses a snout the mesh does not have, read 0.31-0.73
        in every row of girth round 2 no matter what the girth field did.
        Reporting those as a girth error would be reporting the mesh.
        """
        ...

    def _station_axis_px(self, joints: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """``(points, tangent, normal)`` of the projected station chain.

        ``joints`` is (M, J, 3) in the camera frame.  The tangent is the
        central difference ``station[k+1] - station[k-1]`` (one-sided at the
        ends) of the PROJECTED chain and the normal is that turned a quarter
        turn -- the very rule
        :func:`~shark_pose.sim2real.mask_centerline.chord_widths_along_polyline`
        applies to the mask's own polyline, so a width taken here and a width
        taken there are taken along the same line.
        """
        ...

    def station_scanline_widths_px(self, vertices: Tensor, joints: Tensor, stations: Optional[Tensor]=None, half_len_px: Optional[Tensor]=None) -> Tensor:
        """(M, S) chord width of the model's silhouette at each station.

        ``vertices`` is (M, N, 3) and ``joints`` (M, J, 3), both in the CAMERA
        frame; the result is in proxy pixels and differentiable in both.
        ``stations`` selects which of the 13 stations are measured (``None`` =
        all of them); ``half_len_px`` is (M, S) and says how far each line
        reaches, defaulting to :attr:`width_scanline_half_scale` times the band
        operator's own extent -- which costs that operator, so the fitting path
        passes the MASK's width instead and pays nothing.

        THE OPERATOR.  The station point and the image-plane normal to the
        projected spine are the same ones :meth:`station_widths_px` uses (
        :meth:`_station_axis_px`).  What changes is what is measured along that
        normal: instead of the extent of a hand-picked set of band vertices,
        this integrates the soft silhouette's coverage --
        :func:`~shark_pose.losses.soft_silhouette.soft_scanline_width`, the same
        per-face signed distance, sigma and probabilistic union the rasteriser
        uses -- over the contiguous run containing the station point.  For a
        convex cross-section that integral IS the chord length, so this and
        ``mask_centerline.chord_width`` are one operator evaluated on two
        silhouettes, which is the whole point: the previous round's model side
        and its score were different functions and disagreed by 9-16% in
        opposite directions on the two clips, more than the girth effect being
        fitted.

        WHY IT IS AFFORDABLE.  Nothing is rasterised.  A face takes part only
        when it lies across one of the S lines, and is then evaluated only in
        its own window of samples, so the cost is the mesh near 7 lines rather
        than the mesh times a frame.  Measured on MPS at 300 frames, 7 trunk
        stations, all 5628 faces: 0.263 s per forward+backward against 0.085 s
        for the band operator and 0.039 s for the mesh alone
        (``outputs/demo_2026-09-03/work/girth3/stepcost_mps_300f.json``; the
        body is placed at the on-screen width the demo's own fit has, because
        the cost scales with the line's length in pixels).

        ACCEPTANCE.  Against ``chord_width`` marched over the model's own hard
        render at the same points, on the shipped from-scratch fits of both
        clips (row i, 32 frames each), the MEDIAN over frames at every trunk
        station 3-9 agrees to within 1.9% (demo) and 1.6% (clip 2) --
        ``outputs/demo_2026-09-03/work/girth3/operator_accept.json``.  Those two
        figures belong to ROW I: re-measured on all ten of this round's fitted
        optima at 20 frames, the worst trunk station median runs 1.6-2.6% (demo)
        and 1.5-3.3% (clip 2), so a live-girth optimum can sit outside them
        (``girth3/review2/operator_optima.json``).  The band
        operator on the same fits, frames and points is out by 20.0% and 12.3%
        at its worst trunk station.  Per FRAME the agreement is much looser --
        over the ten optima the worst trunk station-frame is 1.03-1.28x (demo)
        and 1.22-1.66x (clip 2) with the chord clear of the frame edge, and up
        to 5.6x counting stations the camera has cropped (26-34 of 260
        station-frames on clip 2, none on the demo;
        ``.../girth3/review2/operator_optima.json``,
        ``.../girth3/fix1/operator_frames.json``) -- so this is a fit-level
        operator match, not a per-frame one.
        """
        ...

    def _station_band_vertices(self) -> Tuple[Tensor, Tensor]:
        """``(index (K, B), mask (K, B))`` -- the template vertices of each band.

        Computed ONCE, from the REST template, and then constant for the whole
        fit: which vertices a station's width is measured from is an anatomical
        fact about the mesh, not something the pose is allowed to move.

        Two rules, in order:

        1. ALONG-BODY BAND.  Every vertex carries an arc fraction along the rest
           spine (``SharkSMPL.girth_fraction``, the nearest point on the station
           polyline) and every station carries one too (the prior's
           ``spine_fraction``).  A vertex joins station ``k`` when its fraction
           is within half the local station SPACING of the station's -- half the
           mean of the gaps to its two neighbours, one-sided at the ends -- so
           each band is the slab around its station.  Because the rig's station
           spacing is non-uniform the bands overlap slightly and leave gaps:
           measured on data/model_assets_v2, 524 vertices land in two bands and
           150 body vertices (girth_gate > 0) in none, which is why the 13 bands
           hold 4082 entries over 3912 vertices.  A model with no girth operator falls
           back to the model frame's ``+y`` (the longitudinal axis, snout
           positive), which is what an arc fraction reduces to for a straight
           rest body.
        2. FIN EXCLUSION, by :attr:`width_fin_rule` -- see
           :data:`WIDTH_FIN_RULES`.  ``"girth"`` keeps the vertices G1's own
           gate keeps (``SharkSMPL.girth_gate > 0``, i.e. inside
           ``GIRTH_GATE_END`` times the local body radius), so the 3-D girth and
           the 2-D width are measured off the same body.  ``"radius_median"`` is
           the fallback described in :data:`WIDTH_FIN_RADIUS_FACTOR`.

        The result is padded to the largest band (index 0, masked False) so the
        term stays one batched tensor op; ``B`` is 1016 on
        ``data/model_assets_v2`` (the trunk_01 band, which spans the whole
        posterior head) against 4082 real entries in 13 bands, and that padded
        gather is what the per-step cost is paid on -- 0.044 s per step at 128
        frames and 0.101 s at 300, on top of the 0.022 / 0.044 s the mesh
        forward and backward already cost
        (``outputs/demo_2026-09-03/work/girth/stepcost.json``).  The soft
        silhouette, which reads the same vertices, costs 0.73 s per step at
        res 96 / 25 masked frames / face_stride 2 on MPS
        (``girth/sil_stepcost_mps_300f_25m_res96_s2.json``) against 0.10 s for
        this term: the difference is the rasteriser.
        """
        ...

    def station_widths_px(self, vertices: Tensor, joints: Tensor) -> Tensor:
        """(M, K) projected body width at every spine station, in proxy pixels.

        ``vertices`` is (M, N, 3) and ``joints`` (M, J, 3), both in the CAMERA
        frame -- i.e. straight out of :class:`SharkSMPL`.  Differentiable in
        both, and rasteriser-free: there is no image, so the term costs the
        gather and the projection and nothing else.

        The measurement, per frame and station ``k``:

        * project the station's band vertices (:meth:`_station_band_vertices`)
          and the 13 stations themselves through the fitter's own camera;
        * take the image-plane direction of the local spine, ``station[k+1] -
          station[k-1]`` (one-sided at the ends), and its normal;
        * project each band vertex's offset from ``station[k]`` onto that
          normal, and take the soft extent -- max minus min
          (:func:`_soft_extent`) -- of those signed offsets.

        HOW THIS COMPARES WITH THE OPERATOR THAT SCORES GIRTH.  The reference
        is the chord of ``outputs/demo_2026-09-03/work/shape_head/e2_score.py``:
        render the mesh, then take the contiguous run of silhouette along the
        station normal nearest the station point.  Measured on a rest-pose
        broadside render (the view a flank mask shows; 1024 px, fx 4000, depth
        22, the shipped ``"girth"`` fin rule, tau 1 px; reproduced by
        ``outputs/demo_2026-09-03/work/girth/agree.py`` ->
        ``agreement.json``), this method reads a MEDIAN 1.052 of that chord over
        the 13 stations and 1.033 over the trunk stations 3-9, with per-station
        ratios 1.04 1.16 0.99 1.07 1.03 1.00 1.05 0.84 1.18 1.03 1.32 1.29 1.09.
        So on that REST-POSE SYNTHETIC render the two agree to ~3% where the
        body is body -- the ~5% the term was asked for.  ON REAL FOOTAGE THEY DO
        NOT: at the fitted optimum (girth round 2, 2026-09-05,
        ``outputs/demo_2026-09-03/work/girth2/taske/scores/*_e3.json`` and
        ``report_rows.json`` ``model_operator_gap_per_station``) this method
        divided by a chord through the fit's OWN rendered silhouette at the same
        station reads a trunk median of 0.885-0.911 on the demo and 1.114-1.156
        on clip 2, per station 0.548-1.741 -- 9-16% off, in OPPOSITE directions
        on the two clips.  That gap is larger than the girth effect being
        fitted, which is why the width term is shipped at weight 0: the objective
        and the score disagree by more than the effect.  On the synthetic render
        they part company in four knowable places:

        * STATIONS 10 and 11 (1.32, 1.29) are the peduncle, where the caudal
          fin's root sits INSIDE the local body-radius envelope: G1's gate keeps
          those vertices (correctly, for a girth field that must not tear the
          fin off the body) while the chord at the peduncle does not cross them.
          This is the largest single disagreement and it is a disagreement about
          where the caudal fin begins, not about how wide the trunk is.
        * The band is a SLAB, half a station spacing deep, so on a tapering
          body its widest vertex is not at the station plane the chord cuts;
          that biases this side UP, and is most of the trunk's 3%.
        * STATION 7 (0.84) is the pectoral station in the broadside view: the
          chord there runs through the pectoral's projected sheet, which the
          gate excludes.
        * The chord is silhouette (it includes whatever face happens to be in
          front) while this is vertices of the band only, so a fin lying ACROSS
          the station -- the pectoral in a TOP-DOWN view -- is in the chord and
          not here: the same rest render viewed from above reads 0.63 at
          station 6 against 1.05 broadside.

        The SOFT part of the extent costs almost nothing on top: over the same
        three views the soft extent is a median 0.9926-0.9936 of the hard
        ``max - min`` of the very same band, i.e. 0.6-0.7% low, which is the
        price of being differentiable.

        None of that is corrected here, and none of it can be: this method
        measures a set of vertices and the score measures a silhouette.  What
        replaced it, on 2026-09-05, is
        :meth:`station_scanline_widths_px` -- the same chord operator as the
        score, made differentiable -- which is now the default
        ``width_model_side``.  What that replacement was measured to do is
        narrower than "agrees with the score", and the difference is the whole
        of this round: it agrees with a chord marched over the MODEL'S OWN hard
        render, at the same station point and the same normal, to within 2% at
        every trunk station on both clips (a per-station MEDIAN over frames, at
        the row-i optimum, 32 frames per clip;
        ``outputs/demo_2026-09-03/work/girth3/operator_accept.json``).  It does
        NOT yet agree with the offline SCORE, which reads at the mask
        CENTRELINE's stations rather than the fitter's: measured on this
        round's six live rows the two headlines differ by 34-71% at the same
        station indices (fit summary
        ``width_series.mean_abs_log_ratio_trunk`` against
        ``girth3/taske/scores/*_e3.json`` -- demo ii/iii/iv
        0.03495/0.01415/0.03027 against 0.05571/0.04952/0.05526; clip 2
        0.18187/0.14549/0.17156 against 0.28322/0.25452/0.26173), because the
        two sets of station points lie up to 0.44 (demo) and 0.66 (clip 2) of a
        mask chord apart along the axis
        (``girth3/taske/station_offset.json``).  This method remains reachable
        (``width_model_side: band``) because every calibration row before then
        was driven by it.  What WAS corrected earlier the same day is the
        target: it is now the mask CHORD
        (:func:`~shark_pose.sim2real.mask_centerline.chord_width`, measured by
        the extractor at the same 13 stations) instead of twice the distance
        transform, which is an INSCRIBED diameter and, at those same stations,
        0.925 (demo) / 0.968 (clip 2) of the chord over the trunk stations 3-9
        and the bend-gated records
        (``outputs/demo_2026-09-03/work/girth2/fix1/dt_over_chord.json``; the
        0.888 / 0.981 pair this comment carried before 2026-09-05 was the same
        ratio measured at ``e2_score.py``'s stations, 13-250 native px
        headward).  Measured on the shipped fits, that one change
        moves the trunk width ratio from 1.203 to 1.049 (demo) and 1.110 to
        1.083 (clip 2) with no change to the fit at all: most of the demo's
        apparent 20% fatness was the ruler.  See ``scripts/fit_video.py``'s
        ``record_station_widths`` and the ``width_series`` block of
        ``configs/inference/fit_video_v2.yaml``.
        """
        ...

    def _station_section_cut(self):
        """``(a, b, w, mask)``, each (K, E): the mesh edges each station cuts.

        TASK D1.  The station's cross-section is the set of points where the
        mesh's EDGES cross the station's own along-body fraction.  That
        fraction (``SharkSMPL.girth_fraction``) is a REST-frame per-vertex
        scalar, so which edges are cut, and how far along each one the crossing
        sits, are decided ONCE here: the section of a posed mesh is then the
        fixed linear map ``(1 - w) * V[a] + w * V[b]``, as differentiable as
        the band's gather and cheaper (54-217 edges a station on
        ``data/model_assets_v2``, against a 1016-wide padded band).

        NO FIN GATE, and that is the point rather than an oversight: the target
        is a chord through a SEGMENTATION MASK, and a mask contains the dorsal
        fin whether or not the girth field is allowed to scale it.  Measured at
        the GT pose of the synthetic two-view set, gating the cut costs 16 %
        (broadside) and 23 % (roll 40) of the mask chord at the first-dorsal
        station and nothing anywhere else
        (``roll/operator/decomposition.json``, rung ``C_over_D_fin_gate``).
        The gate keeps doing its own job -- deciding which vertices the girth
        field moves -- untouched.

        Padded to the widest section with ``mask`` False on the padding, so a
        station with fewer edges neither wins nor takes weight in the extent.
        """
        ...

    def station_section_widths_px(self, vertices: Tensor, joints: Tensor) -> Tensor:
        """(M, K) projected extent of the body's CROSS-SECTION at each station.

        TASK D1's unbiased model side.  ``vertices`` is (M, N, 3) and
        ``joints`` (M, J, 3), both in the CAMERA frame; the result is in proxy
        pixels and differentiable in both.  Rasteriser-free and image-free, like
        :meth:`station_widths_px`, and cheaper than it.

        Per frame and station: cut the posed mesh at the station's own
        along-body fraction (:meth:`_station_section_cut`), project the cut
        points, and take the extent of their signed offsets along the SAME
        image-plane station normal :meth:`_station_axis_px` gives the band and
        the scanline -- the normal
        :func:`~shark_pose.sim2real.mask_centerline.chord_widths_along_polyline`
        also uses on the mask's own polyline.  The extent is the hard
        ``max - min`` at the shipped :data:`WIDTH_SECTION_TAU_PX` of 0, and
        :func:`_soft_extent` above it.

        WHAT IT WAS MEASURED TO DO.  At the TRUE pose and TRUE girth of the
        four synthetic two-view clips (``multiclip/synth``, 20 frames a view),
        against the chord marched over the GT mask at the model's own station
        point and normal, per trunk station 3-9
        (``roll/operator/tau_sweep.json``, ``fix_check.json``):

            operator                 worst trunk deviation, 4 views
            band  (shipped)          34.4 %  31.0 %  31.7 %  31.0 %
            band  at tau 0           17.3 %  18.5 %  17.6 %  18.2 %
            section, fin-gated       15.5 %  23.2 %  14.7 %  23.1 %
            section (this)            1.6 %   1.3 %   1.3 %   1.2 %

        i.e. 1.00 +/- 0.02 at every trunk station on BOTH views, which no
        setting of the band reaches.  Against the shipped ALIGNED TARGET (the
        extractor's own chord, read at the arc fraction the model's station
        falls on) the same measurement is 1.00 +/- 0.021 broadside and
        +/- 0.044 at roll 40; the extra 2 % is the target's, not this
        operator's -- it is the angle between the model's station normal and
        the mask centreline's local normal at the two fin-dominated stations 6
        and 7, where a 3.5 % direction change moves a fin-length chord.
        """
        ...

    def station_widths_for_operator(self, vertices: Tensor, joints: Tensor, operator: str) -> Tensor:
        """(M, K) all 13 station widths by the NAMED operator.

        The one dispatch the post-fit diagnostics share (the girth Jacobian's
        observed axis, the width-agreement score), so a new model side reaches
        them by being added here rather than in each of their ternaries.
        """
        ...

    def _width_data(self, width_series, width_frames: Optional[Sequence[int]], n_frames: int, device, dtype):
        """Validate the per-frame mask width targets.  ``None`` when term is off.

        Returns ``(frame_idx, target (M, S), valid (M, S))`` for the SELECTED
        stations only (``width_stations``); a non-finite or non-positive target
        is dropped as "the mask measured nothing here" rather than fitted to.
        """
        ...

    def _measured_widths_px(self, vertices: Tensor, joints: Tensor, target: Optional[Tensor]=None) -> Tensor:
        """(M, S) model width at the COMPARED stations, by the chosen operator.

        ``target`` is the mask's own (M, S) width, used only to size the
        scanline's reach: the line runs :attr:`width_scanline_half_scale` times
        the mask's width either side of the station, which is a support
        decision (detached) and saves measuring the model twice.  A target that
        is missing or non-positive falls back to the widest one in its frame,
        so a station the mask could not measure is still scanned.  Without a
        target the scanline sizes itself off the band operator.
        """
        ...

    def _width_norm_data(self, width_length_series, width_length_seg_weights, n_width_rows: int, device, dtype):
        """``(length (M,), seg weights (M, K-1))`` in proxy px, or ``None``.

        S6 lane wl_term.  The DENOMINATOR of the length-normalised width
        residual: one mask trunk length per ``width_series`` ROW, in the same
        order, so the loss can divide row by row without a second index.  The
        segment weights are the same cut ``_length_data`` takes -- the mask
        centreline stops at the peduncle while the rig's last station sits 4.4%
        past it, so both sides are cut at the same arc fraction or the ratio
        is biased by that 4.4% (see :meth:`station_chain_length_px`).

        A non-finite or non-positive length is kept as a row and simply makes
        that row's widths invalid in :meth:`_width_loss`: "the mask measured no
        length here" is not "the animal is zero long".
        """
        ...

    def _width_align_data(self, width_polyline, width_profile, width_profile_frames: Optional[Sequence[int]], n_frames: int, device, dtype):
        """Validate the mask's width PROFILE, or ``None`` when alignment is off.

        TASK F1 -- THE STATION-ALIGNMENT FIX.  The shipped width term compares
        two numbers read at two different places on the animal: the MODEL's
        band width at its own projected spine station ``k``, and the MASK's
        ``stations_chord_px[k]``, the chord the extractor measured at arc
        fraction ``station_arc_fractions[k]`` of the mask centreline.  Those
        are the same cross-section only if the fitted body has the mask's own
        length and bend, and it does not: measured on the shipped fits, the
        model's station and the record's sit up to 0.28 (demo) and 0.67
        (clip 2) of a mask chord apart along the axis
        (``outputs/demo_2026-09-03/work/identify/girth/align_ref.json``), which
        is larger than the girth effect being fitted and is why the shipped
        width rows made clip 2 worse.

        The fix moves the TARGET, not the measurement point.  Given the mask
        centreline polyline and its chord profile at the SAME points, the
        target at station ``k`` is the profile read at the arc fraction the
        model's own projected station falls on (foot of perpendicular).  Both
        sides then describe the same cross-section of the same image.  Doing
        it inside the objective rather than once offline matters: the model's
        station moves as the fit converges, so a target aligned to the warm
        start is aligned to the wrong body.

        Returns ``(polyline (T, P, 2), profile (T, P))`` in PROXY pixels,
        indexed by CLIP frame with NaN rows where no record exists -- so the
        loss can gather by its own ``frame_idx`` without a second index -- or
        ``None`` when the term is off or the inputs are missing.
        """
        ...

    def _aligned_width_target(self, joints: Tensor, frame_idx: Tensor):
        """``(target (M, S), valid (M, S))`` read at the MODEL's own stations.

        TASK F1.  Detached throughout by default -- this decides WHERE on the
        mask to read, which is a correspondence, not a residual; the
        differentiable half stays the model's band width at the station it
        already uses.  With ``width_align_grad`` on, the two DISCRETE choices
        (which polyline segment the foot of perpendicular lands on, and which
        profile interval brackets its arc fraction) are still made under
        ``no_grad`` -- they are integers, and no gradient exists through an
        ``argmin`` -- while the continuous read at that fraction is replayed
        with gradient, so a mode that slides the station along the body is
        charged for the width it slides onto.
        """
        ...

    def _width_loss(self, smpl_out, width) -> Tensor:
        """Robust error between the fitted and the mask's station widths.

        The one observation in this objective that can see GIRTH.  The 16
        landmarks are points on the flank and the midline, so they fix the
        body's outline only where they sit; the bend series is an angle, hence
        scale-free by construction; the midline term is a curve with no
        thickness.  A width is the one number that changes when the animal gets
        fatter and nothing else does.

        Robust, per station and frame, because a mask width is wrong in a
        specific way rather than noisily: a fin merged into the trunk, or a
        remora, adds tens of pixels at ONE station and should be rejected, not
        averaged in.

        The comparison is in proxy pixels, so it is a claim about the IMAGE,
        not about the animal: at fixed girth a body 10% further away is 10%
        narrower on screen, and the term will accept that as readily as a
        thinner shark.  What separates the two is the rest of the objective
        (the landmarks fix the apparent LENGTH, the depth is theirs), which is
        why this term is a stage-B refinement of shape and not a placement cue.

        WHAT IT CAN MOVE: the pose, the translation, the betas -- and, when
        ``girth_enabled`` is on, the clip's shared GIRTH field, which is the
        only one of those that can change thickness without changing anything
        else (the shape basis has no girth mode: per unit beta the mid-trunk
        moves <= 3-4% laterally).  With the girth leaf OFF the cheapest way for
        the fit to pay a width error is depth, which is why the leaf exists.
        """
        ...

    def station_chain_length_px(self, joints: Tensor, seg_weights: Optional[Tensor]=None) -> Tensor:
        """(M,) projected arc length of the 13-station chain, in proxy pixels.

        ``joints`` is (M, J, 3) in the CAMERA frame, straight out of
        :class:`SharkSMPL`.  The 13 spine stations are projected through the
        fitter's own camera and the 12 chords between them summed -- so this is
        an APPARENT length, the number the image can actually see, not a metric
        one: at fixed body size an animal 10% further away is 10% shorter here.

        ``seg_weights`` is (M, 12) or (12,) in ``[0, 1]``: how much of each
        chord lies inside the span the mask was measured over.  It exists
        because the two curves do not end in the same place.  With the shipped
        peduncle-terminated axis the rig's last station sits at arc fraction
        1.044 of the mask centreline -- 4.4% PAST its end -- so the mask has no
        arc there to compare against, and counting the model's full chain
        against a mask cut at 1.0 would ask the fit to shrink by that 4.4%.
        The driver therefore cuts BOTH sides at the same arc fraction: the mask
        by :func:`~shark_pose.sim2real.mask_centerline.polyline_span_length_px`
        and the model by the fraction of each chord that survives the cut,
        which is this argument.  ``None`` means the whole chain.

        Differentiable in ``joints``, and rasteriser-free.
        """
        ...

    def _length_data(self, length_series, length_frames: Optional[Sequence[int]], length_seg_weights, n_frames: int, device, dtype):
        """Validate the per-frame mask length targets.  ``None`` when term is off.

        Returns ``(frame_idx, target (M,), seg_weights (M, K-1))``.  A
        non-finite or non-positive target is dropped -- "the mask measured no
        length here" is not "the animal is zero long".
        """
        ...

    def _length_loss(self, smpl_out, length) -> Tensor:
        """Robust error between the fitted and the mask's apparent body length.

        The width series and the girth leaf can only say how THICK the body
        looks; nothing in the objective said how LONG it looks, because the two
        landmarks that would pin it -- the snout tip and the caudal tips -- are
        the two the detector fires on least (measured on 22K real detections:
        96% of frames have no tail keypoint, 82% no head one).  That is the
        degeneracy the girth pass ran into: given a width target and no length
        observation, clip 2 paid for a thinner body by growing 4-7% LONGER, and
        both changes were reported as shape.

        The observation is one scalar per frame, in proxy pixels, so it is a
        claim about the IMAGE: a body 10% further away is 10% shorter on screen
        and this term will accept that as readily as a smaller shark.  What it
        buys is that the length can no longer move for free while the width
        term pushes on thickness -- the two now cost against each other in the
        same currency.

        Geman-McClure, because a mask length is wrong in a specific way rather
        than noisily: a truncated mask, or a centreline that ran out along the
        caudal lobe, is short or long by a large amount in one frame and should
        be rejected, not averaged in.  (The frame gate already drops the worst
        of those; this is the second line.)
        """
        ...

    def _girth_tie_loss(self, girth: Tensor) -> Tensor:
        """``mean((g_x - g_z)^2)`` -- the girth field's isotropy tie.

        The field has two axes per station and this geometry observes one of
        them.  On a broadside view the width a mask shows IS the dorsoventral
        axis ``g_z``; the lateral ``g_x`` is edge-on to the camera and the
        objective barely sees it.  Measured over the whole clip through the
        SCANLINE operator, the width term's sensitivity to ``g_z`` is 5.95x its
        sensitivity to ``g_x`` on the demo and 3.13x on clip 2 (8.14x and 4.51x
        through the band operator -- ``girth3/observed_axis.json``,
        ``summary girth.observed_axis``), and across girth round 2's rows
        ``g_z`` reproduced to +/-3% within a clip while ``g_x`` wandered
        0.74-1.55 between rows that fitted the same animal.  Without this term
        that wander is written out as shape.

        With it the fit says "as wide across as it is deep unless something in
        the data says otherwise": exactly zero, with zero gradient, on an
        isotropic field, and a view that DOES see ``g_x`` -- a second camera, a
        top-down pass -- can still pull the two apart for the price of this
        penalty.  It is a prior, not an observation, and
        :attr:`girth_tie_weight` says how strong.

        IT IS SYMMETRIC, and measurably so: it does not hold the observed axis
        fixed while moving the unobserved one.  Across girth round 3's rows ii
        (tie 1.0) and iv (tie 0.0), identical otherwise, ``g_x`` moves -4.2% to
        -13.4% per station (the intended effect) but ``g_z`` moves +0.4% to
        +1.8% on the demo and -2.2% to +7.2% on clip 2 (station 0:
        0.6767 -> 0.7256), while ``mean|g_x - g_z|`` falls from 0.093 to 0.031
        (demo) and 0.035 (clip 2) -- ``girth3/taske/runs/{demo,clip2}_{ii,iv}``.
        The dorsoventral column is the only girth axis these views can measure,
        so a biometric quoted off a fit with the tie at the shipped weight of
        1.0 is quoted off a number this prior has moved.  A one-sided form --
        penalising ``g_x`` against ``g_z.detach()`` -- would leave the observed
        axis alone; it has not been measured and is not what ships.

        TASK F1 -- WITH A PRIOR MEAN the tie is taken in units of that mean,
        ``mean((g_x/mu_x - g_z/mu_z)^2)``.  It has to be: the tie's whole
        content is "the cross-section keeps the shape the prior says it has
        unless the data show otherwise", and once the prior's two axes differ
        (a shark is deeper than it is wide -- the template's own
        dorsoventral/lateral semi-axis ratio is 0.76 and the literature mean
        is not 1:1 either) pulling the two RAW scales together would be
        pulling the animal round, i.e. undoing the prior with the prior's own
        weight.  ``girth_prior_mean = None`` keeps the shipped expression
        exactly, mu = 1 on both axes making the two forms identical anyway.
        """
        ...

    def _upright_loss(self, pose: Tensor) -> Tensor:
        """Smooth hinge holding the body upright (dorsal fin toward camera up).

        ``pose`` is the (T, J, 3) axis-angle pose; the root station's entry is
        the body's world orientation (the fit's global rotation with the prior's
        own root recoil composed in), so its columns are the body axes expressed
        in camera coordinates: column 0 = lateral, 1 = longitudinal, 2 = dorsal.

        In ``"dorsal"`` mode the penalty is on the angle between the dorsal axis
        and camera up (OpenCV: -y).  In ``"broadside"`` mode it is on how far
        the lateral axis lies OUT of the image plane, i.e. ``asin|R[2, 0]|`` --
        a claim about the CAMERA (it sees the flank), not about the animal, and
        on the demo clip it fights the upright one: 48.6 deg median roll off
        vertical, +1.09 px residual, -0.08 IoU.  Both
        are ``relu(angle - margin) ** 2``: exactly zero, with zero gradient,
        while the body is inside the cone.
        """
        ...

    def _make_params(self, n_frames: int, device, dtype):
        """Leaf tensors at the cruise defaults; clip scalars are (1,) or (T,)."""
        ...

    def _spine_params(self, raw: dict, n_frames: int):
        """Assemble a :class:`SpinePoseParams` with every field at (T, ...)."""
        ...

    def _pose(self, params, times: Tensor, root_orient: Tensor, clamp: bool) -> Tensor:
        """(T, J, 3) axis-angle: the prior's rotations with the root composed in.

        The prior gives the root station its own small local rotation (the body's
        yaw recoil); the fit's global orientation multiplies that on the left.

        TASK D3: with :attr:`frozen_pose` set -- the debug arm, ``None`` in
        every shipped fit -- the given pose is returned verbatim, clamp and
        all: a GT pose is already anatomically valid by construction, and
        clamping it would move the very thing the arm is holding fixed.
        """
        ...

    def _observations(self, smpl_out) -> Tensor:
        """Visible-flank observation landmarks projected to proxy pixels."""
        ...

    def _loss(self, raw: dict, root_orient: Tensor, translation: Tensor, betas: Tensor, betas_init: Tensor, times: Tensor, keypoints_2d: Tensor, keypoint_conf: Tensor, midline=None, bend=None, width=None, girth: Optional[Tensor]=None, length=None, sil_batch: Optional[Tuple[Tensor, Tensor]]=None, stage: Optional[str]=None, *, include_shared: bool=True) -> Tensor:
        """Total objective.

        ``midline`` is the pre-computed centreline correspondence from
        :meth:`_midline_data` (or ``None``); unlike the silhouette it is scored
        on every centreline every step, in both stages, because 13 points per
        frame is cheap and the bend plane is decided during global placement.

        ``bend`` is the per-frame mask bend series from :meth:`_bend_data` (or
        ``None``), ``width`` the per-frame station widths from
        :meth:`_width_data` and ``length`` the per-frame mask arc length from
        :meth:`_length_data`.  ``stage`` is ``"A"`` / ``"B"`` while a stage is
        running and ``None`` when the converged objective is being scored; both
        terms obey their ``*_stages`` in the former case and always run in the
        latter, so every amplitude well is ranked on the same complete
        objective.

        ``girth`` is the clip's shared (K, 2) girth leaf, or ``None`` -- which
        is what a fit with ``girth_enabled = False`` passes, all the way into
        :class:`SharkSMPL`, so such a fit is bit-identical to one built before
        the leaf existed.

        ``sil_batch`` is ``(frame_index, target)`` -- the slice of masked frames
        scored by the silhouette term this step -- or ``None`` to skip it.

        ``include_shared`` -- **TASK B1, default True, i.e. the shipped
        objective bit for bit** -- gates the three terms that belong to a
        quantity SHARED across the clips of one individual rather than to this
        clip: the shape prior, the girth prior and the girth isotropy tie.  A
        joint fit over N clips bills them ONCE, inside clip 0's expression at
        the position they occupy here, and passes ``include_shared=False`` for
        every other clip.  They are billed there rather than added after the
        clip sum on purpose: float addition is not associative and gradient
        accumulation into a shared leaf follows the graph's topology, so at
        N = 1 the statement sequence has to be the one below, unchanged.
        """
        ...

    def _temporal_is_split(self) -> bool:
        """Is either half of the temporal term separately weighted? (TASK F)"""
        ...

    def _temporal_loss(self, raw: dict, root_orient: Tensor, translation: Tensor) -> Tensor:
        """Smoothness of the manoeuvre: turn(t), the fins, the root and the depth.

        The wave itself is NOT smoothed -- its time course is the prior's job --
        so this only penalises jitter in the terms the data can pull frame by
        frame.

        **P4, default off.**  Without an ego track (``self._ego_R is None``) the
        expression below is the original one, evaluated in the CAMERA frame: a
        zero-angular-velocity prior on ``root_orient`` taken component-wise on the
        axis-angle vector, and a translation smoother that charges a camera pan as
        animal motion.  With an ego track the root terms become **geodesic** d1/d2
        on ``R_wb = R_cw.T R(root_orient)`` and the translation terms act on the
        **de-rotated** translation ``R_cw.T t``, so a rigid camera rotation costs
        nothing.  ``R_cw = I`` at an invalid endpoint is PROHIBITED -- ``R_cw`` is
        an integrated ABSOLUTE orientation, so identity-filling one endpoint dumps
        the whole accumulated camera rotation into a single step and the smoother
        charges it as animal motion.  Invalid pairs are dropped from the sum and
        the mean renormalised over the pairs that survive.

        **TASK F.**  This method is the COUPLED form only.  When
        ``root_temporal_weight`` or ``translation_temporal_weight`` is set,
        :meth:`_loss` calls :meth:`_temporal_split_loss` instead and this one is
        not reached; the branch is there so the shipped expression below stays
        byte-for-byte what it was.
        """
        ...

    @staticmethod
    def _masked_mean(sq: Tensor, keep: Tensor) -> Tensor:
        """Mean of ``sq`` over the rows ``keep`` selects; 0 when none survive.

        ``sq`` is (N, ...) and the mean is over EVERY element of the kept rows, so
        a per-component squared difference keeps the same scale it has today.
        """
        ...

    @staticmethod
    def _geodesic_sq(Ra: Tensor, Rb: Tensor) -> Tensor:
        """(N,) squared geodesic angle, radians^2, between two rotation stacks.

        ``theta = atan2(|sin theta|, cos theta)`` read off ``S = Ra^T Rb``: the
        skew part gives ``sin theta * axis`` and the trace gives ``cos theta``.
        Equal to the textbook ``acos((tr(S) - 1) / 2)`` away from the poles, but
        at coincidence it returns a value at the dtype's floor (``sin`` is
        clamped to ``finfo.tiny``, so ~1e-38 in float32 -- NOT exactly zero)
        rather than ``acos``'s clamp plateau, and its gradient stays finite
        there.  That floor is what makes the injected-rotation test meaningful:
        the compensated ``d1`` lands ~27 orders of magnitude below the
        uncompensated term instead of on ``acos``'s ~1e-7 plateau.
        """
        ...

    def _ego_temporal(self, root_orient: Tensor, translation: Tensor) -> Tensor:
        """Geodesic root d1/d2 on the world-frame ROOT rotation + de-rotated translation.

        The rotation smoothed here is ``R_cw^T R(root_orient)`` -- the OPTIMISER
        VARIABLE -- not the record's ``R_w_root`` (``world_frame.body_attitude``),
        which composes the prior's root recoil into ``pose[root_joint]``; the two
        differ by that recoil (<= 0.056 deg on the shipped demo fit).
        """
        ...

    def _temporal_halves(self, root_orient: Tensor, translation: Tensor, smpl_out=None) -> Tuple[Tensor, Tensor]:
        """``(root_term, translation_term)``: geodesic root d1/d2 and the
        translation half, each an unweighted mean.

        Deliberately NOT factored out of :meth:`_ego_temporal`: that method's
        four sub-terms are summed left to right and re-associating them is not
        ``torch.equal`` in float32, so the shipped path is left untouched and
        this one duplicates the (short) arithmetic instead.

        The root half is the **geodesic** d1/d2 of ``R_wb`` -- the world-frame
        root rotation, which is just ``R(root_orient)`` when no ego track is
        installed.  The axis-angle d1 it replaces measures the same trajectory
        as up to 578x stiffer or softer depending on where the animal happens to
        be heading (``root_jitter/scaling/scaling.md`` section 4); the geodesic
        is invariant by construction.  The ``/3.0`` is the one
        :meth:`_ego_temporal` carries, and for the same reason: the geodesic
        mean is over PAIRS while the axis-angle d1 averages over 3 components
        per pair, so dividing by 3 keeps the term on the scale the weights were
        calibrated against.

        ``translation_temporal_target`` picks what the translation half sees;
        ``"centroid"`` and ``"both"`` need ``smpl_out``, whose joints are
        already computed by :meth:`_loss`.
        """
        ...

    def _temporal_split_loss(self, raw: dict, root_orient: Tensor, translation: Tensor, smpl_out=None) -> Tensor:
        """The temporal term with the root and translation halves weighted apart.

        Returns the term ALREADY weighted (the manoeuvre rails at
        ``temporal_weight``, each half at its own weight), because the three
        weights no longer factor out.  A half whose weight is ``None`` keeps
        ``temporal_weight``, so setting only one of them moves only that one.
        """
        ...

    def _aniso_root_term(self, root_orient: Tensor) -> Tensor:
        """Per-camera-axis root smoother, weights already applied.

        ``omega_k = log(R_{k+1} R_k^T)`` -- the LEFT increment, whose rotation
        vector lies on CAMERA axes because ``root_orient`` is the body->camera
        rotation (the right/body increment smears a camera-axis rotation 83-97 %
        into the other two; ``kinematic/aniso/aniso.md`` section 1) -- and

            d1 = mean_k sum_a w_a omega_{k,a}^2
            d2 = mean_k sum_a w_a (omega_{k+1,a} - omega_{k,a})^2

        both divided by 3, the divisor :meth:`_ego_temporal` carries so a
        per-PAIR angle keeps the scale of a per-COMPONENT axis-angle
        difference.  With three equal weights the ``d1`` half is EXACTLY the
        isotropic geodesic ``d1`` (``sum_a omega_a^2 = theta^2``) but the ``d2``
        half is the second difference of the LOG COORDINATES rather than the
        geodesic distance between consecutive increments; they agree only to
        O(theta^3), which is why equal weights never reach this method --
        :func:`_parse_root_temporal_weight` reduces them to the scalar case.

        **With an ego track installed the axes are the WORLD frame's**, not the
        camera's, because the smoothed rotation is ``R_cw^T R(root_orient)``.
        The per-axis weights this leaf was calibrated with are camera-frame
        (``kinematic/aniso/aniso.md`` section 3), so an anisotropic triple is
        only meaningful for ``ego=None`` fits -- which is every shipped fit.
        """
        ...

    def _swim_is_on(self) -> bool:
        """Is any leaf of the swimming prior weighted? (TASK F)"""
        ...

    def _swim_boxcar(self, n: int, device, dtype) -> Tensor:
        """The clip's P7 boxcar matrix, built once and cached."""
        ...

    def _swim_loss(self, pose: Tensor, translation: Tensor) -> Tensor:
        """The swimming prior, already weighted; 0 when every leaf is off.

        Every term is on the **0.5 s low-passed** velocity, because the defect
        it addresses is a low-frequency drift (lag-1 autocorrelation +0.36..+0.80
        on every component of both clips) and the frame-rate jitter belongs to
        the root smoother.  Body axes are the columns of
        ``R(pose[root_joint])`` at the pair midpoints, re-normalised, and they
        carry gradient -- which is what makes this a SWIMMING prior rather than
        a camera-frame one.
        """
        ...

    def _set_ego(self, ego, n_frames: int, device, dtype) -> None:
        """Install (or clear) the P4 camera track for one :meth:`fit` call.

        ``ego`` is ``None`` (the default and every shipped fit), an
        :class:`shark_pose.sim2real.egomotion.EgoTrack` (duck-typed: ``R_cw()``
        and ``valid()``), or a mapping with ``R_cw`` (T, 3, 3) and ``valid`` (T,).
        A track with no measured, valid frame is refused -- an unmeasured camera
        does not become a measurement by being handed to the fitter -- and so is
        a track on a fit whose frames are not a clip (``share_clip_params`` off),
        which has no temporal term at all.
        """
        ...

    def _amp_well(self, amp: float) -> str:
        """Which well of the mixture amplitude prior ``amp`` sits in.

        The barrier between the glide and swimming components lies between
        their modes; in log-amplitude, where the prior lives, "between" is the
        geometric mean.
        """
        ...

    @torch.no_grad()
    def _init_wave(self, raw: dict, *loss_args) -> list[dict]:
        """Coarse (amplitude x frequency x phase) grid search.

        Fills ``self.warm_start_scores`` with the best phase at every
        (amplitude, frequency) pair and ``self.freq_grid_scores`` with the best
        amplitude and phase at every candidate frequency (the per-frequency view
        the summary reports).  Returns the starts stage A should be run from:
        the overall winner first, then the best candidate in the OTHER well of
        the mixture amplitude prior, since Adam cannot cross the barrier between
        the two and the coarse score alone is not enough to rule a well out.

        Skipped -- returning the leaves' own init as the single start -- when
        the frames are independent (``t = 0`` makes the frequency unobservable)
        or when either grid is empty.
        """
        ...

    @property
    def silhouette_anneal_on(self) -> bool:
        """Is the TASK F1 sharpness anneal live on this fitter?"""
        ...

    @property
    def silhouette_sigma_effective(self) -> float:
        """The softness in PROXY px the term is at right now.

        :attr:`silhouette_sigma_px` unless the TASK F1 anneal has been stepped
        (:meth:`set_silhouette_anneal_progress`), which is never in a default
        fit -- so this is the constant it always was.
        """
        ...

    def set_silhouette_anneal_progress(self, progress: float) -> float:
        """Put the rasteriser's softness at ``progress`` along the anneal.

        TASK F1.  ``progress`` is the fraction of STAGE B already run, in
        [0, 1].  The softness moves GEOMETRICALLY from
        :attr:`silhouette_sigma_px` to :attr:`silhouette_sigma_px_final` over
        the first :attr:`silhouette_anneal_frac` of that, then holds -- a blur
        is a scale, so the schedule that is uniform in it is the log one.  A
        no-op (and returns the unchanged softness) while the anneal is off.

        Returns the softness now in force, in proxy pixels.
        """
        ...

    @property
    def silhouette_sigma_raster(self) -> float:
        """The rasteriser ``sigma`` at :attr:`silhouette_res`, in raster px.

        :attr:`silhouette_sigma_px` is a PROXY-pixel length, so it has one
        meaning at every resolution; this is that length expressed in the
        pixels of the raster it is applied to (see
        :func:`~shark_pose.losses.soft_silhouette.sigma_at`).  With the TASK F1
        anneal running it is :attr:`silhouette_sigma_effective` that is scaled,
        which equals :attr:`silhouette_sigma_px` everywhere else.
        """
        ...

    def _silhouette_camera(self) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        """``(base_focal, base_principal)`` for ``soft_silhouette``.

        The rasteriser expresses its camera at
        :data:`~shark_pose.losses.soft_silhouette.BASE_RESOLUTION`; ``__init__``
        has already established that the fitter works there too, so the fitter's
        own focal and principal point ARE the base ones.  ``None`` is returned
        for whichever half equals the pipeline default, so a default fitter
        rasterises through exactly the code path it always did.
        """
        ...

    def _silhouette_batches(self, masks: Optional[Tensor], mask_frames: Optional[Sequence[int]], n_frames: int, device, dtype):
        """Return a callable handing out the next slice of masked frames.

        The slices tile the masked frames in order and wrap around, so a stage
        of S steps scores ``S * frames_per_step`` frame-masks in total and every
        mask is seen roughly ``S * frames_per_step / M`` times -- an unbiased
        stochastic estimate of the clip's mean IoU loss at a bounded cost.
        Returns a callable yielding ``None`` when the term is off.
        """
        ...

    def _shape_scale(self, device, dtype) -> Optional[Tensor]:
        """(1, NUM_BETAS) per-mode sd, or ``None`` when the leaf IS the betas.

        TASK F1.  ``None`` -- the default -- means :meth:`fit` optimises the
        betas directly, which is the shipped code path.
        """
        ...

    @staticmethod
    def _betas_of(leaf: Tensor, scale: Optional[Tensor]) -> Tensor:
        """The betas a shape leaf stands for (the leaf itself when unscaled)."""
        ...

    def _freeze_betas(self, betas: Tensor) -> Tensor:
        """Return ``betas``, with the frozen components' gradient zeroed.

        TASK F1.  With no ``shape_freeze_components`` -- the default -- the
        leaf is returned untouched and NO hook is registered, so the fit is
        bit-identical to one built before this existed.  Otherwise a backward
        hook zeroes those entries of every gradient the leaf ever receives, so
        Adam never accumulates a moment for them and they stay at
        ``betas_init`` for the whole fit.
        """
        ...

    def _girth_init_value(self, width, root_orient, translation, betas, times, n_frames, device, dtype) -> Optional[Tensor]:
        """The clip's (K, 2) girth START value, or ``None`` when the leaf is off.

        Computed ONCE per :meth:`fit` and cloned into a leaf by every amplitude
        well, so the wells are compared from the same shape.

        ``"from_targets"`` measures the WARM START's own station widths with
        :meth:`station_widths_px` and takes MASK / MODEL per station, i.e. the
        factor by which the body has to widen, averaged over the frames that
        carry a target.  Those 13 ratios sit at the prior's ``spine_fraction``
        -- the same along-body ruler ``SharkSMPL.girth_fraction`` uses -- and
        :func:`~shark_pose.model_3d.girth_from_width_ratio` resamples them onto
        the K girth stations.
        """
        ...

    def prepare_clip(self, init_root_orient: Tensor, init_translation: Tensor, keypoints_2d: Tensor, keypoint_conf: Tensor, init_betas: Optional[Tensor]=None, times: Optional[Tensor]=None, masks: Optional[Tensor]=None, mask_frames: Optional[Sequence[int]]=None, centerlines: Optional[Tensor]=None, centerline_frames: Optional[Sequence[int]]=None, centerline_widths: Optional[Tensor]=None, bend_series: Optional[Sequence[float]]=None, bend_frames: Optional[Sequence[int]]=None, width_series: Optional[Tensor]=None, width_frames: Optional[Sequence[int]]=None, width_polyline: Optional[Tensor]=None, width_profile: Optional[Tensor]=None, width_profile_frames: Optional[Sequence[int]]=None, width_length_series: Optional[Sequence[float]]=None, width_length_seg_weights: Optional[Tensor]=None, length_series: Optional[Sequence[float]]=None, length_frames: Optional[Sequence[int]]=None, length_seg_weights: Optional[Tensor]=None, ego=None, clip_id: Optional[str]=None) -> ClipContext:
        """Everything :meth:`fit` computes before its amplitude-well loop.

        TASK B1: a pure extraction, statement for statement, so that
        :meth:`fit` and the joint driver share one preparation path.  The
        arguments are :meth:`fit`'s, documented there.  The girth START value
        is deliberately NOT computed here -- a joint fit combines the clips'
        candidates before any leaf exists (see :meth:`girth_init_for`).
        """
        ...

    def girth_init_for(self, ctx: ClipContext) -> Optional[Tensor]:
        """This clip's (K, 2) girth START value, or ``None`` when the leaf is off."""
        ...

    def clip_loss(self, ctx: ClipContext, raw: dict, root_orient: Tensor, translation: Tensor, betas: Tensor, girth: Optional[Tensor], *, sil_batch=None, stage: Optional[str]=None, include_shared: bool=True) -> Tensor:
        """:meth:`_loss` addressed by :class:`ClipContext`, nothing more.

        A thin adapter so the joint driver never learns the context's layout;
        the argument order below is the shipped one.
        """
        ...

    def stage_a_group(self, raw: dict, root_orient: Tensor, translation: Tensor) -> dict:
        """Stage A: global placement plus the body's overall bend.

        The wave and the turn, and nothing else -- one definition, used by the
        single-clip and the joint driver alike, so the two cannot drift.
        """
        ...

    def stage_b_clip_group(self, raw: dict, root_orient: Tensor, translation: Tensor) -> dict:
        """Stage B's PER-CLIP group: every rail, the root and the translation."""
        ...

    def stage_b_shared_groups(self, shape_leaf: Tensor, girth: Optional[Tensor]) -> list[dict]:
        """Stage B's SHARED groups: the shape leaf, then the girth field.

        Both are one leaf per INDIVIDUAL in a joint fit and one per clip in a
        single-clip one; either way they are stage-B parameters only, because
        stage A has neither the mask terms that can see thickness nor a
        placement good enough to trust them.

        ``shape_leaf`` is the OPTIMISED tensor, which is the betas unless
        :attr:`shape_lr_in_sd` is on (then it is z, with beta = sd * z; see
        :meth:`_betas_of`).  ``girth`` ``None`` drops the girth group and
        leaves the shape group alone, which is what the girth-vs-shape prefix
        (:attr:`girth_delay_frac`) runs.
        """
        ...

    def fit(self, init_root_orient: Tensor, init_translation: Tensor, keypoints_2d: Tensor, keypoint_conf: Tensor, init_betas: Optional[Tensor]=None, times: Optional[Tensor]=None, masks: Optional[Tensor]=None, mask_frames: Optional[Sequence[int]]=None, centerlines: Optional[Tensor]=None, centerline_frames: Optional[Sequence[int]]=None, centerline_widths: Optional[Tensor]=None, bend_series: Optional[Sequence[float]]=None, bend_frames: Optional[Sequence[int]]=None, width_series: Optional[Tensor]=None, width_frames: Optional[Sequence[int]]=None, width_polyline: Optional[Tensor]=None, width_profile: Optional[Tensor]=None, width_profile_frames: Optional[Sequence[int]]=None, width_length_series: Optional[Sequence[float]]=None, width_length_seg_weights: Optional[Tensor]=None, length_series: Optional[Sequence[float]]=None, length_frames: Optional[Sequence[int]]=None, length_seg_weights: Optional[Tensor]=None, ego=None) -> Dict[str, Tensor]:
        """Fit one clip.

        Args:
            init_root_orient: (T, 3) global axis-angle warm start (from a v1 SPIN
                forward pass or a v1 fit).
            init_translation: (T, 3) translation warm start, already retargeted to
                this rig's root pivot (see :func:`retarget_translation`).
            keypoints_2d: (T, 16, 2) detected landmarks in proxy pixels.
            keypoint_conf: (T, 16) detector confidences.
            init_betas: (1, 10) or (T, 10) shape warm start. Defaults to zeros;
                a (T, 10) input is averaged, since the shape is one shark.
            times: (T,) seconds. Defaults to ``arange(T) / fps`` for a clip, or
                zeros when ``share_clip_params`` is off.
            masks: (M, res, res) segmentation targets in [0, 1], already
                letterboxed to ``silhouette_res``.  Ignored unless
                ``silhouette_weight > 0``; not every frame needs one.
            mask_frames: (M,) indices into the clip that ``masks`` belong to.
                Defaults to the first M frames, i.e. every frame when M == T.
            centerlines: (M, P, 2) mask centrelines in PROXY pixels, each
                ordered snout -> tail.  Ignored unless ``midline_weight > 0``;
                not every frame needs one.
            centerline_frames: (M,) indices into the clip that ``centerlines``
                belong to.  Defaults to the first M frames.
            centerline_widths: (M, P) local body widths at the centreline
                points, in proxy pixels.  Only read when
                ``midline_use_widths`` is on.
            bend_series: (M,) signed chord bend of the MASK centreline at the
                13 station arc fractions, in degrees, image y down (see
                ``sim2real.mask_centerline.station_bend``).  Ignored unless
                ``bend_weight > 0``; not every frame needs one.
            bend_frames: (M,) indices into the clip that ``bend_series``
                belongs to.  Defaults to the first M frames.
            width_series: (M, 13) body width the MASK measured at the 13
                station arc fractions, in PROXY pixels.  Ignored unless
                ``width_weight > 0``; not every frame needs one, and a
                non-positive entry means "no measurement at this station".
            width_frames: (M,) indices into the clip that ``width_series``
                belongs to.  Defaults to the first M frames.
            width_polyline: (M, P, 2) mask centreline in PROXY pixels, snout ->
                tail, and ``width_profile`` (M, P) the mask's CHORD width at
                those same points.  Read only when ``width_align_stations`` is
                on, and then they REPLACE ``width_series`` as the target: the
                chord is resampled at the arc fraction the model's own
                projected station falls on, so the two sides of the comparison
                describe the same cross-section.  See
                :meth:`_width_align_data`.
            width_profile: see ``width_polyline``.
            width_profile_frames: (M,) indices into the clip that
                ``width_polyline`` / ``width_profile`` belong to.  Defaults to
                the first M frames.
            width_length_series: (M,) apparent trunk length of the MASK in
                PROXY pixels, ONE PER ``width_series`` ROW and in the same
                order (not per clip frame).  Read only when
                ``width_normalize`` is on, and then it is the denominator of
                the width residual; see :meth:`_width_norm_data`.
            width_length_seg_weights: (M, 12) or (12,) -- the same chord cut
                ``length_seg_weights`` carries, applied to the MODEL's
                denominator so both sides span the same arc.  Defaults to the
                whole chain.
            length_series: (M,) apparent arc length of the MASK centreline over
                the span the 13 stations cover, in PROXY pixels.  Ignored
                unless ``length_weight > 0``; not every frame needs one, and a
                non-positive entry means "no measurement in this frame".
            length_frames: (M,) indices into the clip that ``length_series``
                belongs to.  Defaults to the first M frames.
            length_seg_weights: (M, 12) or (12,) in ``[0, 1]`` -- how much of
                each station-to-station chord lies inside the span the mask was
                measured over (see :meth:`station_chain_length_px`).  Defaults
                to the whole chain.
            ego: **P4, default None.**  A camera track -- an
                ``EgoTrack`` (``R_cw()`` / ``valid()``) or a mapping with
                ``R_cw`` (T, 3, 3) and ``valid`` (T,) -- to compensate the
                temporal terms with; see :meth:`_temporal_loss`.  ``None``
                leaves the objective byte-for-byte as it ships, which is what
                every fit on this footage runs, because Gate B refuses the
                estimate on both clips.  Ignored when ``share_clip_params`` is
                off, since there is no temporal term then.

        Returns:
            ``pose`` (T, J*3, limit-clamped), ``params`` (T, 25), ``root_orient``
            (T, 3), ``shape`` (T, 10), ``camera`` (T, 3), ``joints_3d``
            (T, J, 3), ``landmarks_2d`` (T, 16, 2), ``reprojection_error``
            (T,) and ``girth`` ((K, 2), shared by the clip, ``None`` when the
            leaf is off), all detached.  The coarse search's scores are left in
            ``self.freq_grid_scores`` / ``self.warm_start_scores`` and the
            converged objective of each amplitude well in
            ``self.amp_well_scores``.
        """
        ...
