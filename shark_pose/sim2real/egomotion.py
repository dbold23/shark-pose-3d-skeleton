"""Camera ego-motion from the background, and the staged Gate B refusal test.

Plan items P2 and P3 of ``outputs/demo_2026-09-03/work/egomotion/report.md`` section 4:

    P2  ``shark_pose/sim2real/egomotion.py`` -- the diagnostic promoted to a gated
        library, sourced from **both** scripts, one sign convention, the staged gate,
        B4 as a diagnostic only
    P3  ``scripts/estimate_egomotion.py`` -- one run per clip, writes the full gate audit

Sourced from the two shipped per-clip diagnostics
(``egomotion/demo/egomotion.py``, ``egomotion/clip2/egomotion.py``), their unified
port ``egomotion_impl/gates/sweep.py`` and the correlation pass
``egomotion/correlate/correlate.py``.


THE ONE SIGN CONVENTION
-----------------------
Everything in this module is stated in terms of

    ``R_cw(k)``  -- camera-from-world at frame ``k``: ``X_cam(k) = R_cw(k) @ X_world``

with the **world frame defined as the camera frame of the first valid frame**, so
``R_cw(first_valid) = I`` exactly and by definition.  There is no gravity, no north
and no metric scale in this frame; it is a gauge choice, not a measurement.

The background estimator fits, per frame pair ``k -> k+1``, the rotation ``dR`` that
carries frame-``k`` bearings of a *static* scene onto frame-``k+1`` bearings::

    b_{k+1} = dR @ b_k        (Kabsch on unit bearings, RANSAC on reprojection error)

For a world-static point ``X_cam(k) = R_cw(k) X_w`` and ``X_cam(k+1) = R_cw(k+1) X_w``,
so

    ``dR = R_cw(k+1) @ R_cw(k).T``   and   ``R_cw(k+1) = dR @ R_cw(k)``.

``dR`` is therefore the **scene** rotation.  The **camera's own** rotation in the
world is its transpose, so the camera-in-world rotation vector is

    ``r_cam = -rotvec(dR)``          (deg/s after ``* degrees * fps``)

and every *rate* reported by this module -- ``pitch, yaw, roll`` = rotvec components
``x, y, z`` in the OpenCV camera frame (x right, y **down**, z forward) -- is in that
camera-in-world convention.  The demo's shipped JSON already stores this; clip2's
shipped JSON stores ``+rotvec(dR)`` and its own ``sign_convention`` string says so.

Consequences worth stating once, because both shipped scripts state them separately:

* A camera that yaws by ``+d`` about its own ``+y`` makes a world-static body's
  ``root_orient`` in the camera frame change by ``-d`` about ``y``.
* ``R_cw = I`` at an invalid endpoint is **not** a measurement.  Pairs whose fit was
  rejected are integrated as ``dR = I`` only so that the frame index of ``R_cw`` stays
  aligned with the video; those frames carry ``valid=False`` and a consumer must drop
  any pair with an invalid endpoint rather than treat the identity as a still camera
  (report.md P4).


WHAT THIS MODULE REFUSES
------------------------
Gate B is a **staged AND** over conditions computable by the estimator alone, before
any fit.  An *unmeasured* condition does **not** open the gate.

    stage 1   B1 rigidity      disagreement between disjoint image supports / signal < 1
              B2 repeatability signal / floor > 1
    stage 2   B3 per-axis SNR  > 1.5 on ALL THREE axes against a NAMED floor
                               (injection min-detectable where one exists, else the B2
                                repeatability floor; ``splithalf/sqrt(2)`` is PROHIBITED
                                as a floor and is only ever reported for comparison)
    stage 3   B5 contamination three outcomes on the 95 % CI of |corr(camera pitch
                               rate, shark mask-centroid dy)|:
                               PASS = CI-upper < 0.3, ABORT = CI-lower > 0.3,
                               INCONCLUSIVE otherwise

**B4** (camera share of fitted root-rate variance) is a **diagnostic only**, emitted at
raw / 9-frame / 0.5 s windows.  Its input is a completed fit, it moves by an order of
magnitude across those windows, and it sits at its own circular-shift null; it is
never a gate condition and never a ship-rule check.

B5 is **anti-monotone in estimator quality** -- white noise at the same RMS passes it --
so it is valid only *downstream* of B1/B2, which is why the stages are ordered.

Diagnosis and refusal only.  Nothing here runs or modifies a fit.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
DEFAULT_DILATION = 0.04
SELF_TEST_RATES_DEG_S = (5.0, 20.0)
SELF_TEST_GAIN_RANGE = (0.99, 1.01)
_Z95 = 1.959963984540054

def _cv2():
    ...

def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation minimising ``sum |R a_i - b_i|^2`` for unit bearings ``a``, ``b``."""
    ...

def bearings(pts: np.ndarray, Kinv: np.ndarray) -> np.ndarray:
    """Pixel coordinates -> unit bearings in the camera frame."""
    ...

def project(b: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unit bearings -> pixel coordinates."""
    ...

def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (radians), OpenCV convention."""
    ...

def rotmat(r: Sequence[float]) -> np.ndarray:
    """Rotation vector (radians) -> rotation matrix."""
    ...

def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    ...

def camera_rates_deg_s(dR: np.ndarray, fps: float) -> np.ndarray:
    """Scene rotation ``dR`` for one frame pair -> camera-in-world rate, deg/s.

    ``r_cam = -rotvec(dR)``: the camera turns opposite to the scene.  Components are
    ``(pitch, yaw, roll)`` = rotvec ``(x, y, z)``.
    """
    ...

def integrate_R_cw(dRs: Sequence[Optional[np.ndarray]]) -> np.ndarray:
    """Per-pair scene rotations -> ``R_cw`` per frame, with ``R_cw[0] = I``.

    ``R_cw(k+1) = dR_k @ R_cw(k)``.  A ``None`` pair (no valid fit) is integrated as
    the identity so the frame index stays aligned with the video; the caller marks
    those frames invalid.  See the module docstring: the identity there is a
    placeholder, never a measurement.
    """
    ...

def fit_rotation(p0: np.ndarray, p1: np.ndarray, K: np.ndarray, Kinv: np.ndarray, thresh_px: float, iters: int, sample: int, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    """RANSAC rotation-only fit on bearings, Kabsch refit on the inliers.

    Returns ``None`` when there are too few correspondences or the fit degenerates.
    The returned ``R`` is the **scene** rotation ``dR`` (frame-k bearings -> frame-k+1).
    """
    ...

def recover_injected_rotation(p0: np.ndarray, p1: np.ndarray, R_inject: np.ndarray, K: np.ndarray, Kinv: np.ndarray, thresh_px: float=2.0, iters: int=300, sample: int=3, seed: int=0) -> Optional[Dict[str, Any]]:
    """Estimator-level injection: rotate the frame-1 bearings by a known ``R_inject``
    and ask the identical RANSAC-Kabsch chain to give it back.

    The true motion of the pair cancels in ``R_recovered @ R_baseline.T``, which must
    equal ``R_inject``.  Used by the synthetic self-test; the clip-level self-test in
    :func:`injection_sweep` injects by warping the image instead, so that it also
    exercises the tracker.
    """
    ...

def _nanmedian(a) -> float:
    """``np.nanmedian`` that returns ``nan`` on an empty or all-NaN input, quietly."""
    ...

def _rms(a: Sequence[float]) -> float:
    ...

@dataclass
class EgoFrame:
    """One frame of the ego track.

    ``R_cw`` is camera-from-world with the world frame = the camera frame of the first
    valid frame (module docstring).  ``sigma_deg`` is the 1-sigma uncertainty of the
    *increment* into this frame, in degrees, from the split-half disagreement of that
    pair's correspondences; frame 0 carries 0.0 because it *defines* the frame.
    ``state`` is ``measured`` / ``unmeasured`` per frame, or ``refused`` for every
    frame once Gate B has refused the clip.
    """
    t_s: float
    R_cw: List[List[float]]
    valid: bool
    sigma_deg: float
    state: str

    def matrix(self) -> np.ndarray:
        ...

@dataclass
class GateCondition:
    """One row of the gate audit: its value, its threshold and its verdict."""
    condition: str
    stage: Optional[int]
    variant: str
    value: Optional[float]
    threshold: Optional[float]
    direction: str
    verdict: str

@dataclass
class GateAudit:
    """The staged Gate B verdict, with every condition's value and threshold."""
    conditions: List[GateCondition]
    stage_verdicts: Dict[str, str]
    fires: bool
    refused_at_stage: Optional[int]

    def by_condition(self, condition: str, variant: Optional[str]=None) -> List[GateCondition]:
        ...

    def table(self) -> List[str]:
        """The audit as markdown rows (header included)."""
        ...

@dataclass
class EgoProvenance:
    """Everything needed to say what produced these numbers."""
    script_version: str
    video: str
    masks: str
    camera_json: Optional[str]
    intrinsics: Dict[str, Any]
    dilation: float
    fps: float
    width: int
    height: int
    working_scale: float
    rng_seed: int
    preset: str

@dataclass
class EgoTrack:
    """The per-frame camera track plus the clip-level gate audit and provenance."""
    frames: List[EgoFrame]
    gate: GateAudit
    provenance: EgoProvenance

    @property
    def state(self) -> str:
        """Clip-level state: ``refused`` when Gate B did not fire."""
        ...

    def R_cw(self) -> np.ndarray:
        ...

    def valid(self) -> np.ndarray:
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EgoTrack':
        ...

    def to_json(self, path) -> Path:
        ...

    @classmethod
    def from_json(cls, path) -> 'EgoTrack':
        ...

def fisher_ci_abs(r: float, n: int) -> Tuple[float, float]:
    """95 % Fisher-z interval on ``|r|``.  ``(nan, nan)`` when undefined."""
    ...

def pearson_with_ci(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Pearson r on the finite overlap, with the Fisher CI on ``|r|``."""
    ...

def gate_b1(left_right: Sequence[float], top_bot: Sequence[float], up_down: Sequence[float], signal_deg_s: float, random_half: Sequence[float]) -> Dict[str, Any]:
    """B1 rigidity: do disjoint image supports agree on one rigid distant scene?

    ``disagreement`` = median over gate-kept pairs of the geodesic angle between two
    rotations refit independently on two disjoint image supports, x fps.
    ``signal`` = ``sqrt(rms_yaw^2 + rms_pitch^2 + rms_roll^2)``.

    The **primary** split is left/right -- both halves span the full frame height, so
    the sub-fits are equally conditioned.  The secondary top/bottom split at 0.12 H is
    the one clip2's own ``conditioning_warning`` calls badly conditioned for roll and
    pitch; it is reported but is not the primary verdict.

    Each region fit sees roughly half the correspondences, so part of any disagreement
    is just the smaller sample.  ``random_half`` is the same statistic computed on two
    RANDOM disjoint halves, i.e. the same noise with the spatial structure removed;
    only the EXCESS over that control is evidence of a non-rigid scene.

    B1 is a scale-free ratio, so it is hardest to pass exactly when the camera is
    quietest.  That is the correct direction for a refusal gate, but it means a small
    absolute disagreement can still FAIL.
    """
    ...

def gate_b2(floor_deg_s: float, signal_deg_s: float, n_probe_pairs: int=0, extra: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """B2 repeatability: is the estimate reproducible above its own re-detection floor?

    ``floor`` is the zero-injection repeatability control: an INDEPENDENTLY detected
    feature set pushed through the identical detect -> LK -> RANSAC-Kabsch chain on the
    same frame pair; the error is the geodesic angle to the carried-track fit, x fps.
    Not a split-half of one inlier set -- that measures scatter inside a set RANSAC has
    already selected for mutual consistency and is optimistic by two orders of magnitude.
    """
    ...

def gate_b3(rate_rms_deg_s: Dict[str, float], named_floors_deg_s: Dict[str, float], floor_sources: Optional[Dict[str, str]]=None, splithalf_floors_deg_s: Optional[Dict[str, float]]=None) -> Dict[str, Any]:
    """B3 per-axis SNR against a NAMED floor, on ALL THREE axes.

    The named floor is the injection min-detectable where one exists, else the B2
    repeatability floor.  ``splithalf/sqrt(2)`` is **prohibited** as a floor -- on
    clip2 it would turn a rate buried under its own repeatability into an SNR of
    35-54 -- and is carried here only so the two can be compared.
    """
    ...

def gate_b5(cam_pitch_rate_deg_s: Sequence[float], mask_frames: Sequence[int], centroid_y: Sequence[float]) -> Dict[str, Any]:
    """B5 contamination, three outcomes.

    Statistic: ``|corr(camera pitch rate, shark mask-centroid dy)|`` over the real
    (stride-``s``) mask gaps.  Pair index ``j`` is the frame pair ``j -> j+1``, so the
    gap ``a -> b`` is spanned by pairs ``a .. b-1``: ``rates[a:b]``.

    PASS = 95 % CI **upper** < 0.3; ABORT = CI **lower** > 0.3; INCONCLUSIVE otherwise.
    A high correlation means the "background" is being dragged by near-field particles
    around the animal, so the camera estimate is partly the animal's own motion.
    """
    ...

def build_gate_audit(b1: Dict[str, Any], b2: Dict[str, Any], b3: Dict[str, Any], b5: Dict[str, Any], b4: Optional[Dict[str, Any]]=None) -> GateAudit:
    """Assemble the staged AND.  An unmeasured condition does not open the gate."""
    ...

def _boxcar(x: np.ndarray, k: int) -> np.ndarray:
    ...

def camera_share_b4(cam_rates_deg_s: np.ndarray, fit_root_rates_deg_s: np.ndarray, fps: float, windows: Optional[Dict[str, int]]=None) -> Dict[str, Any]:
    """B4: the camera's share of the fitted root-rate variance -- DIAGNOSTIC ONLY.

    ``share = sum_a r_a^2 var(omega_fit,a) / sum_a var(omega_fit,a)``, per axis
    correlation weighted by that axis' fitted variance, at raw / 9-frame / 0.5 s
    smoothing.  Both inputs must be in the SAME convention; this function takes the
    camera series in the scene convention ``+rotvec(dR)`` because that is the sign a
    fit which fully absorbed the camera would show as slope +1.

    This is never a gate condition (report.md section 4: its input is a completed fit,
    it moves 0.006 -> 0.039 -> 0.09-0.11 across these three windows against a threshold
    no design document derives, and it sits at its own circular-shift null).
    """
    ...

@dataclass
class EgoConfig:
    """Tracker / RANSAC parameters.  Defaults are the demo's shipped values."""
    fps: float = 59.94
    working_scale: float = 0.5
    focal_frac: float = 0.6
    border_frac: float = 0.03
    max_corners: int = 1200
    quality: float = 0.01
    min_dist_frac: Optional[float] = 0.005
    min_dist_px: Optional[int] = None
    block: int = 7
    lk_win: int = 21
    lk_levels: int = 3
    fb_tol: float = 1.0
    redetect_below: int = 400
    ransac_px: float = 2.0
    ransac_iters: int = 300
    ransac_sample: int = 3
    min_corr: int = 8
    top_band: float = 0.12
    rng_seed: int = 0
    n_probe_pairs: int = 25
    inject_rates_deg_s: Tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
    inject_stride: int = 6

def _lk_kwargs(cfg: EgoConfig) -> Dict[str, Any]:
    ...

def build_allow_mask(mask_bool: np.ndarray, dilation: float, W_full: int, Ww: int, Hw: int, border_px: int) -> np.ndarray:
    """255 where features ARE allowed, at working scale.

    The shark is excluded by dilating its SAM mask by ``dilation * W_full`` pixels;
    ``dilation`` is a parameter because it is the knob the contamination sweep turns
    (4 / 8 / 12 %).  A frame border is excluded too, because LK is unreliable there.
    """
    ...

def track_pairs(grays: Sequence[np.ndarray], allows: Sequence[np.ndarray], K: np.ndarray, Kinv: np.ndarray, W_full: int, H_full: int, cfg: EgoConfig) -> List[Dict[str, Any]]:
    """Carried-track LK + per-pair rotation-only RANSAC, with the region sub-fits.

    One record per frame pair.  ``R`` is the scene rotation ``dR``; the rates stored
    are camera-in-world (see the module docstring).
    """
    ...

def pair_validity(per: List[Dict[str, Any]], cfg: EgoConfig) -> np.ndarray:
    """Per-pair validity: a finite fit, and the quality gate where the clip has one."""
    ...

def repeatability_floor(grays: Sequence[np.ndarray], allows: Sequence[np.ndarray], K: np.ndarray, Kinv: np.ndarray, W_full: int, base_R: Sequence[Optional[Any]], cfg: EgoConfig) -> Dict[str, Any]:
    """The B2 floor: re-detect an INDEPENDENT feature set on the same frame pair and
    re-run the identical chain; the geodesic disagreement with the carried-track fit
    is the realistic error bar.

    (A zero-injection control with ``R_test = I``: the warp ``H = K I K^-1`` is the
    identity to 1e-13 and ``warpPerspective`` returns the input bit-identically, so no
    resampling penalty is involved -- the floor is pure re-detection scatter.)
    """
    ...

def injection_sweep(grays: Sequence[np.ndarray], allows: Sequence[np.ndarray], K: np.ndarray, Kinv: np.ndarray, W_full: int, cfg: EgoConfig) -> Dict[str, Any]:
    """Inject a KNOWN camera rotation into the second frame of real pairs and ask the
    estimator to give it back.

    A textureless region gives LK no gradient, so it returns the point UNMOVED and the
    forward-backward check passes trivially: a near-zero measured rate is therefore NOT
    evidence of a static camera.  Injection is the only way to settle observability.

    Two things come out of this:

    * the **self-test** -- yaw at :data:`SELF_TEST_RATES_DEG_S` must be recovered with
      gain inside :data:`SELF_TEST_GAIN_RANGE`, or the estimator is not trusted on this
      footage;
    * the **min-detectable** per axis -- the smallest node of ``cfg.inject_rates_deg_s``
      whose gain exceeds 0.5, which names B3's floor where it exists.  A grid can only
      ever bound this from above, so every SNR against it is a LOWER bound.
    """
    ...

def named_floors(inject_summary: Optional[Dict[str, Any]], repeatability_deg_s: float) -> Tuple[Dict[str, float], Dict[str, str]]:
    """B3's floors: the injection min-detectable where one exists, else B2's floor.

    With one qualification the two shipped scripts left implicit.  The injection
    min-detectable is a statement about the MEDIAN over many probe pairs -- an injected
    rate can shift that median while being invisible in any single pair.  B3's numerator
    is the RMS of a PER-PAIR series, so a floor below the estimator's own per-pair
    repeatability is not a floor for it.  The named floor is therefore

        ``max(injection min-detectable, B2 repeatability floor)``

    which on the demo leaves the injection numbers standing (yaw 1.0, roll 2.0 deg/s,
    both far above its 0.153 deg/s repeatability) and on clip2 discards them (yaw and
    roll recover in the median at 0.5 deg/s, but clip2's per-pair repeatability is
    6.387 deg/s, so 6.387 is the floor on all three axes -- which is what its own
    injection control concluded by a different route).

    ``splithalf/sqrt(2)`` is never a candidate: see :func:`gate_b3`.
    """
    ...

def estimate_from_frames(grays: Sequence[np.ndarray], allows: Sequence[np.ndarray], K: np.ndarray, W_full: int, H_full: int, cfg: EgoConfig, mask_frames: Optional[Sequence[int]]=None, centroid_y: Optional[Sequence[float]]=None, provenance: Optional[EgoProvenance]=None, do_injection: bool=True, do_repeatability: bool=True) -> EgoTrack:
    """Run the estimator and the staged gate on already-decoded working-scale frames."""
    ...

def decode_gray(video, working_scale: float=1.0) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Grayscale frames at the estimator's working scale, plus the full-res shape."""
    ...

def load_mask_paths(mask_dir, mask_index=None) -> Dict[int, Path]:
    """{full-res frame index: mask path} from an index.json or a mask_*.png glob."""
    ...

def read_mask(path, shape: Tuple[int, int]) -> np.ndarray:
    ...
__all__ = ['SCRIPT_VERSION', 'DEFAULT_DILATION', 'GATE_THRESHOLDS', 'SELF_TEST_RATES_DEG_S', 'SELF_TEST_GAIN_RANGE', 'PRESETS', 'EgoFrame', 'EgoTrack', 'EgoConfig', 'EgoProvenance', 'GateAudit', 'GateCondition', 'kabsch', 'bearings', 'project', 'rotvec', 'rotmat', 'geodesic_deg', 'camera_rates_deg_s', 'integrate_R_cw', 'fit_rotation', 'recover_injected_rotation', 'fisher_ci_abs', 'pearson_with_ci', 'gate_b1', 'gate_b2', 'gate_b3', 'gate_b5', 'build_gate_audit', 'camera_share_b4', 'build_allow_mask', 'track_pairs', 'pair_validity', 'repeatability_floor', 'injection_sweep', 'named_floors', 'estimate_from_frames', 'decode_gray', 'load_mask_paths', 'read_mask']
