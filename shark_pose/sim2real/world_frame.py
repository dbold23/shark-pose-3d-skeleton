"""The world-frame record contract: derived attitude, rates, heading and scale.

Plan items **P5**, **P6** and **P7** of ``outputs/demo_2026-09-03/work/egomotion/
report.md`` section 4:

    P5  Record contract in ``fit_video.py``: per-frame ``t_s``, ``R_cw``,
        ``ego_valid``, ``ego_sigma_deg``, ``camera_provenance``, ``g_cam: null``,
        quality byte; derived ``R_wb``, ``g_b`` (null), ``omega_b`` +
        ``omega_b_raw``, ``psi``, ``psi_dot``, ``x_w: null``; clip level
        ``frames_convention``, ``ego{}``, ``ego_diagnostics{}``, ``gravity{}``,
        ``scale{}``, ``time{}``, ``attitude_quality{}`` and ``consumer_needs{}``
    P6  Metric scale from a NAMED source, ``null`` when the clip has none, with
        the range cross-check written as an interval over the focal candidates
    P7  De-noised attitude channel: ``omega_b`` from a 0.5 s zero-phase boxcar of
        ``omega_b_raw``, window AND edge convention stamped.  The objective is
        untouched -- nothing here changes a weight or a loss term.

This module is the pure arithmetic; ``scripts/fit_video.py`` owns the plumbing.
It imports numpy only, so the record contract is testable without torch.


FRAMES AND CONVENTIONS
----------------------
``R_cw``  camera-from-world, ``X_cam = R_cw @ X_world``.  Supplied by
          :mod:`shark_pose.sim2real.egomotion`, whose world frame is the camera
          frame of the clip's first valid frame, so ``R_cw[first_valid] = I`` by
          construction.  It is a gauge choice: no gravity, no north, no scale.
          ``R_cw = I`` when the ego-motion is **unmeasured** is a placeholder and
          the record says so in ``ego_state`` -- it is never a measurement.

``R_cb``  camera-from-body, ``X_cam = R_cb @ X_body + t``.  This is the fit's own
          projection maths: ``SpineSMPLify._pose`` writes
          ``pose[:, root_joint] = compose_axis_angle(root_orient, prior_recoil)``
          and ``SharkSMPL`` rotates the whole chain by that entry, so **the body
          attitude is ``pose[root_joint]``, not ``root_orient``** -- the latter
          omits the prior's own root recoil (measured on the shipped demo fit:
          up to 0.056 deg).  ``R_cb``'s columns are the body axes in camera
          coordinates: 0 lateral, 1 longitudinal (snout-forward), 2 dorsal
          (``SpineSMPLify._upright_loss``).

``R_wb``  world-from-body, ``R_wb = R_cw.T @ R_cb``.  The composition test
          ``R_cw @ R_wb @ X_body + t == X_cam`` is what
          ``tests/test_world_frame.py`` asserts against the model's own forward
          pass, to 1e-6.

``omega_b``  body-frame angular rate, deg/s, from ``R_wb(k).T @ R_wb(k+1)``.  A
          gyro reads body axes, so this is the channel a tag consumer wants; with
          ``R_cw = I`` it degenerates to the camera-frame body rate, which is the
          honest statement of what one view can give.

``psi``   heading about the **world up-axis PROXY**, which is
          ``WORLD_UP_PROXY`` = camera ``-y`` of the first valid frame (OpenCV).
          It is **not gravity**: no gravity cue survived falsification on either
          clip (report.md F1), so ``g_cam`` and ``g_b`` ship ``null`` and every
          heading is relative within a clip and NOT comparable between clips.
"""
from __future__ import annotations
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
WORLD_UP_PROXY = (0.0, -1.0, 0.0)
SMOOTHING_WINDOW_S = 0.5
_FT_PER_M = 0.3048

def rodrigues(axis_angle: Sequence[float]) -> np.ndarray:
    """(3,) axis-angle -> (3, 3) rotation matrix."""
    ...

def rotvec(R: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> (3,) axis-angle."""
    ...

def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic (angle) distance between two rotations, degrees."""
    ...

def body_attitude(pose_root_aa: Sequence[Sequence[float]], R_cw: Optional[np.ndarray]=None) -> np.ndarray:
    """(T, 3, 3) ``R_wb = R_cw.T @ R_cb`` from the fit's ``pose[root_joint]``.

    ``R_cw`` is (T, 3, 3) or None (== identity, i.e. the world frame IS the
    camera frame and the record must stamp ``ego_state: unmeasured``).
    """
    ...

def body_rates_deg_s(R_wb: np.ndarray, fps: float) -> np.ndarray:
    """(T, 3) body-frame angular rate in deg/s -- ``omega_b_raw`` (P5).

    ``omega_fwd(k) = rotvec(R_wb(k).T @ R_wb(k+1)) * fps`` is the rate expressed
    in frame ``k``'s own body axes.  Reported per FRAME as the average of the
    forward and backward increments where both exist (a central difference), and
    the single available one-sided increment at each end.  Nothing is
    extrapolated and no frame is repeated.
    """
    ...

def heading_deg(R_wb: np.ndarray, up: Sequence[float]=WORLD_UP_PROXY) -> np.ndarray:
    """(T,) unwrapped heading ``psi`` in degrees about the world up PROXY.

    The body's longitudinal axis (column 1 of ``R_wb``) is projected onto the
    plane orthogonal to ``up`` and its angle taken in that plane, measured from
    the first in-plane basis vector toward the second.  With the default
    ``up = (0, -1, 0)`` those are world ``+z`` (camera forward) and world ``+x``
    (camera right), so ``psi = atan2(u_x, u_z)``: 0 deg = swimming away from the
    first frame's camera, +90 deg = swimming to its right -- a compass-like
    heading seen from above the up proxy.
    """
    ...

def central_rate(x: Sequence[float], fps: float) -> np.ndarray:
    """(T,) per-frame derivative of a scalar series, same scheme as omega_b_raw."""
    ...

def boxcar_window_frames(fps: float, window_s: float=SMOOTHING_WINDOW_S) -> int:
    """Odd frame count of the P7 boxcar at ``fps`` (odd, so it is zero-phase)."""
    ...

def boxcar_zero_phase(x: np.ndarray, k: int):
    """Centred boxcar with the ``shrink_symmetric`` edge rule (P7).

    Returns ``(smoothed, n_used)``: the smoothed array and, per sample, how many
    frames its average was taken over -- which is what makes the edge convention
    auditable rather than merely stated.  NaNs are excluded from every average.
    """
    ...

def quality_byte(detected: bool, centerline_gate: bool, ego_valid: bool) -> int:
    """bit 0 detected | bit 1 centreline-gate pass | bit 2 ego-valid."""
    ...

def sideslip_deg(translation: Sequence[Sequence[float]], body_long_axis: Sequence[Sequence[float]], fps: float, camera_rate_deg_s: Optional[Sequence[Sequence[float]]]=None) -> np.ndarray:
    """(T-1,) angle between the animal's velocity and its own +snout axis, deg.

    ``translation`` (T, 3) is the fitted root position and ``body_long_axis``
    (T, 3) its longitudinal axis, both in the SAME frame -- column 1 of
    ``R_cb`` in the camera frame, of ``R_wb`` in the world frame.  The axis is
    taken at the midpoint of each pair, exactly as ``fits_extract.py`` does it.

    **claim_2** (``design/camera_leaves_identifiability.json``): the world-frame
    sideslip equals this camera-frame expression, because both arguments are
    ``R_cw.T`` times a camera-frame vector and a common rotation preserves
    angles.  The camera's *attitude* therefore cannot enter at all; its *rate*
    enters only through ``omega x x``, which is what ``camera_rate_deg_s``
    (T-1, 3, deg/s) adds when given.

    **The sign is the estimator's convention, and it is a plus.**  With
    ``R_cw(k+1) = dR R_cw(k)`` and the camera-in-world rate
    ``omega = -rotvec(dR) * fps`` that :func:`egomotion.camera_rates_deg_s`
    reports, differentiating ``x_w = R_cw.T x_c`` gives
    ``v_w = R_cw.T (xdot_c + omega x x_c)``.  report.md section 3(b) writes the
    same identity as ``xdot_c - omega_c x x_c`` for an ``omega_c`` of the
    opposite (scene) sense; feeding this function a rate in that sense makes the
    correction worse, not better, which
    ``tests/test_identifiability.py::test_claim_2_the_camera_rate_enters_only_as_omega_cross_x``
    measures.  Pairs with no measurable speed are NaN, never 0.
    """
    ...

def implied_pan_deg_s(translation: Sequence[Sequence[float]], fps: float) -> np.ndarray:
    """(T-1,) the pure camera pan that would produce the fitted tangential motion.

    A camera pan moves the animal on a sphere about the camera, so the yaw rate
    needed to explain a tangential velocity ``v_tan`` at range ``r`` is
    ``v_tan / r``.  Ship rule 13 turns that into a MODEL-FREE falsifier: such a
    pan would drag the whole background across the frame by ``f * omega / fps``
    px per frame, which is directly observable.
    """
    ...
PLAUSIBLE_LENGTH_M = (0.5, 8.0)

def parse_length_label(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """A length field in a file NAME, e.g. ``FAR18101701_12ft_U.mp4`` -> 3.6576 m.

    Returns ``None`` when the name carries no length field.  ``FAR22101703_NOID.mov``
    carries none, and a scale is **never borrowed from another clip**.
    """
    ...
HALF_NORMAL_MEAN_ABS_OVER_SIGMA = 0.7979
HALF_NORMAL_SD_OVER_SIGMA = 0.6028
MAY2019_CALIBRATION_RANGE_M = (2.41, 4.7)
SCALE_CI_SIGMA_MULTIPLE = 1.0
LABEL_SPREAD_CONTESTED_FRAC = 0.05
LABEL_VALUE_RESOLUTION_M = 1e-06

def clip_meta_label(clip_meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The resolved length label a ``clip_meta.json`` carries, validated.

    ``None`` when the sidecar carries no label field at all -- which is every
    sidecar written before this contract, and is why a record written without
    one is byte-identical to a record written before it existed.

    Raises ``ValueError`` on a MALFORMED label: a value with no ``label_source``,
    an unknown ``label_state``, a non-numeric value.  A length with no named
    source is precisely the un-provenanced scale the label ladder exists to
    refuse, so it is refused loudly here rather than quietly downgraded.
    """
    ...

def clip_meta_pass(clip_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The recording PASS a ``clip_meta.json`` names, for :func:`time_block`.

    Returns ``{pass_id, creation_time, creation_time_convention, pass_source}``
    with every value ``None`` when the sidecar names no pass -- which
    :func:`time_block` reads as "stamp nothing", so a window prepared without
    the contract writes the time block it always wrote.

    ``creation_time_utc`` is validated as ISO-8601 here rather than at read
    time: ``aggregate._creation_time_s`` silently returns ``None`` on a string
    it cannot parse, and a silent ``None`` there is a clause-6 fallback nobody
    ordered.
    """
    ...

def observer_sigma_cm(observer_class: str) -> Dict[str, Any]:
    """The May et al. sigma of one observer class, as a range plus a point.

    ``sigma_cm`` is the MAE inversion -- the number the design's ladder is
    priced at, and the NARROW end for every class -- and ``sigma_cm_range`` is
    the interval the two moment inversions span.  ``sigma_cm_model_free`` is
    ``sqrt(MAE^2 + SD^2)``, which assumes neither zero mean nor a half-normal.
    """
    ...

def observer_class_of(observer: Optional[str]) -> tuple:
    """``(class, why)`` for one label row's recorded observer.

    **The ``observer`` column is the record's ``user`` field** -- the data-entry
    account (508 of 514 field-sheet rows read ``DM``, constant within every one
    of the 28 files) -- and NOT a field observer.  ``survey.crew`` is pasted
    into a free-text note and is never read.  So ``scientist`` here means "the
    row was entered by a named member of the lab", ``crew`` means "no name is
    recorded at all", and the class is published with that basis attached.
    """
    ...

def label_row(value_m: float, source: str, observer: Optional[str]=None, crew_recorded: Optional[str]=None, obs_date: Optional[str]=None, source_path: Optional[str]=None, raw: Optional[str]=None, contested: Optional[str]=None) -> Dict[str, Any]:
    """One length-label row, in the shape :func:`scale_resolve` consumes.

    ``contested`` is a REFUSAL the row carries with it: the reason an upstream
    authority -- the ledger -- already declined to publish an absolute for this
    individual.  :func:`scale_resolve` honours it whatever the pooled values
    happen to do, which is what makes the verdict survive EVERY rebuild path
    (the clip's own, ``resolve_individual_scale``'s, ``_fuse_scale``'s) instead
    of being re-derived from whichever rows are in hand.  Without it a
    contested individual whose sibling window carries a readable filename token
    republished a bounded absolute at the token's value -- the very number the
    ledger refused (review, 2026-09-14).

    The key is OMITTED when there is no refusal, so a row built without it is
    byte-identical to the row this function returned before it existed.
    """
    ...

def scale_resolve(rows: Sequence[Dict[str, Any]], ci_sigma_multiple: float=SCALE_CI_SIGMA_MULTIPLE, prefer_source: Optional[str]=None, sigma_cm_override: Optional[float]=None) -> Dict[str, Any]:
    """Which length label is THE label for this individual, and its band.

    The rule, in order (spec 4.6), replacing "the first clip that carried one":

    1. drop every quarantined source (``catalog_size_ft``, always) and every
       implausible value;
    2. group what is left by ``independence_group`` -- **five in-corpus sources
       are one observation**, so N sources is not N measurements;
    3. within the best-ranked group take the median of the DISTINCT values;
    4. spread over all distinct non-quarantined values above
       ``LABEL_SPREAD_CONTESTED_FRAC`` -> ``contested``, no value, and the
       individual drops to the ratio lane;
    5. the band is ``sigma_cm(observer_class) / (value_m * 100)`` -- PER
       INDIVIDUAL, because May's error is a length in centimetres -- reported at
       1 and 2 sigma as named scalars, with ONE of them, named by
       ``ci_sigma_multiple``, in ``uncertainty_frac``.

    Returns a block whose ``state`` is ``resolved``, ``contested`` or ``unmet``.
    ``uncertainty_frac`` is symmetric (``[f, f]``): the propagator takes
    ``max()`` of it, so both ends must be the multiple this block declares.

    ``prefer_source`` breaks a tie INSIDE the chosen group: step 3 takes the
    median of the group's distinct values, which on two disagreeing rows is
    their mean -- a number neither source said.  Naming a source makes that
    source's median the value (the boxfix rule: when the file name's token and
    the ledger's resolved label disagree under the contested bar, the LEDGER's
    is adopted, because it is the reconciliation of every source the individual
    has and the token is one of its inputs).  The band is unaffected: the
    disagreement is still priced by ``label_spread_widen_frac``.

    ``sigma_cm_override`` replaces the May et al. class sigma with a band the
    caller already holds (the ledger's ``scale_sigma_band``).  Both default to
    ``None``, and with both ``None`` this function returns exactly what it
    returned before they existed -- every key, every value.
    """
    ...

def _meta_source_path(meta: Dict[str, Any]) -> str:
    """WHERE the sidecar's label came from, for :func:`label_row`.

    The sidecar's own ``label_source_path`` citation when it carries one (the
    ledger ``file:line`` the prep already computes), and otherwise the FIELD
    NAME, which is what this route published unconditionally before: a row
    saying ``clip_meta.length_label_m`` cannot be traced to a sidecar, a ledger
    build or a row, where the filename route's row names the video it read.
    """
    ...

def _label_provenance(label: Optional[Dict[str, Any]], meta: Dict[str, Any], source_name: Optional[str]) -> Dict[str, Any]:
    """Reconcile the file name's token with a ``clip_meta`` label (boxfix).

    THE RULE, and the whole of it:

    * the token alone -> the token, exactly as before this existed;
    * the clip_meta label alone -> the clip_meta label;
    * both, AGREEING (to ``LABEL_VALUE_RESOLUTION_M``) -> the token wins, and
      both rows are still recorded so the agreement is visible;
    * both, DISAGREEING -> both are recorded and the LEDGER's is adopted,
      because it is the reconciliation of every source this individual has and
      the token is one of its inputs.  The band is widened by half the
      disagreement (``scale_resolve``'s ``label_spread_widen_frac``), and above
      ``LABEL_SPREAD_CONTESTED_FRAC`` the individual is ``contested``: no
      absolute, ratio lane, both labels listed rather than averaged.

    Returns the block stamped as ``scale.label_provenance``, with ``rows`` (what
    to hand :func:`scale_resolve`) and ``verdict`` (a short circuit, or None).
    """
    ...

def scale_block(source_name: Optional[str], mesh_span_model_units: Optional[float], tz_median_model_units: Optional[float]=None, focal_candidates_px: Optional[Dict[str, float]]=None, fit_focal_px: Optional[float]=None, label_rows: Optional[Sequence[Dict[str, Any]]]=None, ci_sigma_multiple: float=SCALE_CI_SIGMA_MULTIPLE, clip_meta: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """The ``scale{}`` block: model units -> metres, from a NAMED source (P6).

    ``mesh_span_model_units`` is the fitted body's snout-to-caudal-tip extent of
    the MESH at the fit's own betas -- the same thing the filename's length field
    measures.  The design's route was the 31-joint span times a constant
    ``mesh_over_joint_span`` (report.md P6: 1.0445 -> 0.8084 m/unit on the demo);
    measuring the mesh directly removes the constant and is the same quantity.
    Only the mesh route is reported in the block; on the demo it delivers
    0.8049 m/unit, 0.43 % below the design figure, because the shipped asset's
    rest 31-joint span is 4.3076 model units, not the 4.3509 the design assumed.

    When the name carries no length field the block is **UNMET with a reason** --
    never a bare ``null`` that a consumer could read as satisfied.

    ``label_rows`` widens the source of the label beyond the file name.  The
    filename token is the only route the shipped path ever had, and it is the
    scale of **111 of 111** fit-capable individuals -- which makes the whole
    named-observer rung of the ladder unreachable in code, because a field-sheet
    or FinID row lives on no file name.  Pass the individual's label rows (see
    :func:`label_row`) and they are reconciled by :func:`scale_resolve`, which
    also quarantines ``catalog_size_ft``, refuses an individual whose own
    sources disagree, and bands the label PER INDIVIDUAL off May et al.'s
    centimetre sigma.  With ``label_rows=None`` the file name's token is the one
    row, so the filename path is the general path with n = 1.

    The band this block publishes is therefore no longer the uncited
    ``[0.10, 0.20]`` (a flat +-20 % once ``max()`` is taken of it, i.e. a x3.375
    volume factor that no derivation exists for anywhere in either repo).  It is
    ``sigma_cm / label_m`` at the multiple ``ci_sigma_multiple`` names, and
    ``label_resolution`` carries the named scalars beside it.

    ``clip_meta`` is the window's ``clip_meta.json``, and when it carries the
    label fields (:data:`CLIP_META_LABEL_FIELDS`) it becomes a SECOND route to a
    label, reconciled with the file name's token by :func:`_label_provenance`
    and stamped as ``label_provenance``.  It exists because
    ``parse_length_label`` reads a file NAME and about a third of this corpus's
    names carry a unit-less number it cannot read, so those windows published
    ``unmet`` -- and no absolute length, volume or mass -- for individuals whose
    label the ledger resolves.  With no clip_meta label the key is not written
    and this function returns exactly the block it returned before the contract
    existed: a record whose filename token parses is byte-identical.
    """
    ...

def time_block(fps: float, n_frames: int, t0_frame: Optional[int], t0_source: str, pass_id: Optional[str]=None, creation_time: Optional[str]=None, creation_time_convention: Optional[str]=None, pass_source: Optional[str]=None) -> Dict[str, Any]:
    """``time{}``: fps, the window's origin in the SOURCE video, and its source.

    ``t_s`` is ``(t0_frame + i) / fps``.  When the window offset into the source
    video is not recorded anywhere the pipeline can read, ``t0_frame`` is 0 and
    the state is ``clip_local`` -- the timestamps are then correct WITHIN the
    clip and are not absolute in the source video.  That is stated, never hidden.

    **THE PASS.**  ``pass_id`` / ``creation_time`` / ``creation_time_convention``
    are the window's RECORDING pass, and this block is the only place they reach
    the record: ``aggregate._clip_independence`` (gate clause 6) has read
    ``summary.time.pass_id`` and ``summary.time.creation_time`` since it was
    written, and nothing ever wrote them, so the clause fell back to "different
    video" on every record in the corpus.  Feed them from
    :func:`clip_meta_pass`.

    The three keys are stamped ONLY when a pass is actually named, so a window
    prepared without them writes the block it always wrote, byte for byte, and
    clause 6 sees the same absence it has always seen -- which it publishes as
    ``clause6_basis = "different_video_only"`` rather than as a pass test.
    """
    ...

def stamp_pass(pass_id: Optional[str], creation_time: Optional[str], creation_time_convention: Optional[str], pass_source: Optional[str]=None) -> Dict[str, Any]:
    """The four ``time{}`` keys gate clause 6 reads, as one block.

    Separate from :func:`time_block` so an already-fitted record can be given a
    pass it was written without -- the pass is a property of the SIDECAR, not of
    the fit (see :func:`repair_summary_from_clip_meta`).
    """
    ...

def repair_summary_from_clip_meta(summary: Dict[str, Any], clip_meta: Optional[Dict[str, Any]], tz_median_model_units: Optional[float]=None, focal_candidates_px: Optional[Dict[str, float]]=None, fit_focal_px: Optional[float]=None, clip_meta_path: Optional[str]=None) -> Dict[str, Any]:
    """Give an ALREADY-FITTED record the label and the pass its sidecar lacked.

    NO REFIT.  Both defects the boxfix repairs are properties of
    ``clip_meta.json``, and every quantity the two blocks need is already in the
    record: ``scale.source_name``, ``scale.mesh_span_model_units``, and the
    depth and focal candidates ``scale.range_cross_check`` stores.  So the fixed
    blocks are a pure function of (this summary, this sidecar), and the 97 box
    records are repairable at aggregation.

    ONE input is not recoverable from an UNMET block: ``scale_block`` returns
    before it computes ``range_cross_check``, so a record that published no
    scale never stored the depth and the focal candidates that check is built
    from.  Pass them (``tz_median_model_units`` is the median of the record's
    own ``frames[i].camera[2]``) to rebuild it; leave them out and the repaired
    block carries ``range_cross_check: null``, which is what it already
    carried.  The report says which happened.  The check is a published
    NON-DISCRIMINATING diagnostic either way (report.md G2/C14).

    Mutates ``summary`` in place and returns a REPORT of what changed -- the old
    and new scale state and metres-per-model-unit, and the pass that was
    stamped.  A sidecar that carries neither a label nor a pass changes nothing
    and the report says so.

    ``clip_meta_path`` is the sidecar this is being rebuilt FROM, and it is
    stamped into the repaired block (``scale.rebuilt_by``) beside the marker the
    time block has always carried in ``time.pass_source``.  The report is not
    the record: a caller that drops it must still leave a record that says a
    human-readable where-from.
    """
    ...

def require_frames_convention(summary: Dict[str, Any]) -> str:
    """Reader side of graft G4: REFUSE a record whose convention is not known.

    A consumer that guesses which frame ``pose`` / ``camera`` / ``joints_3d`` live
    in is the failure this stamp exists to prevent, so an absent or unrecognised
    ``frames_convention`` raises rather than defaulting.
    """
    ...

def gravity_block() -> Dict[str, Any]:
    """``gravity{}``: null, with the falsification that put it there (F1)."""
    ...

def attitude_quality_block(omega_raw: np.ndarray, omega_smooth: np.ndarray, window_s: float, window_frames: int, n_used: Optional[np.ndarray]=None) -> Dict[str, Any]:
    """``attitude_quality{}``: what the P7 smoother removed, per body axis.

    The jitter fraction is ``1 - var(smoothed) / var(raw)`` per axis: the share of
    the frame-rate signal the 0.5 s boxcar removes.  On the shipped fits that is
    87.5 % (demo) / 55.6 % (clip2) of ``root_orient`` -- a fit-conditioning
    problem this channel REPORTS AROUND and does not solve.
    """
    ...

def consumer_needs_block(scale: Dict[str, Any], ego: Dict[str, Any], time: Dict[str, Any]) -> Dict[str, Any]:
    """The six downstream needs of ``egomotion/audit/audit.md`` section 3, each
    with met / UNMET-and-why for THIS clip -- never a bare null (Gate A).
    """
    ...
