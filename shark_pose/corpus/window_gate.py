"""The corpus scan's window trust gate, with the view-conditional trunk gate.

The six-clause gate below is the one the complete scan ran
(``outputs/demo_2026-09-03/work/individuals/scan_complete/src/verify_complete.py``);
this module is its single shared implementation so the scan, the recount and the
tests cannot drift apart.

The one addition is the **view-conditional trunk gate**, which is DEFAULT OFF.
When ``enable_trunk_gate=True`` and only for windows whose ``view_class`` is
``"oblique"``, the whole-animal clause

    frac_both_ends >= 0.5                          ("animal_not_wholly_in_frame")

becomes the DISJUNCTION

    frac_both_ends >= 0.5  OR  frac_trunk_bracket >= 0.80

i.e. the window is admitted either because snout and caudal notch are in frame
(the original clause) or because the pectoral-base, first-dorsal-base and pelvic
stations are all detected and comfortably inside the frame in at least 80 % of
the scored frames.  The motivation (S0 lane ``oblique``) is that an obliquely
viewed animal is routinely rejected for a snout or a caudal tip that has left the
frame while its entire measurable trunk is present and sharp.

The form is a disjunction, not a replacement, so the change can only ADD trust:
pre-registration clause 3 ("all 62 currently-trusted oblique windows must remain
trusted") then holds by construction, and it was measured to hold on the 29 of
those 62 that an independent 5-frame re-detection reproduced as trusted (29/29).

A window admitted only by the trunk arm carries ``window_kind == "TRUNK_ONLY"``:
downstream fits must use trunk stations only and must never quote a total length
from it.

Two measured departures from the S0 proposal (S1 lane ``oblique_redetect``,
2026-09-15; report S3):

* **The residual stays whole-body.**  S0 proposed recomputing ``fit_rms_over_L``
  over trunk stations only.  Measured on the 258 rejected oblique windows, the
  trunk-only residual is WORSE, not better -- median 0.199 against 0.163, because
  dropping snout and caudal shortens the lever arm of the along-axis regression
  -- and it turns ``incoherent_landmarks`` from 142/258 into 221/258.  The trunk
  arm therefore reads the whole-body residual and length.  ``fit_rms_trunk`` /
  ``L_trunk_native_median`` are still accepted and used when
  ``trunk_residual=True``, which is off by default.
* **The bracket threshold is 0.80, not 0.50.**  At 0.50 the newly admitted
  windows were separable from currently-trusted oblique windows on trunk
  completeness alone at 0.734, above the 0.70 bar that S0's falsifier 5 set; 0.80
  brings that to 0.603.  This is the one revision that pre-registration clause 6
  permits, and it was taken through falsifier 5 exactly as written.

Nothing here changes for broadside or intermediate windows under any flag
setting: the branch is guarded on ``view_class == "oblique"``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

def trunk_frame_ok(keypoints: np.ndarray, conf: np.ndarray, width: float, height: float, conf_ge: Optional[float]=None, border_frac: Optional[float]=None) -> bool:
    """True when every trunk station is detected and inside the frame margin.

    ``keypoints`` is (16, 2) in the SAME pixel frame as ``width``/``height``.
    """
    ...

def trunk_bracket_fraction(keypoints: Sequence[np.ndarray], conf: Sequence[np.ndarray], width: float, height: float, **kw) -> float:
    """Fraction of the supplied (scored) frames whose trunk bracket is complete."""
    ...

@dataclass
class WindowStats:
    """Per-window medians/fractions over the window's scored frames."""
    n_detected: int
    fit_rms_over_L: Optional[float]
    L_px_native_median: Optional[float]
    frac_both_ends: float
    frac_multi_det: float
    mean_visible_kp: Optional[float]
    frac_trunk_bracket: Optional[float] = None
    fit_rms_trunk: Optional[float] = None
    L_trunk_native_median: Optional[float] = None

def gate_failures(s: WindowStats, *, enable_trunk_gate: bool=False, trunk_residual: bool=False, thresholds: Optional[Dict[str, float]]=None, trunk: Optional[Dict[str, float]]=None) -> Tuple[list, str]:
    """Return ``(failures, window_kind)`` for one window.

    With ``enable_trunk_gate=False`` (the default) this reproduces the shipped
    six-clause gate exactly, for every view class.

    ``trunk_residual`` (off by default) additionally reads the residual and the
    length from the trunk-only axis fit when the trunk arm carries the window.
    It is off because the trunk-only residual measured WORSE than the whole-body
    one on the 258 rejected oblique windows (median 0.199 vs 0.163); the option
    exists so that the measurement can be repeated rather than re-argued.
    """
    ...

@dataclass
class FrameStats:
    """One scored frame, in the quantities the six clauses read.

    ``fit_rms_over_L`` / ``L_px_native`` come from the scan's own along-axis cue
    (``corpus_mine/verify/rescore.my_cue``); ``trunk_ok`` from
    :func:`trunk_frame_ok`.  A frame with no detection is NOT a FrameStats -- it
    is simply absent, which is what makes a block "consecutive SCORED frames".
    """
    n_det: int
    n_visible_kp: int
    both_ends: bool
    fit_rms_over_L: Optional[float] = None
    L_px_native: Optional[float] = None
    trunk_ok: bool = False

def aggregate_frames(frames: Sequence[FrameStats], view_class: str='broadside') -> Optional[WindowStats]:
    """Medians/fractions over a set of scored frames -> the gate's WindowStats."""
    ...

def subwindow_scan(frames: Sequence[FrameStats], view_class: str='broadside', *, enable_trunk_gate: bool=False, trunk_residual: bool=False, thresholds: Optional[Dict[str, float]]=None, trunk: Optional[Dict[str, float]]=None, subwindow: Optional[Dict[str, float]]=None, frame_numbers: Optional[Sequence[int]]=None, window_end_frame: Optional[int]=None) -> Dict[str, object]:
    """Slide a block of consecutive scored frames and report how often it passes.

    Returns ``n_runs``, ``n_pass``, ``frac_pass``, ``longest_pass_run``,
    ``best_index`` and ``best_stats`` (the passing block with the largest
    ``L_px_native``; if none passes, the largest-``L`` block).
    """
    ...

def subwindow_pass(frames: Sequence[FrameStats], view_class: str='broadside', *, subwindow: Optional[Dict[str, float]]=None, **kw) -> bool:
    """The pre-registered SUSTAINED rule: >= ``sustain_frac`` of blocks pass."""
    ...

def score_window(frames: Sequence[FrameStats], view_class: str='broadside', *, enable_subwindow: bool=False, enable_trunk_gate: bool=False, trunk_residual: bool=False, thresholds: Optional[Dict[str, float]]=None, trunk: Optional[Dict[str, float]]=None, subwindow: Optional[Dict[str, float]]=None) -> Dict[str, object]:
    """Trust decision for one window from its scored frames.

    ``enable_subwindow`` is DEFAULT OFF; off, this is the whole-window gate.
    On, the window is trusted if EITHER the whole window passes OR the
    pre-registered sustained sub-window rule passes -- a disjunction, so turning
    the flag on can only ADD trust.
    """
    ...

@dataclass
class WindowCandidate:
    """One candidate window, in the quantities A1/A2 read -- all pre-fit.

    ``detection_rate`` is the prep sidecar's (``prep/<clip>/quality.json``,
    ``detector.detection_rate``); ``sil_margin_clear_frac`` and
    ``centreline_ok_frac`` are fractions over the stride-2 centreline records
    (``quality.edge_near_end_px >= quality.edge_end_margin_px`` and
    ``quality.ok_quality``).  ``n_frames`` is the window length in frames.
    """
    window_id: str
    n_frames: int
    detection_rate: float
    sil_margin_clear_frac: Optional[float] = None
    centreline_ok_frac: Optional[float] = None
    L_px_native_median: Optional[float] = None

def admit_window(cand: WindowCandidate, *, params: Optional[Dict[str, float]]=None) -> Dict[str, object]:
    """A1.  Admissible, trunk-admissible, or rejected -- with the clause named.

    A window that fails ONLY ``sil_margin_clear_frac`` is *trunk-admissible*:
    it may be fitted and may report girths, cross-sections and ratios at the
    stations that are in frame, and may never report TL, volume or mass.  That
    is the existing TRUNK_ONLY contract, triggered by the mask rather than by a
    bbox flag.  A missing centreline is not a failure -- it is unknown, and the
    window falls back to trunk-admissible rather than being rejected on absence.
    """
    ...

def expected_measured_frames(cand: WindowCandidate, *, params: Optional[Dict[str, float]]=None) -> float:
    """A2's first factor: how many frames this window is expected to MEASURE.

    ``n_frames x detection_rate x sil_margin_clear_frac x residual_keep_rate``.
    An unknown margin is taken as 1.0 and the window is not admissible anyway,
    so the number is only ever used to order trunk-admissible candidates.
    """
    ...

def _resolution_factor(cand: WindowCandidate, p: Dict[str, float]) -> float:
    ...

def admit_then_rank(candidates: Sequence[WindowCandidate], *, enabled: bool=False, views_already_held: Sequence[str]=(), params: Optional[Dict[str, float]]=None) -> Dict[str, object]:
    """A1 + A2.  DEFAULT OFF; off, the input order is returned untouched.

    ``rank_score = expected_measured_frames x resolution x view_bonus``, and the
    view bonus is what turns ranking from "pick the two best windows" into "pick
    the two windows that together observe the most axes": an oblique window is
    worth full score while the individual holds none, and 0.6 once it does.  The
    bonus is therefore STATEFUL, so selection is greedy -- each pick updates the
    views held and the remaining candidates are re-scored.  It is a stated
    policy, not a measured coefficient; the three factors before it are each
    measured per window.
    """
    ...
CLIP_FRAMES_MIN_SPAN = 30
CLIP_FRAMES_MAX_FRAC = 0.9

def best_span(frames: Sequence[FrameStats], frame_numbers: Sequence[int], view_class: str='broadside', *, window_end_frame: Optional[int]=None, scored_stride: Optional[int]=None, **kw) -> Optional[Dict[str, object]]:
    """C1.  The best sub-window as SOURCE frame numbers, half-open.

    The rule: the longest run of consecutive scored frames whose block gate
    passes, expanded to the enclosing unscored frames (the scored frames are a
    stride-sampled subset, so the span between two scored frames belongs to the
    span).  With too few scored frames for a block -- which is every window in
    this corpus at ``run_len = 5`` -- the whole window is the span if the
    whole-window gate passes, and there is no span otherwise.

    ``scored_stride`` defaults to the median gap between scored frames, so the
    last scored frame's own interval is included rather than truncated.
    """
    ...

def window_relative_clip_frames(span: Optional[Dict[str, object]], frame_start: int, n_frames: int, *, clip_id: Optional[str]=None, min_span: int=CLIP_FRAMES_MIN_SPAN, max_frac: float=CLIP_FRAMES_MAX_FRAC) -> Dict[str, object]:
    """C2.  ``best_span`` (source frames) -> ``--clip-frames ID=start:end``.

    Window-relative and half-open, exactly as ``fit_individual.parse_clip_frames``
    reads it, clipped to ``[0, n_frames]``.  The flag is REFUSED, with the reason
    recorded, when the span is shorter than ``min_span`` frames or longer than
    ``max_frac`` of the window (nothing was gained by trimming).
    """
    ...
