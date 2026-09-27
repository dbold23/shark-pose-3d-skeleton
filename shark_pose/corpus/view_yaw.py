"""The corpus scan's view (yaw) cue, and a depth-free replacement, DEFAULT OFF.

The defect
----------
The shipped cue is ``dvL`` -- the perpendicular dorso-ventral extent of the
visible fin/base landmarks divided by a least-squares projected snout->caudal-
notch length -- mapped to degrees by a TWO-ANCHOR linear map

    obl_deg_est = 7.545 + 52.78756 * (dvL_median - 0.4253)

(``outputs/demo_2026-09-03/work/individuals/scan_complete/src/verify_complete.py``
reading ``corpus_mine/verify/candidates_verified.json``).  Because the map is
linear and strictly increasing, ``obl_deg_est`` is a DETERMINISTIC function of
``dvL_median``: over the 2098 rows of ``individuals/ledger/windows.csv`` it
reproduces the shipped column for 2098 of 2098 windows and Spearman rho is
0.99997 (1.0 up to the column's rounding to 0.1 deg: 1566 distinct dvL values
collapse onto 406 distinct degrees).

The physics the cue leans on is real -- a yaw foreshortens the LONGITUDINAL axis
by cos(theta) and leaves transverse extents alone -- but the transverse
reference it picks is the TRUNK DEPTH, which is body condition.  Two animals at
the same yaw whose depth/length ratios differ by 0.089 (the p10-p90 spread of
``dvL`` over the 14 detection-backed windows the fit puts at <= 10 deg of yaw)
are read 4.7 deg apart in "obliqueness" for no optical reason.

The replacement
---------------
Same physics, a transverse reference that is a FIN span rather than the trunk:

    cue = L_proj / T,  T = caudal span, or KAPPA_CS_OVER_DH * dorsal-fin height
                           when the caudal tips are not detected

``L_proj`` is the confidence-weighted least-squares projected snout->caudal-notch
length over every visible landmark (the shipped ``dvL`` denominator, unchanged).
Neither the caudal span nor the dorsal-fin height is trunk girth, so the cue
cannot read a fat shark as an oblique one.  The cue FALLS as yaw rises.

Evidence (lane ``s1/yaw_defect``, report.md)
--------------------------------------------
* 9991 synthetic renders with EXACTLY known yaw and an EXACTLY constant
  depth/TL of 0.17699 (``data/synthetic/annotations``): 5-fold CV MAE 4.46 deg
  (median 3.70, p90 9.13) and Spearman 0.967 against truth, against the legacy
  cue's 10.91 deg / 7.96 / 25.13 and 0.735 -- i.e. the legacy cue is beaten in
  the regime that is most favourable to it, where nothing varies but the yaw.
* 110 prepared windows that have both a ``detections.npz`` and a fitted
  obliqueness (the only yaw truth on real frames, and itself noisy): every
  candidate lands between 11.35 and 14.70 deg of leave-one-out MAE, legacy
  12.41 and this cue 14.43, a spread the real truth cannot resolve.  The flag
  is therefore DEFAULT OFF and this module changes no shipped number until a
  lane turns it on deliberately.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np

@dataclass(frozen=True)
class ViewYawConfig:
    """Which view cue the corpus layer uses.

    ``estimator="legacy_dvL"`` is the DEFAULT and reproduces the shipped column
    bit-for-bit.  ``estimator="fin_referenced"`` is the depth-free replacement.
    """

    def __post_init__(self) -> None:
        ...

def legacy_obl_deg(dvL_median: float) -> float:
    """The shipped ``obl_deg_est``, to the shipped rounding.

    Reproduces ``individuals/ledger/windows.csv`` for 2098 of 2098 rows.
    """
    ...

def classify_dvL(dvL_median: float, thresholds: Optional[Dict[str, float]]=None) -> str:
    """The shipped view class, on the legacy cue."""
    ...

def _axis(kp: np.ndarray, vis: np.ndarray) -> Optional[np.ndarray]:
    """Anterior->posterior unit direction from the midline landmarks."""
    ...

def frame_cue(kp: np.ndarray, conf: np.ndarray, config: ViewYawConfig=DEFAULT_CONFIG) -> Optional[dict]:
    """Both cues for ONE frame of 16 skeleton-order keypoints, or ``None``.

    Returns ``L`` (projected snout->caudal-notch length), ``rms_over_L``,
    ``depth`` (the legacy dorso-ventral numerator), ``T`` (the fin-span
    transverse reference), ``dvL`` and ``fin_cue``.
    """
    ...

def window_cue(kp: np.ndarray, conf: np.ndarray, config: ViewYawConfig=DEFAULT_CONFIG) -> dict:
    """Per-window medians of both cues over a ``(F, 16, 2)`` / ``(F, 16)`` clip."""
    ...

def classify(cue_median: Optional[float], config: ViewYawConfig=DEFAULT_CONFIG, thresholds: Optional[Dict[str, float]]=None) -> Optional[str]:
    """View class from whichever cue ``config`` selects.

    With the default config this is ``classify_dvL`` exactly.  With
    ``fin_referenced`` the cue FALLS as yaw rises, so the comparisons invert and
    ``thresholds`` must be given on the fin cue's own scale
    (``{"broadside_ge": ..., "oblique_le": ...}``).
    """
    ...

def rank_preserving_thresholds(cue_medians: Sequence[Optional[float]], legacy_classes: Sequence[str]) -> Dict[str, float]:
    """Fin-cue thresholds that reproduce the legacy class PROPORTIONS.

    The legacy degree scale and the fitted-yaw scale disagree in level (legacy
    corpus median about 8 deg against a fitted-yaw median of 22.3 deg over the
    110 windows that have both), so cutting the new cue at the legacy degree
    thresholds would relabel almost everything for a reason that has nothing to
    do with the re-ordering.  Cutting at matched population fractions asks the
    honest question instead: does the new cue put the SAME windows in the same
    band?
    """
    ...
