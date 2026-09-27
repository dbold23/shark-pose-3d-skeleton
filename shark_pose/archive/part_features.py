"""Identity features from canonical part crops (S8 re-ID front end). Each works on a part crop made by
``parts.crop_part`` (anterior left, dorsal up) with its mask and the keypoints mapped into the crop.

  fin_edge        trailing-edge detail profile of the first dorsal (7 tip, 8 base front, 9 base back) or
                  pectoral fin (5 tip, 4 base front, 6 base back): ``fin_edge.profile_from_mask``
  notches         the discrete notches on that edge: positive peaks of the detail (indentations into the fin)
                  deeper than max(NOTCH_MIN, NOTCH_K x MAD); position 0 at the tip, 1 at the base
  fin_shape       dorsal fin geometry: height / base, apex position along the base, trailing-edge sweep,
                  leading-edge bulge, area / base^2
  caudal_edges    the upper-lobe and lower-lobe trailing margins (tip -> fork) as detail profiles; the fork
                  and the tips are re-seated on the contour first
  pigment_line    the countershading boundary along the flank: per column, the height (0 ventral outline,
                  1 dorsal outline) where grey rises fastest from dark back to white belly, resampled to
                  N_PROFILE, detail = minus its smoothed trend. Surface feature: one per SIDE.

Notch sets of one encounter are pooled (``pool_notches``): a notch must recur in several frames to count,
which removes contour noise that one frame alone would call a notch.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from shark_pose.archive.fin_edge import N_PROFILE, TRIM, EdgeProfile, _gauss_smooth, edge_profile, fin_contour, profile_from_mask
NOTCH_MIN = 0.008
NOTCH_K = 2.5
NOTCH_SEP = 4
NOTCH_TOL = 0.035
PIGMENT_MIN_VALID = 0.6
BASE_RADIUS = 0.35

def fin_edge(mask_crop, kp_crop, part: str='dorsal_fin') -> EdgeProfile:
    ...

def notches(detail: np.ndarray) -> list[tuple[float, float]]:
    ...

def pool_notches(sets: list[list[tuple[float, float]]], tol: float=NOTCH_TOL) -> list[float]:
    """Notch positions that recur: clustered within tol; a cluster must appear in >= max(2, 40 %) of the
    frames (all notches kept when there is one frame)."""
    ...

def notch_score(a: list[float], b: list[float], tol: float=NOTCH_TOL) -> float:
    """Kernel-weighted one-to-one matches over sqrt(n_a n_b); NaN when neither edge has a notch (two smooth
    fins say nothing about identity), 0 when only one does."""
    ...

def fin_shape(mask_crop, kp_crop, edge: EdgeProfile | None=None) -> np.ndarray:
    """[height/base, apex position along the base, trailing sweep, leading bulge, area/base^2] of the dorsal fin."""
    ...

def _arc(contour, i0, i1, avoid):
    ...

def caudal_edges(mask_crop, kp_crop) -> dict[str, EdgeProfile]:
    """Upper- and lower-lobe trailing margins, each tip -> fork, as ``EdgeProfile`` (positive toward the lobe)."""
    ...

@dataclass
class LineProfile:
    profile: np.ndarray
    detail: np.ndarray
    distinctiveness: float
    valid_frac: float

def _robust_outline(x, y, upward: bool, deg: int=4, iters: int=4):
    """Smooth outline through per-column extremes, rejecting the fins that stick out of it (up for the dorsal
    outline, down for the ventral one)."""
    ...

def pigment_line(rgb_crop, mask_crop, x_range=(0.25, 0.95)) -> LineProfile:
    """x_range in crop width: from behind the gill slits (which cross the boundary in every animal alike) to
    the end of the crop."""
    ...
