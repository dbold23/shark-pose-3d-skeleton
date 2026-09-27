"""Dorsal-fin trailing-edge descriptor for white shark photo-ID (S8 R3).

The central California catalogue identifies white sharks by the notches of the
first dorsal fin's trailing edge (Anderson et al. 2011, Mar Biol 158:1233;
Chapple et al. 2011, Biol Lett 7:581). This module turns one frame's body mask
plus the detector's three fin keypoints (YOLO 7 ``first_dorsal_tip``, 8
``dorsal_base_front``, 9 ``dorsal_base_back``) into a view-normalised
trailing-edge profile, and compares two profiles with a shift-tolerant
correlation of their notch detail (after CurvRank's idea of comparing
curvature, not raw position; Weideman et al. 2017).

Pipeline per frame
  1. ``fin_mask_from_body``: the part of the body mask on the tip side of the
     base line (the base keypoints sit on the dorsal outline, so that side is the
     fin alone), largest component.
  2. ``fin_contour`` -> ``trailing_edge``: the contour path from the point nearest
     the tip to the point nearest the posterior base that does not pass the
     anterior base.
  3. ``edge_profile``: signed offset of that path from its own tip->posterior-base
     chord, as a function of arc-length fraction, in chord units. Positive points
     toward the anterior base, so a left-swimming and a right-swimming view of one
     fin give the same profile (the fin outline is a silhouette).
  4. ``detail``: profile minus a Gaussian-smoothed copy: the notches, with the
     fin's overall sweep (which obliquity and fin flex change) removed.

A profile carries ``distinctiveness`` (RMS of the detail) so smooth, unmarked
fins can be reported as a stratum rather than as matching failures.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
N_PROFILE = 128
MAX_SHIFT = 6
TRIM = 4

@dataclass
class EdgeProfile:
    profile: np.ndarray
    detail: np.ndarray
    distinctiveness: float
    chord_px: float
    n_edge_px: int

def _side(p, a, b):
    """Sign of point(s) p relative to the directed line a->b (z of the 2D cross product)."""
    ...

def fin_mask_from_body(mask: np.ndarray, tip, base_front, base_back, margin_frac: float=0.02) -> np.ndarray:
    """Pixels of ``mask`` on the tip side of the base line, largest connected component."""
    ...

def fin_contour(fin_mask: np.ndarray) -> np.ndarray:
    ...

def trailing_edge(contour: np.ndarray, tip, base_front, base_back) -> np.ndarray:
    """Contour path tip -> posterior base that does not pass the anterior base."""
    ...

def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    ...

def _gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    ...

def edge_profile(edge: np.ndarray, base_front, n: int=N_PROFILE) -> EdgeProfile:
    """Signed offset of the trailing edge from its chord, positive toward the anterior base."""
    ...

def refine_keypoints(contour: np.ndarray, tip, base_front, base_back, cut_band_frac: float=0.06, band_radius_frac: float | None=None):
    """Re-seat the three fin keypoints on the fin contour itself.

    The detector's keypoints only say roughly where the fin is; the edge endpoints they
    pick move the whole profile (5 % keypoint noise cut synthetic rank 1 from 50/60 to
    17/60). The tip becomes the contour point farthest from the base line; the two base
    points become the extreme contour points, along the base direction, of the band
    where the fin mask was cut from the body.
    """
    ...

def profile_from_mask(mask, tip, base_front, base_back, refine: bool=True, band_radius_frac: float | None=None) -> EdgeProfile:
    ...
STRETCHES = (1.0,)

def _stretch(x: np.ndarray, f: float) -> np.ndarray:
    """Re-sample x as if its arc were f times as long, anchored at the tip (index 0)."""
    ...

def match_score(a: np.ndarray, b: np.ndarray, max_shift: int=MAX_SHIFT, stretches: tuple[float, ...]=STRETCHES) -> float:
    """Best normalised correlation of two detail profiles over slide and stretch of ``b``."""
    ...

def encounter_similarity(details_a: list[np.ndarray], details_b: list[np.ndarray]) -> float:
    """Symmetrised median-of-best: each edge's best match in the other encounter, median, averaged both ways.

    The median (not the max) keeps an encounter with many edges from winning on one lucky
    random correlation.
    """
    ...

def _shift_segments(D: np.ndarray, s: int):
    """Centred, unit-norm overlapping segments of every row for slide s (a[s:] vs b[:-s] convention)."""
    ...

def profile_similarity_matrix(details: np.ndarray, max_shift: int=MAX_SHIFT, block: int=2000) -> np.ndarray:
    """All-pairs ``match_score`` (stretch 1.0) for a stack of detail profiles, vectorised and blocked.

    Row i vs column j equals ``match_score(details[i], details[j])``.
    """
    ...

def encounter_similarity_matrix(details: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encounter x encounter similarity by the symmetrised median-of-best rule of ``encounter_similarity``.

    ``groups[k]`` is the encounter index of profile k. Returns (unique group ids, matrix).
    """
    ...
