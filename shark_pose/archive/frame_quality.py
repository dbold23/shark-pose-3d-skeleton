"""Per-part frame quality for the S8 re-ID front end, and the diverse best-frame selection.

Measured on the part's canonical crop (``parts.crop_part``), so every number means the same thing whatever the
camera, range or zoom:

  kp_conf     lowest detector confidence of the part's keypoints
  inside      fraction of the crop that lies inside the source frame (a fin cut by the frame edge is useless)
  coverage    silhouette parts: fraction of each base -> tip segment the mask fills (an occluded, cut or
              half-segmented fin falls); surface parts: fraction of the crop's body band the body mask covers
              (glare, bubbles, a bait line across the flank fall)
  sharpness   var(Laplacian) / var(grey) inside the part: gain-free (a darker frame is not a blurrier one),
              falls monotonically with defocus and motion blur
  clipped     fraction of part pixels at 0-4 or 251-255 grey (blown highlights, crushed shadows)
  scale       source px per crop px; below 1 the crop magnifies and adds no detail
  view        the frame's depth-free view cue (``corpus.view_yaw`` fin_cue: body length over a fin span, which
              falls as yaw rises), used for DIVERSITY, not as a gate

Score = kp_conf x min(1, scale) x coverage x (1 - clipped) x rank(sharpness within this video's candidates for
the part), after the gates below. Selection (``select_diverse``) then spreads the picks over flank side and
view tertile, best first, at least ``min_gap`` frames apart, so an encounter is described from every angle it
offers rather than by twelve near-copies of its best second.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from shark_pose.archive.parts import PartSpec
SHORTLIST = 2

def sharpness(gray, mask=None) -> float:
    ...

def clipped(gray, mask=None) -> float:
    ...

def coverage(mask_crop, spec: PartSpec, kp_crop=None) -> float:
    """Silhouette: fraction of points along each base -> tip segment that the mask covers (an occluded, cut or
    half-segmented fin falls). Surface: fraction of the crop's central band that the body mask covers."""
    ...

@dataclass
class Candidate:
    frame: int
    part: str
    side: str
    kp_conf: float
    inside: float
    scale: float
    view: float | None = None
    silhouette: bool = True
    sharp: float = 0.0
    coverage: float = 1.0
    clipped: float = 0.0
    score: float = 0.0

    def passes(self, gates=GATES) -> bool:
        ...

def score_candidates(cands: list[Candidate], gates=GATES) -> list[Candidate]:
    """Gate, then score with the sharpness RANK among this list (one video, one part).

    A surface part's coverage is first taken relative to the best in the list: how much of the flank a crop
    holds depends on the body's depth, so only the drop against the same animal's clearest frame is occlusion."""
    ...

def view_bins(cands: list[Candidate], n_bins: int=3) -> dict[int, int]:
    """Frame -> view tertile (0 = most broadside) among these candidates; frames without a view cue -> -1."""
    ...

def select_diverse(cands: list[Candidate], k: int, min_gap: int) -> list[Candidate]:
    """Best frames spread over (side, view tertile): round-robin over the bins in order of their best score,
    each turn taking that bin's best remaining frame at least ``min_gap`` from every frame already taken."""
    ...
