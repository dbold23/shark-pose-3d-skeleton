"""Body parts of a white shark as canonical, side-labelled crops (S8 re-ID front end).

Every part is cut in ONE body-aligned frame: anterior to the LEFT, dorsal UP, scaled by a span of the part's
own keypoints. Silhouette features (a fin's edge) are then mirror-free; surface features (pigment, marks) are
compared only with the same flank, because the two flanks of one shark carry different marks.

Body frame from the 16 YOLO keypoints (``sim2real.keypoint_mapping.YOLO_KEYPOINT_NAMES``):
  heading h   least-squares direction of the visible midline keypoints ordered by their along-body station
              (``corpus.view_yaw.PROPORTIONS``), pointing to the snout
  dorsal n    the normal to h on the side of the dorsal keypoints (7-10) and away from the ventral ones (4-6, 11, 12)
  side        c = h_x n_y - h_y n_x in y-down pixels; c > 0 = LEFT flank (swimming to the image left, dorsal up)

Parts (YOLO indices): dorsal_fin (8, 9, 7), caudal_fin (13, 14, 15), pectoral_fin (4, 6, 5), head_gill (0, 2, 3),
flank (2, 12, and 8 for depth), pelvic (11, 12), second_dorsal (10, scaled by 12 -> 13).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
MIDLINE = (0, 1, 2, 3, 13)
DORSAL_KP, VENTRAL_KP = ((7, 8, 9, 10), (4, 5, 6, 11, 12))
KP_CONF = 0.5

@dataclass
class BodyFrame:
    h: np.ndarray
    n: np.ndarray
    side: str
    length_px: float

def body_frame(kxy, kconf, conf_min: float=KP_CONF) -> BodyFrame | None:
    ...

@dataclass(frozen=True)
class PartSpec:
    name: str
    need: tuple[int, ...]
    anchor: int
    scale: tuple[int, int]
    ax: float
    ay: float
    kx: float
    ky: float
    size: tuple[int, int]
    silhouette: bool
    min_scale_px: float

def part_affine(spec: PartSpec, kxy, kconf, frame: BodyFrame, conf_min: float=KP_CONF):
    """2x3 affine source px -> canonical crop, and the span in source px; None if the part is not usable."""
    ...

def crop_part(img, mask, M, spec: PartSpec, water=(127, 127, 127)):
    """(rgb crop, mask crop 0/1, fraction of the crop that lies inside the source frame)."""
    ...

def to_crop(M, pts):
    ...

def source_scale(M) -> float:
    """Source px per crop px (> 1: the crop downsamples; < 1: it magnifies and adds no detail)."""
    ...

def view_cue(kxy, kconf, frame: BodyFrame, conf_min: float=KP_CONF) -> float | None:
    """Depth-free view cue of ``corpus.view_yaw`` (fin_referenced) in YOLO order: body length along the heading
    over a transverse fin span (caudal tip-to-tip, else KAPPA x dorsal-fin height). Falls as yaw rises."""
    ...
