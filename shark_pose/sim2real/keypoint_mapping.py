"""
YOLO 2D keypoint detector -> 3D SharkSMPL skeleton remap.

The 2D detector (YOLOv8-pose, 16 keypoints) and the 3D SharkSMPL skeleton
(16 joints) use DIFFERENT anatomical orderings. The detector annotates
front/back variants for three body-axis landmarks (gill slit, pectoral
base, first dorsal base), while the 3D skeleton carries a single joint on
the body axis for each of them. The 3D skeleton also includes interior
joints (body_midpoint_dorsal/ventral, the internal second_dorsal_base)
that have no 2D equivalent at all.

Feeding YOLO output directly to anything that expects 3D-skeleton order
(ProxyExtractor heatmap channels, SMPLify reprojection loss, the SPIN
joints_2d head) produces 12-15 px of avoidable error per real sample
because half the joints land at wrong anatomical positions.

Merge policy (2026-09-03 decision)
----------------------------------
This module used to take a SINGLE source per skeleton joint (YOLO 3, 4
and 8 for skeleton joints 2, 3 and 5), while the real-data importer
(scripts/import_real_data.py) took the MIDPOINT of the front/back pair.
The two conventions disagreed by 12-23 px on exactly those three joints,
so proxies built by the importer and keypoints remapped at inference time
lived in different anatomical frames.

The MIDPOINT is now the single policy for the whole repo: skeleton joints
2, 3 and 5 are the confidence-weighted average of their front/back pair
(:data:`SKELETON_PAIR_SOURCES`, :func:`merge_pair`). For the binary
visibility flags that come out of the human annotations this reduces
exactly to the importer's old rule -- both variants labelled gives the
midpoint, one variant labelled gives that one, neither gives zero -- and
for soft detector confidences it degrades gracefully towards whichever
variant the detector is more sure about.

:data:`SKELETON_TO_YOLO` is kept because it still documents the primary
(anatomically closest) YOLO source for every skeleton joint, and it drives
the 13 direct one-to-one joints.

YOLO order is documented in scripts/analyze_real_poses.py based on the
v3/v4/v5 trained detectors. SharkSMPL order is shark_pose.core.skeleton.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

def _is_torch(x) -> bool:
    """True if x is a torch.Tensor, without importing torch eagerly."""
    ...

def merge_pair(p_a, c_a, p_b, c_b):
    """Confidence-weighted merge of a front/back keypoint pair.

    Parameters
    ----------
    p_a, p_b : array-like, shape (..., 2)
        Positions of the two variants, in any (shared) pixel space.
    c_a, c_b : array-like, shape (...)
        Their confidences.  Negative values are clamped to zero.

    Returns
    -------
    position : same type/shape as ``p_a``
        ``(c_a * p_a + c_b * p_b) / (c_a + c_b)`` where at least one
        confidence is positive, and zero where both are non-positive.
    confidence : same type/shape as ``c_a``
        ``max(c_a, c_b)``, zero where both are non-positive.

    Notes
    -----
    For binary visibility flags this is exactly the historical
    ``import_real_data.midpoint`` rule: both variants visible gives the
    midpoint, one visible gives that one, neither gives zero.  numpy
    arrays and torch tensors are both accepted (dispatched on type), so
    the same policy serves the numpy and torch remap paths.
    """
    ...

def remap_yolo_to_skeleton(yolo_keypoints: np.ndarray, yolo_confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder YOLO 2D keypoints into the 3D SharkSMPL skeleton order.

    Skeleton joints 2 (gill_slit), 3 (pectoral_base) and 5
    (first_dorsal_base) are the confidence-weighted merge of their
    front/back YOLO pair (:func:`merge_pair`); the other ten mapped joints
    are copied straight from :data:`SKELETON_TO_YOLO`.

    Args:
        yolo_keypoints: (16, 2) keypoints in YOLO order, in any pixel space.
        yolo_confidences: (16,) per-keypoint confidence in [0, 1].

    Returns:
        skeleton_keypoints: (16, 2) keypoints reordered to 3D skeleton index.
            Joints with no 2D equivalent (7, 14, 15) are zero.
        skeleton_confidences: (16,) confidences reordered to 3D skeleton index.
            Joints with no 2D equivalent are zero (treated as occluded).
    """
    ...

def remap_yolo_to_skeleton_torch(yolo_keypoints, yolo_confidences):
    """Torch-tensor version. Same semantics as remap_yolo_to_skeleton.

    Accepts (B, 16, 2) and (B, 16) for batch inputs, or (16, 2) and (16,)
    for a single sample. Returns matching shapes.
    """
    ...
