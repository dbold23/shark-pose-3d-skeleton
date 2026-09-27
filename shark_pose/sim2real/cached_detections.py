"""Cached 2D detections: a clip's ``detections.npz`` instead of a live detector.

The fitter normally runs the v5 YOLO detector on every decoded frame
(``scripts/fit_video.py``'s ``detect_frame``).  Under the flag
``detector.cached_detections: true`` it reads the clip's ``detections.npz``
instead, one row per frame, and never loads detector weights at all.  The flag
is OFF by default and OFF in production: the corpus ``detections.npz`` files
predate the letterbox fix and are a cache, not a contract.  It exists so that
synthetic bundles (whose keypoints are projected, not detected) go through the
shipped fitter unchanged, and so that any window's fit is reproducible on a box
with no GPU detector.

File format
-----------
Written by ``outputs/demo_2026-09-03/work/corpus_fits/make_detections.py``:

``kp``    (F, 16, 2) float32 -- PROXY pixels (the 256 px letterbox canvas), in
          the REMAPPED skeleton order of
          :data:`shark_pose.core.skeleton.SHARK_KEYPOINT_SEQUENCE`, i.e. the
          rows are already past ``remap_yolo_to_skeleton``.
``conf``  (F, 16) float32 -- per-keypoint confidence, same order.
``fps``   scalar -- the source video's frame rate (informational here; the fit
          reads its own fps off the video, as it does today).

Index base
----------
``F`` is indexed by the WINDOW frame number: row 0 is the window video's first
frame.  ``fit_video`` decodes from ``start_frame``, so the row for the frame it
is holding is ``start_frame + n_processed`` -- exactly the number it already
stores in ``buf_idx`` / ``all_idx`` for that frame.  A window frame past the end
of the cache (``i >= F``) is handed back as an all-zero row, which is the
fitter's own definition of "undetected" (``detected = conf.sum() > 0``).

Confidence semantics
--------------------
``YOLOKeypointBackend.predict`` applies its ``conf_threshold`` at BOX level --
ultralytics drops whole detections below it, and the per-keypoint ``conf``
values that survive are NOT thresholded again.  Cached rows were written by that
same backend (or by a generator imitating it), so they are already post-gate:
this module must NOT re-gate them per keypoint, or it would silently zero
keypoints the live path keeps.  ``detector.yolo_conf`` therefore has no effect
when the cached path is on, which is the same semantics, not a different one.

Coordinate spaces
-----------------
The cache stores proxy pixels; the live path produces native image pixels and
maps them to proxy at the very end (after the optional lens undistortion).  So
:func:`cached_detection_row` hands back BOTH: the cached proxy row verbatim, and
its exact inverse through :func:`~shark_pose.sim2real.proxy_geometry.proxy_to_image_xy`
in native pixels, so the SPIN proxy extractor and the undistortion downstream
receive what they receive today.  The two are consistent by construction:
``kp_2d == proxy_to_image_xy(kp_proxy, W, H, res)`` for every row, the zero row
included.

The cache's proxy resolution is assumed to be the fit's own ``res``
(``camera.image_size[0]``, 256 for every shipped config and for
``make_detections.py``'s default ``--proxy-res``).
"""
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Tuple
import numpy as np
from shark_pose.core.skeleton import NUM_JOINTS
from shark_pose.sim2real.proxy_geometry import proxy_to_image_xy

def load_cached_detections(path: str | Path) -> dict:
    """Read one ``detections.npz`` into plain numpy arrays.

    Returns ``{"kp": (F,16,2) float32, "conf": (F,16) float32, "fps": float or
    None, "path": str, "n_frames": F}``.  Raises on a missing file or a shape
    that is not the documented one -- a silently mis-shaped cache would be a
    wrong fit, not a slower one.
    """
    ...

def cached_detection_row(npz_arrays: Mapping[str, np.ndarray], i: int, H: int, W: int, res: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cached detection of ONE window frame, in both coordinate spaces.

    Parameters
    ----------
    npz_arrays : mapping with ``kp`` (F,16,2) and ``conf`` (F,16)
        A :func:`load_cached_detections` result, or the ``np.load`` object
        itself.
    i : int
        WINDOW frame index (row 0 = the window's first frame).  ``i >= F``
        yields the all-zero row: zero confidence is "undetected" downstream.
    H, W : int
        Native frame size, for the letterbox inverse.
    res : int
        Proxy resolution the cache is written in, and the fit's own.

    Returns
    -------
    kp_2d : (16, 2) float32 -- native image pixels.
    kp_proxy : (16, 2) float32 -- the cached row, verbatim (proxy pixels).
    conf : (16,) float32 -- the cached confidences, verbatim, NOT re-gated
        (see the module docstring: the live gate is at box level).
    """
    ...
