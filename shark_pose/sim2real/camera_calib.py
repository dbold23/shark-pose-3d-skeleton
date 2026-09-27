"""
Lens calibration for the real cameras the footage comes off.

Everything upstream of this module assumes a PINHOLE camera: the proxy
letterbox (``proxy_geometry``), the SMPLify reprojection term, the soft
silhouette rasteriser and the mask centreline all treat an image pixel as
``f * X / Z + c``.  Real dive footage is nothing of the sort.  The demo clip is
a GoPro HERO5 in Wide FOV, whose barrel distortion moves a corner pixel by 442
px (28% of the half-diagonal) and makes a perfectly straight shark at the demo's
positions read as bent by 1.6 deg median / 3.6 deg while it is near the frame
edge -- about a third of the "bend" the mask centreline measures.  The second
error is the focal length: the pipeline's fixed ``BASE_FOCAL = 1000`` at 256 px
is a 14.6 deg horizontal field of view, where this camera has ~80 deg, so the
within-body perspective that carries the tail-toward/away depth cue is an order
of magnitude weaker in the model than it is in reality.

The fix is to UNDISTORT THE OBSERVATIONS (keypoints, masks, centrelines) into a
pinhole frame and to fit with that pinhole's real focal length, rather than to
re-encode the video.  This module owns both halves.

Coordinate frames
-----------------
``K``/``dist``      the physical camera: raw video pixels.
``new_K``           the ideal pinhole every consumer works in, from
                    ``cv2.getOptimalNewCameraMatrix(..., alpha=1.0)``.  alpha=1
                    keeps EVERY source pixel inside the same ``width x height``
                    canvas (the undistorted image has curved black borders), so
                    no observation is ever cropped away and the letterbox
                    geometry is unchanged.  It also means ``new_K``'s focal is
                    SHORTER than ``K``'s -- the ideal camera has to be wider to
                    hold the barrel-corrected corners.
proxy frame         ``new_K`` pushed through
                    ``proxy_geometry.image_to_proxy_xy``: a uniform scale of
                    ``resolution / max(W, H)`` plus a centring offset, so
                    ``proxy_focal() = new_K.fx * scale`` and the principal
                    point moves by the same map.

Every consumer must use ``new_K`` CONSISTENTLY -- points, masks, centrelines,
the fitter's focal length AND its principal point.  Mixing an undistorted
keypoint with a distorted mask is worse than doing nothing at all.

Specs
-----
``load_camera(spec, width, height)`` accepts

* a path to a JSON written by ``marine-cv/video-distortion`` (`profile`,
  `calibrate` or `detect`): keys ``camera_matrix``, ``dist_coeffs`` and
  optionally ``width``/``height``;
* ``"preset:<name>"``, a hard-coded GoPro profile (see :data:`GOPRO_PRESETS`)
  evaluated at the given frame size;
* ``"none"`` (or ``None``), which returns ``None`` -- the caller then keeps its
  legacy no-camera behaviour bit for bit.

Presets are APPROXIMATIONS of a camera model, not measurements of the lens that
shot the clip.  A checkerboard calibration taken through the actual housing,
underwater, is the gold standard; ``load_camera`` takes such a calibration as a
JSON file, so dropping one in replaces the preset with no code change.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np
from .proxy_geometry import image_to_proxy_xy, letterbox_params
__all__ = ['CameraIntrinsics', 'GOPRO_PRESETS', 'load_camera', 'preset_intrinsics']

def _cv2():
    ...

class CameraIntrinsics:
    """One physical camera: ``K``, ``dist``, the frame size they refer to.

    Attributes
    ----------
    K : np.ndarray, (3, 3)
        Intrinsic matrix of the PHYSICAL camera, in raw video pixels.
    dist : np.ndarray, (N,)
        Brown-Conrady coefficients in OpenCV order (k1, k2, p1, p2, k3, ...).
    width, height : int
        Frame size ``K`` and ``dist`` were measured at.
    source : str
        Provenance string stamped into every record this camera touches.
    """

    def __init__(self, K: Sequence[Sequence[float]] | np.ndarray, dist: Sequence[float] | np.ndarray, width: int, height: int, source: str='unknown') -> None:
        ...

    @property
    def new_K(self) -> np.ndarray:
        """The ideal pinhole every consumer projects into, ``alpha = 1.0``.

        ``alpha = 1`` keeps every source pixel inside the SAME ``width x
        height`` canvas, so nothing observable is cropped and the letterbox
        that follows is unchanged.  Cached: OpenCV solves a small optimisation
        for it.
        """
        ...

    def undistort_points(self, xy) -> np.ndarray:
        """Raw video pixels -> ideal-pinhole (``new_K``) pixels.

        Accepts any array of shape ``(..., 2)`` and returns the same shape as
        float64.  Non-finite points are passed through untouched (a missing
        keypoint is stored as NaN in places, and OpenCV would turn it into
        garbage rather than propagate it).
        """
        ...

    def distort_points(self, xy) -> np.ndarray:
        """Ideal-pinhole (``new_K``) pixels -> raw video pixels.

        Exact inverse of :meth:`undistort_points` (round-trip error is well
        under 0.05 px over the whole frame; see ``tests/test_camera_calib.py``).
        Use it to draw a fit back onto the ORIGINAL, still-distorted video.
        """
        ...

    def undistort_image(self, image: np.ndarray, interpolation: Optional[int]=None) -> np.ndarray:
        """Remap a raw frame (or mask) into the ``new_K`` pinhole.

        The rectify maps are built once and cached, so a 300-frame clip pays
        for ``initUndistortRectifyMap`` exactly once.  ``interpolation``
        defaults to bilinear; pass ``cv2.INTER_NEAREST`` for a segmentation
        mask, which must stay binary.
        """
        ...

    def proxy_focal(self, resolution: int=256) -> float:
        """``new_K``'s focal length expressed in proxy pixels.

        The letterbox is a UNIFORM scale, so one focal length serves both axes;
        ``new_K``'s fx and fy differ by ~4e-3 px here and fx is the reference
        (the horizontal field of view is the one the tool's preset fixes).
        """
        ...

    def proxy_principal_point(self, resolution: int=256) -> Tuple[float, float]:
        """``new_K``'s principal point mapped through the proxy letterbox.

        Note this is NOT ``(res/2, res/2)`` in general: ``getOptimalNewCameraMatrix``
        is free to move the principal point, and the fitter has to be told,
        otherwise the whole fit absorbs the offset into the translation.
        """
        ...

    def rescaled(self, width: int, height: int) -> 'CameraIntrinsics':
        """The same lens at another frame size (identical aspect ratio).

        A resized encode of the same recording has the same distortion in
        normalised coordinates, so ``K`` scales and ``dist`` is untouched.  A
        DIFFERENT aspect ratio means a different sensor crop, not a resize, and
        is refused -- profile that video instead.
        """
        ...

    def stamp(self) -> dict:
        """The ``"camera"`` block every record/JSON this camera touches carries."""
        ...

    def __repr__(self) -> str:
        ...

def preset_intrinsics(name: str, width: int, height: int) -> CameraIntrinsics:
    """A :data:`GOPRO_PRESETS` entry evaluated at one frame size.

    Reproduces ``distortion.py cmd_profile`` exactly: ``f = frac * W``,
    principal point at the frame centre, ``dist = [k1, k2, 0, 0, 0]``.
    """
    ...

def load_camera(spec: Optional[str], width: Optional[int]=None, height: Optional[int]=None) -> Optional[CameraIntrinsics]:
    """Resolve a camera spec into a :class:`CameraIntrinsics`, or ``None``.

    Parameters
    ----------
    spec :
        ``None``, ``""`` or ``"none"`` -> ``None``: no camera model, and every
        caller keeps its pre-camera behaviour bit for bit.
        ``"preset:<name>"`` -> :func:`preset_intrinsics` (needs ``width`` and
        ``height``).
        anything else -> a path to a ``video-distortion`` JSON with keys
        ``camera_matrix`` and ``dist_coeffs`` (and usually ``width``/``height``).
    width, height :
        The frame size the camera will be USED at.  Required for a preset.  For
        a JSON, the stored size wins and the camera is
        :meth:`~CameraIntrinsics.rescaled` if the frames differ (same aspect
        ratio only).

    Returns
    -------
    CameraIntrinsics or None
    """
    ...
