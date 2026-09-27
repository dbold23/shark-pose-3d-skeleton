"""
Shared image <-> proxy coordinate geometry (letterbox convention).

ALL real-image -> proxy coordinate transforms in this repo MUST go through
the three public functions defined here:

    letterbox_params(width, height, resolution)   -> (scale, off_x, off_y)
    image_to_proxy_xy(xy, width, height, resolution)
    proxy_to_image_xy(xy, width, height, resolution)
    letterbox_image(image_hwc, resolution)        -> (res, res, C) uint8

Rationale (2026-09-03 audit).  The pipeline previously squashed frames
anisotropically (scale_x = res / W, scale_y = res / H), which distorts a
16:9 frame by 1.78x in y relative to x.  The proxy heatmaps therefore lived
in a different geometry from the pinhole camera model the 3D fit assumes,
and every reprojection error inherited that distortion.  Switching to an
aspect-preserving letterbox (uniform scale + centred padding) measurably
improved test-time SMPLify fits: 18.3 -> 17.5 px mean and 38.8 -> 34.6 px
on unobserved joints.

Convention
----------
    scale = resolution / max(width, height)
    off_x = (resolution - width  * scale) / 2
    off_y = (resolution - height * scale) / 2
    x_proxy = x_image * scale + off_x
    y_proxy = y_image * scale + off_y

A square input (the 256x256 synthetic renders) gives scale = res / W and
zero offsets, i.e. a pure resize -- so synthetic behaviour is unchanged.

Both coordinate functions accept numpy arrays and torch tensors of shape
(..., 2) and dispatch on type; they are exact inverses of each other.
"""
from __future__ import annotations
from typing import Tuple, TypeVar
import numpy as np
import torch

def letterbox_params(width: int, height: int, resolution: int) -> Tuple[float, float, float]:
    """Compute the aspect-preserving letterbox mapping for one image size.

    Parameters
    ----------
    width, height : int
        Original image size in pixels.
    resolution : int
        Square proxy resolution (H = W = resolution).

    Returns
    -------
    scale : float
        Uniform scale factor applied to both axes.
    off_x, off_y : float
        Padding offsets (pixels, in proxy space) of the top-left corner of
        the scaled image.  Exactly one of them is zero unless the input is
        square, in which case both are.
    """
    ...

def image_to_proxy_xy(xy: ArrayLike, width: int, height: int, resolution: int) -> ArrayLike:
    """Map (x, y) from original image pixels into proxy pixels.

    Parameters
    ----------
    xy : np.ndarray or torch.Tensor, shape (..., 2)
        Coordinates in the original image frame.
    width, height : int
        Original image size.
    resolution : int
        Square proxy resolution.

    Returns
    -------
    Same type and shape as ``xy``, in proxy pixel coordinates.
    """
    ...

def proxy_to_image_xy(xy: ArrayLike, width: int, height: int, resolution: int) -> ArrayLike:
    """Map (x, y) from proxy pixels back into original image pixels.

    Exact inverse of :func:`image_to_proxy_xy` for the same size arguments.

    Parameters
    ----------
    xy : np.ndarray or torch.Tensor, shape (..., 2)
        Coordinates in proxy space.
    width, height : int
        Original image size.
    resolution : int
        Square proxy resolution.

    Returns
    -------
    Same type and shape as ``xy``, in original image pixel coordinates.
    """
    ...

def letterbox_image(image_hwc: np.ndarray, resolution: int, pad_value: int=0) -> np.ndarray:
    """Aspect-preserving resize of an image into a square proxy canvas.

    The image is scaled by ``letterbox_params`` and centred on a
    ``pad_value`` background, so pixel (x, y) of the original lands at
    ``image_to_proxy_xy((x, y), W, H, resolution)`` in the output.  Use this
    for every image handed to a proxy sub-extractor so that the flow and
    edge channels stay pixel-aligned with the keypoint heatmaps.

    Parameters
    ----------
    image_hwc : np.ndarray, shape (H, W) or (H, W, C)
        Input image, uint8.
    resolution : int
        Output square size.
    pad_value : int
        Fill value for the padded border (default 0).

    Returns
    -------
    np.ndarray, shape (resolution, resolution, C) -- or (resolution,
    resolution) for a 2D input -- same dtype as the input.
    """
    ...
