"""
Edge and silhouette feature extraction for the proxy representation.

Channel 18 of the 19-channel proxy tensor carries an edge/silhouette map.
Edges are highly domain-invariant: a shark's silhouette against water
produces similar edge responses whether the image is synthetic or real.
This makes edge features an ideal bridge for sim2real transfer.

Supported methods:
- conv_filter (default): Sobel/Scharr convolutional filters (cheap, robust)
- canny: Canny edge detection with adaptive thresholds
- hed: Holistically-Nested Edge Detection (learned, highest quality)
"""
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class EdgeExtractor:
    """Extract edge maps from images for the proxy representation.

    Parameters
    ----------
    method : str
        One of ``"conv_filter"``, ``"canny"``, ``"hed"``.
    filter_type : str
        ``"sobel"`` or ``"scharr"`` (only used when method="conv_filter").
    kernel_size : int
        Kernel size for Sobel filter (only used for Sobel; Scharr is fixed
        at 3x3).
    canny_low : float
        Lower threshold ratio for Canny (relative to Otsu threshold).
    canny_high : float
        Upper threshold ratio for Canny (relative to Otsu threshold).
    hed_model_path : str or None
        Path to HED prototxt and caffemodel (only used when method="hed").
    resolution : int or None
        If set, resize output to (resolution, resolution).
    gaussian_blur_ksize : int
        Gaussian blur kernel size applied before edge detection to reduce
        noise (especially important for underwater imagery).
    device : str
        Torch device for HED inference.
    """

    def __init__(self, method: str='conv_filter', filter_type: str='sobel', kernel_size: int=3, canny_low: float=0.5, canny_high: float=1.0, hed_model_path: Optional[str]=None, resolution: Optional[int]=None, gaussian_blur_ksize: int=5, device: str='cpu') -> None:
        ...

    def extract(self, image: np.ndarray, method: Optional[str]=None) -> torch.Tensor:
        """Extract edge map from an image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape (H, W, 3).
        method : str or None
            Override the instance default method for this call.

        Returns
        -------
        edges : torch.Tensor, shape (1, H', W')
            Edge map normalised to [0, 1].  H', W' equals
            ``self.resolution`` if set, otherwise matches input.
        """
        ...

    def extract_silhouette(self, mask: np.ndarray) -> torch.Tensor:
        """Extract silhouette contour from a binary segmentation mask.

        Designed for use with SAM2 or similar segmentation outputs.  The
        contour of the mask provides a clean shape boundary that is
        perfectly domain-invariant (synthetic masks and real SAM2 masks
        both give crisp contours).

        Parameters
        ----------
        mask : np.ndarray
            Binary mask, shape (H, W), dtype uint8, values in {0, 255}
            or {0, 1}.

        Returns
        -------
        silhouette : torch.Tensor, shape (1, H', W')
            Silhouette edge map normalised to [0, 1].
        """
        ...

    def _extract_conv_filter(self, image: np.ndarray) -> np.ndarray:
        """Compute edge magnitude using Sobel or Scharr filters.

        Returns
        -------
        edge_map : np.ndarray, shape (H, W), float32 in [0, 1].
        """
        ...

    def _extract_canny(self, image: np.ndarray) -> np.ndarray:
        """Compute Canny edges with Otsu-adaptive thresholds.

        Returns
        -------
        edge_map : np.ndarray, shape (H, W), float32 in [0, 1].
        """
        ...

    def _try_load_hed(self) -> None:
        """Attempt to load HED network via OpenCV DNN."""
        ...

    def _extract_hed(self, image: np.ndarray) -> np.ndarray:
        """Compute edges using Holistically-Nested Edge Detection.

        Returns
        -------
        edge_map : np.ndarray, shape (H, W), float32 in [0, 1].
        """
        ...
