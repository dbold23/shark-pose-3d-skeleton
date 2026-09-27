"""
Underwater image correction for pre-processing real frames.

Underwater imagery suffers from wavelength-dependent attenuation (red
channel degrades fastest), backscatter (light scattered by suspended
particles), and low contrast.  These corrections are applied to real
images *before* proxy extraction so that edge features and keypoint
detections are more reliable.

Note: corrections are applied to the raw image for feature extraction
quality, but the proxy representation itself (heatmaps + flow + edges)
is what the network sees.  The network never sees raw RGB.
"""
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np

class UnderwaterCorrector:
    """Pre-process underwater images to improve proxy feature extraction.

    Parameters
    ----------
    clip_limit : float
        CLAHE clip limit for contrast enhancement.
    tile_grid_size : tuple of int
        CLAHE tile grid size.
    white_balance_method : str
        ``"gray_world"`` or ``"max_rgb"`` for white balance algorithm.
    backscatter_percentile : float
        Percentile of darkest pixels used to estimate backscatter level.
    backscatter_strength : float
        How aggressively to subtract estimated backscatter (0-1).
        Higher values remove more haze but risk colour shifts.
    enabled : bool
        Master toggle. When False, ``correct()`` returns input unchanged.
    """

    def __init__(self, clip_limit: float=3.0, tile_grid_size: tuple=(8, 8), white_balance_method: str='gray_world', backscatter_percentile: float=5.0, backscatter_strength: float=0.8, enabled: bool=True) -> None:
        ...

    def correct(self, image: np.ndarray) -> np.ndarray:
        """Apply the full underwater correction pipeline.

        Pipeline order:
        1. White balance (compensate colour cast)
        2. Backscatter removal (reduce haze / veiling light)
        3. Contrast enhancement (CLAHE on luminance)

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape (H, W, 3).

        Returns
        -------
        corrected : np.ndarray
            BGR uint8 image, shape (H, W, 3).
        """
        ...

    def white_balance(self, image: np.ndarray) -> np.ndarray:
        """Correct underwater colour cast via white balance.

        Underwater, red wavelengths attenuate first, producing a strong
        blue-green cast.  White balance restores more natural colour
        distribution, improving edge detection and keypoint localisation.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape (H, W, 3).

        Returns
        -------
        balanced : np.ndarray
            BGR uint8 image, shape (H, W, 3).
        """
        ...

    def _gray_world_wb(self, image: np.ndarray) -> np.ndarray:
        """Gray-world assumption: average colour should be gray.

        Scale each channel so that its mean equals the global mean.
        """
        ...

    def _max_rgb_wb(self, image: np.ndarray) -> np.ndarray:
        """Max-RGB white balance: scale channels by their max values.

        Assumes the brightest pixels in each channel represent white.
        """
        ...

    def remove_backscatter(self, image: np.ndarray) -> np.ndarray:
        """Remove backscatter (veiling light) from underwater images.

        Estimates the backscatter component as the low-percentile pixel
        values per channel (the background haze), then subtracts it
        weighted by ``backscatter_strength``.  This is a simplified
        version of the dark channel prior adapted for underwater scenes.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape (H, W, 3).

        Returns
        -------
        dehazed : np.ndarray
            BGR uint8 image, shape (H, W, 3).
        """
        ...

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE on the luminance channel.

        CLAHE (Contrast Limited Adaptive Histogram Equalisation) is applied
        in LAB colour space to the L channel only, preserving colour
        information while boosting local contrast.  This significantly
        improves edge detection in murky underwater conditions.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape (H, W, 3).

        Returns
        -------
        enhanced : np.ndarray
            BGR uint8 image, shape (H, W, 3).
        """
        ...
