"""
Optical flow extraction for the proxy representation.

Channels 16-17 of the 19-channel proxy tensor carry (u, v) optical flow,
providing temporal motion cues that help disambiguate depth from monocular
video.  Flow is domain-invariant: it depends on motion, not appearance, so
synthetic and real flows share the same distribution.

Primary backend: RAFT (Teed & Deng, 2020) for high-quality dense flow.
Fallback: OpenCV Farneback for environments without RAFT.
"""
from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class OpticalFlowExtractor:
    """Extract dense optical flow between consecutive frames.

    Parameters
    ----------
    method : str
        ``"raft"`` for RAFT neural flow, ``"farneback"`` for classical
        OpenCV Farneback.  If ``"raft"`` is requested but unavailable,
        falls back to ``"farneback"`` with a warning.
    raft_model_path : str or None
        Path to RAFT checkpoint (only used when method="raft").
    raft_iters : int
        Number of RAFT refinement iterations (higher = better but slower).
    device : str
        Torch device for RAFT inference.
    farneback_params : dict or None
        Override default Farneback parameters. Keys match
        ``cv2.calcOpticalFlowFarneback`` arguments.
    resolution : int or None
        If set, resize flow output to (resolution, resolution).
    normalize : bool
        If True, normalise flow to [-1, 1] range using the maximum
        displacement magnitude.  Recommended for network input.
    """

    def __init__(self, method: str='farneback', raft_model_path: Optional[str]=None, raft_iters: int=20, device: str='cpu', farneback_params: Optional[dict]=None, resolution: Optional[int]=None, normalize: bool=True) -> None:
        ...

    def extract(self, frame1: np.ndarray, frame2: np.ndarray) -> torch.Tensor:
        """Compute dense optical flow from frame1 to frame2.

        Parameters
        ----------
        frame1 : np.ndarray
            Previous frame, BGR uint8, shape (H, W, 3).
        frame2 : np.ndarray
            Current frame, BGR uint8, shape (H, W, 3).

        Returns
        -------
        flow : torch.Tensor, shape (2, H', W')
            (u, v) flow.  H', W' equals ``self.resolution`` if set,
            otherwise matches input spatial dimensions.
        """
        ...

    def extract_zero(self, height: int, width: int) -> torch.Tensor:
        """Return zero flow tensor (for first frames with no predecessor).

        Parameters
        ----------
        height : int
        width : int

        Returns
        -------
        flow : torch.Tensor, shape (2, H, W)
        """
        ...

    @staticmethod
    def compute_per_keypoint_flow(flow: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """Sample flow vectors at keypoint locations for depth disambiguation.

        The magnitude and direction of flow at each keypoint provides a
        strong cue about relative depth: closer objects move faster under
        the same camera motion (motion parallax).

        Parameters
        ----------
        flow : torch.Tensor, shape (2, H, W)
            Dense optical flow field.
        keypoints : torch.Tensor, shape (K, 2)
            (x, y) keypoint coordinates in pixel space.

        Returns
        -------
        kp_flow : torch.Tensor, shape (K, 2)
            (u, v) flow sampled at each keypoint location.
        """
        ...

    def _try_load_raft(self) -> None:
        """Attempt to load RAFT. Falls back to Farneback on failure."""
        ...

    @torch.no_grad()
    def _extract_raft(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Extract flow using RAFT.

        Returns
        -------
        flow : np.ndarray, shape (H, W, 2)
        """
        ...

    def _extract_farneback(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Extract flow using OpenCV Farneback.

        Returns
        -------
        flow : np.ndarray, shape (H, W, 2)
        """
        ...
