"""
Keypoint detection, tracking, and heatmap generation for proxy representations.

The heatmap channels form the primary signal in the 19-channel proxy tensor.
Each heatmap is confidence-weighted so the network receives soft detection
signals rather than hard binary locations. This is critical for sim2real:
synthetic data uses confidence=1.0 (perfect GT), while real detections carry
detector uncertainty, and the network learns to be robust to both.

Heatmap formula:
    H_i(x, y) = w_i * exp(-((x - x_i)^2 + (y - y_i)^2) / (2 * sigma^2))
where w_i in [0, 1] is the detector confidence for keypoint i.
"""
from __future__ import annotations
from typing import Optional, Protocol, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shark_pose.core.constants import DEFAULT_HEATMAP_SIGMA, NUM_KEYPOINT_CHANNELS

class KeypointDetectorBackend(Protocol):
    """Any object exposing a predict(image) -> (keypoints, confidences) API."""

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

class KeypointDetector:
    """Wrapper around a trained 2D keypoint detector.

    Parameters
    ----------
    backend : KeypointDetectorBackend
        Any model that implements ``predict(image) -> (kps, confs)``.
        Examples: HRNet, ViTPose, SimpleBaseline, or a custom shark detector.
    num_keypoints : int
        Expected number of keypoints (default 16 for the shark skeleton).
    confidence_threshold : float
        Keypoints below this threshold are kept but marked low-confidence
        (the heatmap will be near-zero due to weighting).
    """

    def __init__(self, backend: KeypointDetectorBackend, num_keypoints: int=NUM_KEYPOINT_CHANNELS, confidence_threshold: float=0.1) -> None:
        ...

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run 2D keypoint detection on a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR or RGB image, shape (H, W, 3), uint8.

        Returns
        -------
        keypoints : np.ndarray, shape (num_keypoints, 2)
            (x, y) pixel coordinates for each keypoint.
        confidences : np.ndarray, shape (num_keypoints,)
            Detection confidence in [0, 1] per keypoint.
        """
        ...

class KeypointTracker:
    """Track keypoints across video frames using Hungarian matching.

    Maintains a running state of the previous frame's keypoints and matches
    new detections to minimize total displacement cost.  This stabilises
    keypoint identity across frames and smooths jittery detections.

    Parameters
    ----------
    num_keypoints : int
        Number of keypoints per instance.
    max_displacement : float
        Maximum pixel displacement allowed between matched keypoints across
        consecutive frames.  Unmatched detections beyond this threshold keep
        the previous frame's position (with decayed confidence).
    confidence_decay : float
        Multiplicative decay applied to confidence when a keypoint is
        carried over (not re-detected).
    """

    def __init__(self, num_keypoints: int=NUM_KEYPOINT_CHANNELS, max_displacement: float=50.0, confidence_decay: float=0.8) -> None:
        ...

    def reset(self) -> None:
        """Clear tracker state (call at sequence boundaries)."""
        ...

    def update(self, keypoints: np.ndarray, confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Match new detections against previous frame and return tracked result.

        Parameters
        ----------
        keypoints : np.ndarray, shape (num_keypoints, 2)
        confidences : np.ndarray, shape (num_keypoints,)

        Returns
        -------
        tracked_keypoints : np.ndarray, shape (num_keypoints, 2)
        tracked_confidences : np.ndarray, shape (num_keypoints,)
        """
        ...

def generate_heatmaps(keypoints: torch.Tensor, confidences: torch.Tensor, resolution: int, sigma: float=DEFAULT_HEATMAP_SIGMA) -> torch.Tensor:
    """Generate confidence-weighted Gaussian heatmaps for all keypoints.

    This is the core function that converts discrete keypoint detections into
    the dense spatial representation consumed by the proxy encoder.  The
    confidence weighting is the key sim2real mechanism: synthetic data always
    has w_i = 1.0, while real data carries detector uncertainty.

    Parameters
    ----------
    keypoints : torch.Tensor, shape (K, 2)
        (x, y) pixel coordinates, in the range [0, resolution).
    confidences : torch.Tensor, shape (K,)
        Per-keypoint confidence in [0, 1].
    resolution : int
        Spatial resolution (H = W) of the output heatmaps.
    sigma : float
        Gaussian standard deviation in pixels (default
        ``DEFAULT_HEATMAP_SIGMA`` = 3.0, the value all training/eval
        proxies were built with).

    Returns
    -------
    heatmaps : torch.Tensor, shape (K, H, W)
        Confidence-weighted Gaussian heatmaps, values in [0, 1].
    """
    ...

def generate_heatmaps_batch(keypoints: torch.Tensor, confidences: torch.Tensor, resolution: int, sigma: float=DEFAULT_HEATMAP_SIGMA) -> torch.Tensor:
    """Batched version of generate_heatmaps.

    Parameters
    ----------
    keypoints : torch.Tensor, shape (B, K, 2)
    confidences : torch.Tensor, shape (B, K)
    resolution : int
    sigma : float

    Returns
    -------
    heatmaps : torch.Tensor, shape (B, K, H, W)
    """
    ...

def soft_argmax_2d(heatmaps: torch.Tensor, temperature: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable soft-argmax extraction of keypoints from heatmaps.

    Uses spatial softmax to compute expected (x, y) coordinates.  This is
    fully differentiable, enabling end-to-end training through heatmap
    prediction layers.

    Parameters
    ----------
    heatmaps : torch.Tensor, shape (B, K, H, W) or (K, H, W)
        Input heatmaps (need not be normalised).
    temperature : float
        Softmax temperature. Lower = sharper (closer to hard argmax).

    Returns
    -------
    keypoints : torch.Tensor, shape (B, K, 2) or (K, 2)
        Extracted (x, y) coordinates in pixel space [0, H).
    confidences : torch.Tensor, shape (B, K) or (K,)
        Confidence scores derived from peak heatmap values.
    """
    ...
