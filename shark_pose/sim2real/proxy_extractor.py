"""
Proxy representation extractor — the central sim2real bridge.

Combines all proxy channels into a single 19-channel tensor that serves
as the ONLY input to the pose regression network.  The network NEVER
sees raw RGB.  This is the critical design principle that enables
sim2real transfer: because both synthetic and real inputs are converted
to the same proxy representation space (heatmaps + flow + edges), the
domain gap is minimised at the input level.

Channel layout:
    Ch  0-15: Confidence-weighted keypoint heatmaps (one per joint)
              H_i(x,y) = w_i * exp(-((x-x_i)^2 + (y-y_i)^2) / (2*sigma^2))
    Ch 16-17: Optical flow (u, v) as depth/motion cue
    Ch    18: Edge / silhouette map

For synthetic data:
    - Keypoints come from projected GT 3D joints (confidence = 1.0)
    - Flow comes from rendered frame pairs
    - Edges come from rendered silhouettes or Sobel on renders

For real data:
    - Keypoints come from a trained 2D detector (confidence = detector output)
    - Flow comes from RAFT/Farneback on video frames
    - Edges come from Sobel/Canny on (optionally corrected) frames
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from shark_pose.core.constants import DEFAULT_HEATMAP_SIGMA, DEFAULT_RESOLUTION, NUM_EDGE_CHANNELS, NUM_FLOW_CHANNELS, NUM_KEYPOINT_CHANNELS, TOTAL_PROXY_CHANNELS
from shark_pose.sim2real.edge_features import EdgeExtractor
from shark_pose.sim2real.keypoint_tracks import generate_heatmaps
from shark_pose.sim2real.optical_flow import OpticalFlowExtractor
from shark_pose.sim2real.proxy_geometry import image_to_proxy_xy, letterbox_image
from shark_pose.sim2real.underwater_correction import UnderwaterCorrector

class ProxyExtractor:
    """Extract the 19-channel proxy representation from images.

    This class orchestrates all sub-extractors (heatmaps, flow, edges)
    and produces the unified tensor that the pose network consumes.

    Parameters
    ----------
    resolution : int
        Output spatial resolution (H = W).
    sigma : float
        Gaussian sigma for keypoint heatmaps.
    flow_method : str
        Optical flow method: ``"raft"`` or ``"farneback"``.
    edge_method : str
        Edge extraction method: ``"conv_filter"``, ``"canny"``, ``"hed"``.
    edge_filter_type : str
        Filter type for conv_filter edges: ``"sobel"`` or ``"scharr"``.
    underwater_correction : bool
        Whether to apply underwater image corrections before extracting
        proxy features from real images.
    keypoints_only : bool
        If True, skip underwater correction, optical flow and edge
        extraction entirely and emit zeros in channels 16-18.  2026-09-03
        audit: those three channels are exactly zero in 100% of the
        training data, and the corresponding ``conv1`` weights of the
        trained encoder are exactly 0 -- so computing them changes nothing
        downstream while accounting for most of the per-frame cost.
    raft_model_path : str or None
        Path to RAFT checkpoint.
    hed_model_path : str or None
        Path to HED model directory.
    device : str
        Torch device.
    """

    def __init__(self, resolution: int=DEFAULT_RESOLUTION, sigma: float=DEFAULT_HEATMAP_SIGMA, flow_method: str='farneback', edge_method: str='conv_filter', edge_filter_type: str='sobel', underwater_correction: bool=True, keypoints_only: bool=False, raft_model_path: Optional[str]=None, hed_model_path: Optional[str]=None, device: str='cpu') -> None:
        ...

    def extract(self, image: np.ndarray, keypoints: torch.Tensor, confidences: torch.Tensor, prev_image: Optional[np.ndarray]=None, mask: Optional[np.ndarray]=None) -> torch.Tensor:
        """Extract the full 19-channel proxy representation from a real image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape (H, W, 3).
        keypoints : torch.Tensor, shape (16, 2)
            Detected 2D keypoint coordinates (x, y) in pixel space.
            Should be in the coordinate system of the *original* image;
            they are mapped into proxy space internally with the shared
            aspect-preserving letterbox transform
            (:func:`shark_pose.sim2real.proxy_geometry.image_to_proxy_xy`).
        confidences : torch.Tensor, shape (16,)
            Per-keypoint detection confidence in [0, 1].
        prev_image : np.ndarray or None
            Previous frame for optical flow.  If None, flow channels
            are zero (suitable for first frame or single images).
        mask : np.ndarray or None
            Optional binary segmentation mask for silhouette extraction.
            If provided, the edge channel uses the silhouette contour
            instead of image-based edges.

        Returns
        -------
        proxy : torch.Tensor, shape (19, resolution, resolution)
            The complete proxy representation.
        """
        ...

    def extract_from_synthetic(self, render_data: Dict) -> torch.Tensor:
        """Extract proxy representation from synthetic render data.

        For synthetic data, keypoints are projected from GT 3D joints
        with perfect confidence (w_i = 1.0).  This produces the same
        19-channel tensor format as real extraction, ensuring the
        network sees a consistent input space across domains.

        Parameters
        ----------
        render_data : dict
            Expected keys:
            - ``"image"``: np.ndarray, BGR uint8, shape (H, W, 3)
            - ``"keypoints_2d"``: np.ndarray, shape (16, 2), projected GT
            - ``"prev_image"``: np.ndarray or None, previous render
            - ``"mask"``: np.ndarray or None, rendered silhouette mask

        Returns
        -------
        proxy : torch.Tensor, shape (19, resolution, resolution)
        """
        ...

    def extract_batch(self, images: list[np.ndarray], keypoints_batch: torch.Tensor, confidences_batch: torch.Tensor, prev_images: Optional[list[Optional[np.ndarray]]]=None, masks: Optional[list[Optional[np.ndarray]]]=None) -> torch.Tensor:
        """Extract proxy representations for a batch of images.

        Parameters
        ----------
        images : list of np.ndarray
            List of BGR uint8 images.
        keypoints_batch : torch.Tensor, shape (B, 16, 2)
        confidences_batch : torch.Tensor, shape (B, 16)
        prev_images : list of np.ndarray or None
        masks : list of np.ndarray or None

        Returns
        -------
        proxies : torch.Tensor, shape (B, 19, resolution, resolution)
        """
        ...
