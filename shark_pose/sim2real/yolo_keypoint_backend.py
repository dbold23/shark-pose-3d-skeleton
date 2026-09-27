"""
YOLOv8-pose backend for the KeypointDetectorBackend protocol.

Wraps the trained v4 YOLOv8 detector (16 keypoints, 1 class white_shark)
from the shark-morphometrics sister project as a drop-in detector for
ProxyExtractor's real-image pipeline.

The model lives outside this repo; default weights path points at the
external location. See ``docs/real_data_assets.md`` for details.

Usage::

    from shark_pose.sim2real.yolo_keypoint_backend import YOLOKeypointBackend
    detector = YOLOKeypointBackend(weights_path=".../v4/weights/best.pt")
    keypoints, confidences = detector.predict(image_bgr)
    # keypoints: (16, 2) pixel coords in the original image space
    # confidences: (16,) per-keypoint confidence in [0, 1]
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
NUM_SHARK_KEYPOINTS = 16

class YOLOKeypointBackend:
    """Drop-in KeypointDetectorBackend backed by an Ultralytics YOLO model.

    Loaded lazily so importing this module doesn't require ultralytics
    until you actually instantiate the backend.

    Parameters
    ----------
    weights_path : str
        Path to a YOLOv8-pose .pt checkpoint (16 keypoints, shark class).
    device : str
        Inference device. ``"cpu"``, ``"cuda"``, ``"mps"``, or a specific
        cuda index like ``"cuda:0"``. Default ``"cpu"``.
    conf_threshold : float
        Minimum box confidence for a detection to be considered. Default 0.25.
    iou_threshold : float
        NMS IoU threshold. Default 0.45.

    Notes
    -----
    The backend expects YOLO output to provide both 2D keypoint locations
    and per-keypoint visibility/confidence values (``results[0].keypoints.xy``
    and ``.conf``). Both are required; if either is missing the backend
    falls back to zero outputs.

    If multiple sharks are detected in a single frame, the detection with
    the highest box confidence is returned (the model's primary subject).
    If no shark is detected, the backend returns zeroed keypoints + zeroed
    confidences — downstream proxy extraction will produce an effectively
    empty heatmap stack, which the SPIN model handles via the masking it
    was trained with.
    """

    def __init__(self, weights_path: str | Path, device: str='cpu', conf_threshold: float=0.25, iou_threshold: float=0.45) -> None:
        ...

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect 2D keypoints in a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape (H, W, 3). Other shapes accepted by
            ultralytics also work (RGB, etc.) since ultralytics handles
            color/format internally.

        Returns
        -------
        keypoints : np.ndarray, shape (NUM_SHARK_KEYPOINTS, 2)
            (x, y) pixel coordinates in the input image space. If no
            detection, all zeros.
        confidences : np.ndarray, shape (NUM_SHARK_KEYPOINTS,)
            Per-keypoint confidence in [0, 1]. If no detection, all zeros.
        """
        ...
