"""
Adapter wrapping SharkScarAnnotator's SAM2Segmenter for silhouette extraction.

Produces binary shark body masks for the edge/silhouette proxy channel.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

class SegmentationAdapter:
    """Wraps SAM2Segmenter to produce silhouette masks for edge proxy features.

    The silhouette mask is used as input to the edge feature extractor
    (channel 18 of the 19-channel proxy representation).
    """

    def __init__(self, annotator_root: str | Path):
        """
        Args:
            annotator_root: path to the shark-scar-annotator/ repo directory
        """
        ...

    @property
    def available(self) -> bool:
        ...

    def initialize(self) -> None:
        """Lazy-load the SAM2 model."""
        ...

    def segment_shark(self, image_path: str | Path, bbox: tuple[int, int, int, int] | None=None, point_coords: list[tuple[int, int]] | None=None) -> np.ndarray:
        """Segment the shark body from an image.

        Args:
            image_path: path to the frame image
            bbox: optional (x, y, w, h) bounding box prompt
            point_coords: optional point prompts [(x, y), ...]

        Returns:
            mask: (H, W) binary numpy array
        """
        ...

    def get_silhouette_tensor(self, image_path: str | Path, bbox: tuple[int, int, int, int] | None=None, target_size: tuple[int, int]=(256, 256)) -> Tensor:
        """Get shark silhouette as a torch tensor for the proxy pipeline.

        Args:
            image_path: path to frame image
            bbox: bounding box prompt
            target_size: (H, W) to resize to

        Returns:
            silhouette: (1, H, W) float tensor (0 or 1)
        """
        ...
