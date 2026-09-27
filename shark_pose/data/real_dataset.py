"""
Dataset for loading proxy representations from real underwater video frames.

NEVER loads raw RGB — only pre-extracted proxy tensors + 2D keypoint annotations.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .synthetic_dataset import DEFAULT_FOCAL_PX

def _row_focal(ann: dict, dataset_focal: float | None) -> float:
    """Proxy focal in px for one annotation: its own, the set's, or the default."""
    ...

class RealDataset(Dataset):
    """Loads proxy representations extracted from real underwater frames + 2D keypoints.

    Directory structure expected:
        data_dir/
            proxy_features/     — cached 19-channel proxy tensors (.npy)
            annotations/        — COCO-format JSON or per-frame JSON with 2D keypoints
            <split>_index.json  — optional train/val split (maps sample_id -> paths)

    Without ``split`` the whole ``annotations/`` directory is used (backward
    compatible). With ``split`` set and ``<data_dir>/<split>_index.json`` present,
    only the sample ids listed in that index are used, in index order — so the
    held-out evaluation frames stay out of training.
    """

    def __init__(self, data_dir: str | Path, annotation_file: str | Path | None=None, resolution: int=256, transform=None, split: str | None=None, focal_px: float | None=None):
        """
        Args:
            focal_px: Proxy focal length in pixels to label every row with, for
                a set shot on one known lens (``CameraIntrinsics.proxy_focal``).
                A row whose annotation carries its own ``focal_length`` uses
                that instead. None keeps the pipeline default -- see the note
                in ``__getitem__``: that is a placeholder, not a measurement.
        """
        ...

    def _apply_split(self, sample_ids: list[str]) -> list[str]:
        """Restrict sample ids to a split index, if one is requested and present."""
        ...

    def _load_coco_annotations(self, ann_file: str | Path) -> dict:
        """Load COCO-format annotations (from SharkScarAnnotator export)."""
        ...

    def _scan_annotations(self) -> dict:
        """Scan for per-frame JSON annotations."""
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict:
        ...
