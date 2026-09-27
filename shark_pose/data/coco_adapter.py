"""
COCO format adapter for importing annotations from SharkScarAnnotator.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import torch
from torch import Tensor
from ..core.skeleton import SHARK_KEYPOINT_SEQUENCE, JOINT_NAME_TO_IDX, NUM_JOINTS

def load_coco_keypoints(coco_json_path: str | Path) -> list[dict]:
    """Load shark keypoint annotations from COCO-format JSON.

    Expected format (from SharkScarAnnotator export):
        categories[0].keypoints = ["snout_tip", "eye_center", ...]
        annotations[].keypoints = [x1, y1, v1, x2, y2, v2, ...]

    Args:
        coco_json_path: path to COCO JSON file

    Returns:
        List of dicts with:
            - image_id, file_name
            - joints_2d: Tensor (16, 2)
            - visibility: Tensor (16,) — COCO visibility codes
            - confidence: Tensor (16,) — normalized [0, 1]
            - bbox: optional (x, y, w, h)
    """
    ...

def coco_to_training_format(coco_json_path: str | Path, output_dir: str | Path) -> int:
    """Convert COCO annotations to per-frame JSON format for training.

    Args:
        coco_json_path: input COCO JSON
        output_dir: directory to write per-frame JSONs

    Returns:
        Number of annotations written.
    """
    ...
