"""
Keypoint format converter between SharkScarAnnotator and the pose model.

Since both use the same 16-keypoint SHARK_KEYPOINT_SEQUENCE, conversion
is an identity mapping with format normalization (dict -> tensor).
"""
from __future__ import annotations
import torch
from torch import Tensor
from ..core.skeleton import SHARK_KEYPOINT_SEQUENCE, JOINT_NAME_TO_IDX, NUM_JOINTS
from ..core.constants import COCO_VISIBLE

def annotator_keypoints_to_tensor(keypoints: list[dict]) -> tuple[Tensor, Tensor]:
    """Convert annotator keypoint dicts to model tensors.

    Args:
        keypoints: list of {"name": str, "x": float, "y": float, "v": int, "confidence": float}

    Returns:
        joints_2d: (16, 2)
        confidences: (16,)
    """
    ...

def coco_keypoints_to_tensor(coco_kps: list[float]) -> tuple[Tensor, Tensor]:
    """Convert COCO-format flat keypoint list to model tensors.

    Args:
        coco_kps: [x1, y1, v1, x2, y2, v2, ...] — 48 values for 16 joints

    Returns:
        joints_2d: (16, 2)
        confidences: (16,)
    """
    ...

def tensor_to_coco_keypoints(joints_2d: Tensor, confidences: Tensor) -> list[float]:
    """Convert model tensors back to COCO-format flat list.

    Args:
        joints_2d: (16, 2)
        confidences: (16,) in [0, 1]

    Returns:
        COCO-format: [x1, y1, v1, x2, y2, v2, ...]
    """
    ...

def tensor_to_annotator_keypoints(joints_2d: Tensor, confidences: Tensor) -> list[dict]:
    """Convert model tensors to annotator keypoint dict format.

    Args:
        joints_2d: (16, 2)
        confidences: (16,)

    Returns:
        List of keypoint dicts compatible with SharkScarAnnotator.
    """
    ...
