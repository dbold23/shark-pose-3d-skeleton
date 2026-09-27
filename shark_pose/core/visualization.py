"""
Visualization utilities: 3D mesh overlay, skeleton drawing, side-by-side comparisons.
"""
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
import torch
from torch import Tensor
from .skeleton import SHARK_KEYPOINT_SEQUENCE, BONE_PAIRS, NUM_JOINTS
BONE_COLOR = (200, 200, 200)

def draw_skeleton_2d(image: np.ndarray, joints_2d: np.ndarray, confidences: np.ndarray | None=None, confidence_threshold: float=0.3, joint_radius: int=4, bone_thickness: int=2) -> np.ndarray:
    """Draw 2D skeleton overlay on an image.

    Args:
        image: (H, W, 3) BGR image
        joints_2d: (16, 2) joint pixel coordinates
        confidences: (16,) optional per-joint confidences
        confidence_threshold: minimum confidence to draw a joint
        joint_radius: circle radius for joints
        bone_thickness: line thickness for bones

    Returns:
        image with skeleton overlay
    """
    ...

def draw_mesh_overlay(image: np.ndarray, vertices_2d: np.ndarray, faces: np.ndarray, alpha: float=0.3, color: tuple[int, int, int]=(100, 200, 255)) -> np.ndarray:
    """Draw semi-transparent mesh overlay on an image.

    Args:
        image: (H, W, 3) BGR image
        vertices_2d: (N, 2) projected vertex positions
        faces: (F, 3) face indices
        alpha: transparency (0=invisible, 1=opaque)
        color: mesh fill color (BGR)

    Returns:
        image with mesh overlay
    """
    ...

def create_comparison(image: np.ndarray, gt_joints: np.ndarray | None, pred_joints: np.ndarray, pred_confidences: np.ndarray | None=None) -> np.ndarray:
    """Create side-by-side comparison: original | GT skeleton | predicted skeleton.

    Args:
        image: (H, W, 3) original frame
        gt_joints: (16, 2) ground truth 2D joints (optional)
        pred_joints: (16, 2) predicted 2D joints
        pred_confidences: (16,) predicted confidences

    Returns:
        comparison image (H, W*2 or W*3, 3)
    """
    ...

def write_video(frames: list[np.ndarray], output_path: str, fps: int=30) -> None:
    """Write frames to video file.

    Args:
        frames: list of (H, W, 3) BGR images
        output_path: output video file path
        fps: frames per second
    """
    ...
