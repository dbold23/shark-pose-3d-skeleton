"""
Bridge to SharkScarAnnotator: read annotation data from its SQLite database.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional
import torch
from torch import Tensor
from ..core.skeleton import SHARK_KEYPOINT_SEQUENCE, NUM_JOINTS

class AnnotatorBridge:
    """Import encounters and keypoints from SharkScarAnnotator's SQLite database.

    The annotator stores data in database/catalog.db with encounter records
    containing keypoint annotations in COCO format.
    """

    def __init__(self, annotator_root: str | Path):
        """
        Args:
            annotator_root: path to the shark-scar-annotator/ repo directory
        """
        ...

    def get_connection(self) -> sqlite3.Connection:
        ...

    def get_encounters(self) -> list[dict]:
        """Get all annotated encounters with their metadata."""
        ...

    def get_keypoints_for_encounter(self, encounter_code: str) -> list[dict]:
        """Get keypoint annotations for a specific encounter.

        Returns list of frame-level keypoint dicts with joints_2d and confidences.
        """
        ...

    def _parse_keypoints(self, keypoints: list[dict]) -> tuple[Tensor, Tensor]:
        """Parse annotator keypoint format to tensors.

        The annotator stores keypoints as list of dicts:
            [{"name": "snout_tip", "x": 100, "y": 200, "v": 2, "confidence": 1.0}, ...]

        Args:
            keypoints: list of keypoint dicts from annotator

        Returns:
            joints_2d: (16, 2) tensor
            confidences: (16,) tensor
        """
        ...

    def get_all_annotated_keypoints(self) -> list[dict]:
        """Get all keypoint annotations across all encounters."""
        ...

    def export_to_training_format(self, output_dir: str | Path) -> int:
        """Export all annotations to per-frame JSON format for training.

        Args:
            output_dir: directory to write per-frame annotation JSONs

        Returns:
            Number of annotations exported.
        """
        ...
