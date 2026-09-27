"""Canonical locations of the 2D YOLOv8-pose detector weights.

v5 (`v5_oks`) is the primary detector and the default everywhere. v4 is
superseded and kept only so older experiments stay reproducible.

Override the default at runtime with the ``SHARK_YOLO_WEIGHTS`` environment
variable (useful on machines where the sister repo lives elsewhere).
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

def default_yolo_weights() -> str:
    """Resolve the default detector weights path.

    Returns ``$SHARK_YOLO_WEIGHTS`` when set, otherwise :data:`V5_OKS_WEIGHTS`.
    Logs a warning (but still returns the path) when it does not exist, so
    ``--yolo-weights`` misconfiguration surfaces before the model loads.
    """
    ...
__all__ = ['V5_OKS_WEIGHTS', 'V4_WEIGHTS', 'default_yolo_weights']
