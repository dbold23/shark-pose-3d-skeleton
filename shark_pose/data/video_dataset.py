"""
Video dataset: sliding window sequences for temporal models (BiLSTM + Skeletor).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """Sliding window sequences of proxy representations for temporal processing.

    Each sample is a clip of T consecutive frames with their proxy features
    and per-frame SPIN outputs (pose, shape, joints).
    """

    def __init__(self, data_dir: str | Path, sequence_length: int=16, stride: int=4, transform=None):
        """
        Args:
            data_dir: directory with per-video subdirectories
            sequence_length: number of frames per clip (T)
            stride: step between consecutive clips
            transform: optional transform applied to each clip
        """
        ...

    def _build_clip_index(self) -> list[dict]:
        """Scan video directories and create sliding window clips."""
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict:
        ...
