"""
Dataset for loading pre-extracted proxy representations from synthetic renders.

NEVER loads raw RGB — only cached proxy tensors + full ground truth.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from ..core.constants import NUM_BETAS, NUM_JOINTS, POSE_DIM, TOTAL_PROXY_CHANNELS

class SyntheticDataset(Dataset):
    """Loads pre-extracted proxy representations from synthetic renders + full GT.

    Directory structure expected:
        data_dir/
            index.json          — maps render_id to file paths
            proxy_features/     — cached 19-channel proxy tensors (.npy)
            annotations/        — JSON files with GT pose, shape, joints, etc.
    """

    def __init__(self, data_dir: str | Path, split: str='train', resolution: int=256, transform=None):
        ...

    def _build_index(self) -> dict:
        """Build index by scanning the proxy_features directory."""
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict:
        ...
