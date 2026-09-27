"""
Dataset for loading pre-extracted DINOv2 features + optional SAM2 masks.

Used by Experiment A (DINOv2 frozen) and Experiment D (hybrid).
Same GT fields as SyntheticDataset, but loads a (768,) feature vector
instead of a (19, 256, 256) proxy tensor.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class DINOFeatureDataset(Dataset):
    """Loads pre-extracted DINOv2 CLS features + full GT annotations.

    Directory structure expected:
        data_dir/
            train_index.json        — same index as SyntheticDataset
            dino_features/          — pre-extracted (768,) .npy vectors
            annotations/            — JSON with GT pose, shape, joints
            sam2_masks/ (optional)  — (1, 256, 256) .npy binary masks

    Returns sample dict with "features" key instead of "proxy".
    If load_masks=True, also includes "mask" key.
    """

    def __init__(self, data_dir: str | Path, split: str='train', load_masks: bool=False, domain: str='synthetic', transform=None):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict:
        ...
