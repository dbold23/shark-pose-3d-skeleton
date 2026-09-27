"""
Dataset for loading RGB images directly (for DINOv2 LoRA fine-tuning).

Used by Experiment B where the encoder needs raw images in the computation
graph for gradient flow through LoRA adapters.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class RGBDataset(Dataset):
    """Loads RGB images + GT annotations for online encoder forward.

    Directory structure expected:
        data_dir/
            train_index.json     — sample index
            renders/             — RGB images ({sample_id}.png)
            annotations/         — JSON with GT pose, shape, joints

    Returns sample dict with "image" key: (3, 224, 224) normalized tensor.
    """

    def __init__(self, data_dir: str | Path, split: str='train', image_size: int=224, domain: str='synthetic', image_subdir: str='renders', transform=None):
        ...

    def _find_image(self, sample_id: str) -> Optional[Path]:
        """Find image file for a sample (supports .png and .jpg)."""
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict:
        ...
