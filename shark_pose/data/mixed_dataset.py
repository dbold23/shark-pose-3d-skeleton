"""
Mixed dataset: weighted sampling from synthetic + real datasets.

Both datasets provide proxy tensors (never raw RGB).

Synthetic and real items carry different keys (synthetic has full 3D GT, real
has only 2D keypoints + confidences), so a mixed batch needs `mixed_collate`
below — the default collate raises KeyError on the first mixed batch.
"""
from typing import Optional
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from ..core.constants import CAMERA_DIM, NUM_BETAS, NUM_JOINTS, POSE_DIM
from .synthetic_dataset import DEFAULT_FOCAL_PX, SyntheticDataset
from .real_dataset import RealDataset

class MixedDataset(Dataset):
    """Combines synthetic and real datasets with configurable sampling weights.

    During each epoch, samples are drawn from synthetic and real datasets
    according to the specified ratio.
    """

    def __init__(self, synthetic_dataset: SyntheticDataset, real_dataset: RealDataset, synthetic_weight: float=0.5):
        """
        Args:
            synthetic_dataset: dataset of synthetic proxy tensors + full GT
            real_dataset: dataset of real proxy tensors + 2D annotations
            synthetic_weight: probability of sampling from synthetic (0-1)
        """
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict:
        ...

    @staticmethod
    def get_sampler_weights(syn_len: int, real_len: int, synthetic_weight: float=0.5) -> list[float]:
        """Get per-sample weights for WeightedRandomSampler.

        Args:
            syn_len: number of synthetic samples
            real_len: number of real samples
            synthetic_weight: desired proportion of synthetic samples

        Returns:
            weights: list of per-sample weights
        """
        ...

def mixed_collate(items: list[dict]) -> dict:
    """Collate synthetic and real items into one uniform batch dict.

    SyntheticDataset items carry {pose, betas, camera, joints_3d, joints_2d,
    has_3d_gt=True}; RealDataset items carry {joints_2d, confidences,
    has_3d_gt=False}. The default collate raises KeyError on any mixed batch.
    Here every tensor is present for every row: fields a row does not have are
    filled with zeros (synthetic confidences are ones — full GT), and
    ``has_3d_gt`` tells the loss which rows the 3D terms may use.

    Top-level function so DataLoader workers can pickle it.

    Returns:
        proxy (B, 19, H, W) float, joints_2d (B, 16, 2), confidences (B, 16),
        joints_3d (B, 16, 3), pose (B, 48), betas (B, 10), camera (B, 3),
        focal (B,), has_3d_gt (B,) bool, plus the lists ``domain`` and
        ``sample_id``.
    """
    ...

def make_mixed_sampler(mixed: MixedDataset, synthetic_weight: float, generator: Optional[torch.Generator]=None) -> WeightedRandomSampler:
    """Build the WeightedRandomSampler that enforces the synthetic/real mix.

    Without it a MixedDataset epoch is just concatenation, so 416 real frames
    against 10K synthetic would be ~4% of the samples seen.

    Args:
        mixed: the MixedDataset to sample from.
        synthetic_weight: target fraction of drawn samples that are synthetic.
        generator: optional torch.Generator for reproducible draws.

    Returns:
        WeightedRandomSampler over len(mixed) samples, with replacement.
    """
    ...
