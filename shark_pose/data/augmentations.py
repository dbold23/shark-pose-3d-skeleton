"""
Underwater-specific data augmentations for proxy representations.

Includes keypoint masking (occlusion recovery), input downsampling (noise filter),
and standard spatial augmentations.
"""
import random
import cv2
import numpy as np
import torch
from torch import Tensor

class KeypointMaskingAugmentation:
    """Copy-paste background over keypoint heatmap channels to simulate occlusion.

    Forces the network to learn "occlusion recovery" — handling fins obscured
    by body, reef, kelp, or murky water.
    """

    def __init__(self, max_masked_joints: int=4, mask_probability: float=0.5, mask_radius: int=16):
        ...

    def __call__(self, sample: dict) -> dict:
        ...

class CorrelatedMaskingAugmentation:
    """Simulate realistic occlusion patterns derived from 22K real detections.

    Instead of masking random joints, samples from 5 empirically-observed
    correlated occlusion groups weighted by their real-world frequency.
    """

    def __init__(self, mask_probability: float=0.8, max_extra_random: int=2):
        ...

    def __call__(self, sample: dict) -> dict:
        ...

class InputDownsampling:
    """Apply mild Gaussian blur + downsample-then-upsample.

    Acts as a filter to remove salt-and-pepper noise typical in
    low-quality or turbid underwater video.
    """

    def __init__(self, downsample_factor: int=2, blur_sigma: float=1.0, probability: float=0.3):
        ...

    def __call__(self, sample: dict) -> dict:
        ...

class UnderwaterColorJitter:
    """Underwater-specific augmentations on the edge/silhouette channel.

    Applies noise and minor perturbations to proxy features to improve robustness.
    """

    def __init__(self, edge_noise_std: float=0.05, flow_noise_std: float=0.02, probability: float=0.4):
        ...

    def __call__(self, sample: dict) -> dict:
        ...

class RandomFlip:
    """Random horizontal flip of proxy features + keypoints."""

    def __init__(self, probability: float=0.5):
        ...

    def __call__(self, sample: dict) -> dict:
        ...

class ComposeAugmentations:
    """Compose multiple augmentations."""

    def __init__(self, augmentations: list):
        ...

    def __call__(self, sample: dict) -> dict:
        ...

def get_train_augmentations(keypoint_masking: bool=True, downsampling: bool=True, color_jitter: bool=True, flip: bool=True) -> ComposeAugmentations:
    """Get default training augmentation pipeline."""
    ...
