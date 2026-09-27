"""
Generalized SPIN model for encoder comparison experiments.

Identical to SPINModel but accepts any encoder from the factory instead
of hardcoding ProxyEncoder. The training loop is responsible for passing
the correct input to this model's forward().

Usage:
    encoder = create_encoder(cfg.encoder)
    model = ExperimentSPINModel(shark_model, encoder)
    output = model(encoder_input)  # tensor or dict depending on encoder
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..core.constants import CAMERA_DIM, NUM_JOINTS, NUM_SURFACE_LANDMARKS, SHAPE_DIM, POSE_DIM
from ..core.camera import PerspectiveCamera
from ..model_3d.shark_smpl import SharkSMPL, SharkSMPLOutput
from .iterative_regressor import IterativeRegressor

class ExperimentSPINModel(nn.Module):
    """SPIN architecture with a swappable encoder.

    Pipeline:
        encoder_input (varies by experiment)
            -> encoder -> features (B, encoder.output_dim)
            -> IterativeRegressor -> pose, shape, camera (per iteration)
            -> SharkSMPL -> vertices, joints, surface_landmarks

    The encoder can be any nn.Module with an ``output_dim`` property.
    """

    def __init__(self, shark_model: SharkSMPL, encoder: nn.Module, focal_length: float=1000.0, image_size: tuple[int, int]=(256, 256)):
        ...

    def forward(self, encoder_input, init_pose: Optional[Tensor]=None, init_shape: Optional[Tensor]=None, init_camera: Optional[Tensor]=None) -> Dict[str, Tensor | List[Dict[str, Tensor]]]:
        """Full SPIN forward pass with swappable encoder.

        Args:
            encoder_input: Whatever the encoder expects:
                - ProxyEncoder: (B, 19, H, W) proxy tensor
                - DINOv2FrozenEncoder: (B, 768) pre-extracted features
                - DINOv2LoRAEncoder: (B, 3, 224, 224) RGB images
                - HybridEncoder: dict with "features" and "mask" keys
            init_pose: (B, 48) optional initial axis-angle pose.
            init_shape: (B, 10) optional initial shape betas.
            init_camera: (B, 3) optional initial camera translation.

        Returns:
            Same dict as SPINModel: vertices, joints_3d, joints_2d,
            surface_landmarks, pose, shape, camera, pose_6d,
            rotation_uncertainty, per_iteration_outputs.
        """
        ...

    def _project_joints(self, joints_3d: Tensor, device: torch.device) -> Tensor:
        ...

    def encode(self, encoder_input) -> Tensor:
        """Extract features only (useful for downstream tasks)."""
        ...
