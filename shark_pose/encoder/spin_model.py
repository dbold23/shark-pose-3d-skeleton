"""
SPIN (SMPL Parameter Inference Network) adapted for SharkSMPL.

Combines:
    1. ProxyEncoder (ResNet-50 on 19-channel proxy input)
    2. IterativeRegressor (IEF with 3 iterations)
    3. SharkSMPL (differentiable parametric shark model)

The model predicts pose, shape, and camera parameters from proxy
representations, then passes them through SharkSMPL to obtain 3D
vertices, joints, and surface landmarks.

Supports:
    - Multi-view shape fusion via beta mean pooling
    - Per-iteration intermediate outputs for adversarial training
    - Perspective projection of 3D joints to 2D for reprojection losses
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..core.constants import CAMERA_DIM, NUM_JOINTS, NUM_SURFACE_LANDMARKS, SHAPE_DIM, POSE_DIM
from ..core.camera import PerspectiveCamera
from ..model_3d.shark_smpl import SharkSMPL, SharkSMPLOutput, select_visible_landmarks
from .resnet_encoder import ProxyEncoder
from .iterative_regressor import IterativeRegressor

class SPINModel(nn.Module):
    """SPIN architecture for 3D shark pose and shape estimation.

    Pipeline:
        proxy_input (B, 19, H, W)
            -> ProxyEncoder -> features (B, 2048)
            -> IterativeRegressor -> pose, shape, camera (per iteration)
            -> SharkSMPL -> vertices, joints, surface_landmarks

    The model returns all intermediate iteration outputs to support
    per-iteration adversarial and reprojection losses during training.
    """

    def __init__(self, shark_model: SharkSMPL, pretrained_encoder: bool=True, focal_length: float=1000.0, image_size: tuple[int, int]=(256, 256), use_landmarks: bool=False, focal_conditioned: bool=False):
        """
        Args:
            shark_model: Pre-built SharkSMPL instance (holds mesh topology,
                joint regressor, skinning weights, etc.).
            pretrained_encoder: Whether to init encoder conv1[:3] from ImageNet.
            focal_length: Camera focal length in pixels for projection. Used for
                every sample that does not carry its own ``focal``.
            image_size: (H, W) input image resolution.
            use_landmarks: Emit ``joints_2d`` from the visible-flank OBSERVATION
                landmarks rather than the kinematic joints, matching what a human
                annotator labels. Default False: every trained checkpoint was
                supervised against the joint projection, so training defaults must
                not change. Only inference/fitting code should turn this on.
            focal_conditioned: Feed the per-sample focal into the regressor (see
                IterativeRegressor) and stamp the model as such. A conditioned
                model's translation is metric rather than expressed in the units
                of one fixed projection focal, so a fitter must NOT rescale its
                depth by a focal ratio -- it passes the focal in instead.
        """
        ...

    def forward(self, proxy_input: Tensor, init_pose: Optional[Tensor]=None, init_shape: Optional[Tensor]=None, init_camera: Optional[Tensor]=None, focal: Optional[Tensor]=None) -> Dict[str, Tensor | List[Dict[str, Tensor]]]:
        """Full SPIN forward pass.

        Args:
            proxy_input: (B, 19, H, W) proxy representation.
            init_pose: (B, 48) optional initial axis-angle pose.
            init_shape: (B, 10) optional initial shape betas.
            init_camera: (B, 3) optional initial camera translation.
            focal: (B,) per-sample focal length in proxy pixels. The 2D
                projection uses it per sample, and a focal-conditioned model
                also takes it as a regressor input. None means
                ``self.focal_length`` for the whole batch, which is exactly what
                every pre-focal checkpoint was trained with.

        Returns:
            Dictionary with:
                vertices: (B, V, 3) posed mesh vertices
                joints_3d: (B, 16, 3) 3D joint locations
                joints_2d: (B, 16, 2) projected 2D joint locations
                surface_landmarks: (B, 91, 3) dense surface landmarks
                pose: (B, 48) final axis-angle pose
                shape: (B, 10) final shape betas
                camera: (B, 3) final camera translation (tx, ty, tz)
                pose_6d: (B, 96) final 6D rotation pose
                rotation_uncertainty: (B, 16) per-joint kappa concentration
                per_iteration_outputs: list of dicts with intermediate results
        """
        ...

    def _resolve_focal(self, focal: Optional[Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Per-sample focal lengths, defaulting to the model's own.

        Returns:
            (B,) focal lengths in proxy pixels.
        """
        ...

    def project(self, points_3d: Tensor, focal: Optional[Tensor]=None) -> Tensor:
        """Project camera-frame points to proxy pixels, one focal per sample.

        Args:
            points_3d: (B, N, 3) points in the camera coordinate frame.
            focal: (B,) focal lengths in proxy pixels; None uses the model's
                ``focal_length`` for the whole batch. The principal point is the
                image centre, as it is everywhere in the pipeline.

        Returns:
            (B, N, 2) pixel coordinates.
        """
        ...

    def pop_clamp_stats(self) -> tuple[float, int]:
        """Fraction and count of joints the near-plane clamp touched, then reset.

        Counts every joint that went through :meth:`project` in ``train()`` mode
        since the previous call (all regressor iterations included); ``(0.0, 0)``
        when nothing was projected. Projection in ``eval()`` is not counted and
        leaves the module untouched.
        """
        ...

    def _project_joints(self, joints_3d: Tensor, device: torch.device, focal: Optional[Tensor]=None) -> Tensor:
        """Project 3D joints to 2D using perspective camera.

        Args:
            joints_3d: (B, 16, 3) joint positions in camera frame.
            device: Target device.
            focal: (B,) per-sample focal lengths; None uses the model's own.

        Returns:
            joints_2d: (B, 16, 2) projected 2D pixel coordinates.
        """
        ...

    def _project_observations(self, smpl_output: SharkSMPLOutput, device: torch.device, focal: Optional[Tensor]=None) -> Tensor:
        """Project the points the 2D supervision corresponds to.

        Kinematic joints by default; visible-flank observation landmarks when
        ``use_landmarks`` is on. Identical to ``_project_joints`` whenever the
        flag is off or the model carries no landmark asset.
        """
        ...

    def load_state_dict_padded(self, state_dict, strict: bool=True) -> List[str]:
        """Load weights that may predate the regressor's focal input column.

        A focal-conditioned regressor's ``fc1`` / ``residual_proj`` take one
        extra input column, appended LAST. Zero-padding a pre-focal weight
        matrix there is exact: the focal feature of the reference focal is
        ``log(1) = 0``, so the padded model reproduces the original bit for bit
        at focal 1000 and starts training with the focal input ignored rather
        than randomised.

        Args:
            state_dict: weights to load, of either shape.
            strict: passed through to ``load_state_dict``.

        Returns:
            The keys that were zero-padded (empty when the shapes already match).
        """
        ...

    @staticmethod
    def pool_shape(betas_list: List[Tensor]) -> Tensor:
        """Multi-view beta mean pooling.

        Averages shape parameters across multiple views of the same individual
        to enforce shape consistency.

        Args:
            betas_list: List of (B, 10) shape tensors from different views.

        Returns:
            pooled_betas: (B, 10) averaged shape parameters.
        """
        ...

    def encode(self, proxy_input: Tensor) -> Tensor:
        """Extract features only (useful for downstream tasks).

        Args:
            proxy_input: (B, 19, H, W) proxy representation.

        Returns:
            features: (B, 2048) feature vector.
        """
        ...
