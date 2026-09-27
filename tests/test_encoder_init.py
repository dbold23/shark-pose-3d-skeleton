"""
Regression tests for the conv1 initialisation of `ProxyEncoder`.

Background (2026-09-03 audit).  conv1 used to be zeroed for every input
channel with the ImageNet RGB filters copied into channels 0-2 only.  After
training, channels 3-15 carried 19x (round_004) to 608x (stage 1) less
weight than 0-2 and channels 16-18 were exactly 0, so the encoder barely
saw the keypoint heatmaps.  These tests pin the replacement init.
"""
from __future__ import annotations
from pathlib import Path
import pytest
import torch
from shark_pose.core.constants import NUM_KEYPOINT_CHANNELS, TOTAL_PROXY_CHANNELS
from shark_pose.encoder.resnet_encoder import ProxyEncoder, conv1_channel_balance

def _imagenet_conv1_weight() -> torch.Tensor:
    """The reference ImageNet conv1 weight, or skip if it is unavailable."""
    ...

def _pretrained_encoder() -> ProxyEncoder:
    ...

class TestConv1ChannelBalance:
    """The module-level helper reused by scripts/diagnose_shift_response.py."""

    def test_uniform_weight_has_ratio_one(self):
        ...

    def test_reproduces_the_old_zero_init_signature(self):
        """The historical init: RGB in 0-2, everything else exactly zero."""
        ...

    def test_known_imbalance_ratio(self):
        ...

class TestPretrainedInit:

    def test_all_channels_carry_equal_weight(self):
        ...

    def test_matches_scaled_mean_rgb_filter(self):
        """Each channel == mean of the three RGB filters * 3 / in_channels."""
        ...

    def test_uniform_input_matches_original_rgb_scale(self):
        """A spatially uniform input yields the original pre-activation."""
        ...

    def test_flow_and_edge_channels_are_initialised_too(self):
        ...

class TestRandomInit:

    def test_not_zeroed(self):
        ...

    def test_channels_are_balanced(self):
        ...

    def test_forward_returns_finite_features(self):
        ...

class TestCheckpointCompatibility:
    """The init only decides the starting point; loading must be unaffected."""

    @pytest.mark.skipif(not ROUND_004_CHECKPOINT.exists() or not MODEL_ASSETS.exists(), reason='round_004 checkpoint / model assets not present')
    def test_round_004_state_dict_loads_into_fresh_model(self):
        ...
