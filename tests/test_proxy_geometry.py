"""Tests for the shared letterbox proxy geometry and heatmap sigma constant."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from shark_pose.core.constants import DEFAULT_HEATMAP_SIGMA
from shark_pose.sim2real.proxy_extractor import ProxyExtractor
from shark_pose.sim2real.proxy_geometry import image_to_proxy_xy, letterbox_image, letterbox_params, proxy_to_image_xy
RESOLUTION = 256

@pytest.mark.parametrize('width,height', SIZES)
def test_letterbox_params_formula(width, height):
    ...

def test_letterbox_params_square_is_identity_scale():
    ...

def test_letterbox_params_rejects_bad_sizes():
    ...

@pytest.mark.parametrize('width,height', SIZES)
def test_roundtrip_numpy(width, height):
    ...

@pytest.mark.parametrize('width,height', SIZES)
def test_roundtrip_torch(width, height):
    ...

def test_numpy_and_torch_agree():
    ...

def test_square_input_is_pure_scale():
    ...

def test_letterbox_centres_the_content():
    ...

def test_image_to_proxy_does_not_mutate_input():
    ...

def test_letterbox_image_shape_and_padding_position():
    ...

def test_letterbox_image_square_is_pure_resize():
    ...

def test_letterbox_image_2d_mask():
    ...

def test_letterbox_image_rejects_bad_ndim():
    ...

def test_default_heatmap_sigma_is_three():
    ...

def test_keypoints_only_zeroes_flow_and_edge_channels():
    ...

def test_keypoints_only_matches_full_pipeline_on_heatmap_channels():
    ...
