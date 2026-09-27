"""Regression tests for the LBS vertex path (shark_pose/model_3d/skinning.py).

Before 2026-09-03 the vertex path composed world_T @ rest_inv although local
transforms already rotate about the rest joint, so posed vertices collapsed
toward the origin while joints (world_T alone) stayed correct.
"""
from pathlib import Path
import numpy as np
import pytest
import torch
from shark_pose.core.skeleton import BONE_PAIRS, PARENT_INDICES
from shark_pose.core.transforms import axis_angle_to_rotation_matrix
from shark_pose.model_3d.skinning import lbs

def _rotmats(pose_aa):
    ...

@pytest.fixture(scope='module')
def assets():
    ...

def test_rest_pose_reproduces_template(assets):
    ...

def test_rest_extent_matches_template(assets):
    ...

def test_posed_joints_and_bone_lengths_rigid(assets):
    ...

def test_vertices_follow_their_joint_rigidly(assets):
    """A vertex fully bound to one joint must move with that joint's world transform."""
    ...

def test_mesh_does_not_collapse_under_pose(assets):
    ...
