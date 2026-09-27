"""Tests for shark_pose.sim2real.keypoint_mapping."""
import numpy as np
import pytest
from shark_pose.sim2real.keypoint_mapping import MAX_MAPPED_3D_JOINTS, SECONDARY_YOLO_INDICES, SKELETON_INDICES_WITHOUT_2D, SKELETON_PAIR_SOURCES, SKELETON_TO_YOLO, YOLO_KEYPOINT_NAMES, merge_pair, remap_yolo_to_skeleton, remap_yolo_to_skeleton_torch

class TestMapping:

    def test_skeleton_to_yolo_has_16_entries(self):
        ...

    def test_three_skeleton_joints_have_no_2d(self):
        ...

    def test_three_yolo_indices_are_secondary(self):
        ...

    def test_pair_sources(self):
        ...

    def test_max_mapped_is_thirteen(self):
        ...

    def test_no_yolo_index_used_twice(self):
        ...

    def test_yolo_names_have_16_entries(self):
        ...

class TestMergePair:

    def test_both_visible_gives_weighted_midpoint(self):
        ...

    def test_weighting_follows_confidence(self):
        ...

    def test_only_one_visible_returns_that_one(self):
        ...

    def test_both_zero_returns_zero(self):
        ...

class TestRemapNumpy:

    def test_known_joint_carries_through(self):
        ...

    def test_paired_joints_merge_front_and_back(self):
        ...

    def test_paired_joint_with_one_half_visible(self):
        ...

    def test_paired_joint_with_neither_half_visible(self):
        ...

    def test_binary_flags_reproduce_midpoint_rule(self):
        ...

    def test_unmapped_skeleton_joints_are_zero(self):
        ...

    def test_secondary_yolo_indices_feed_their_pair_only(self):
        ...

    def test_shape_validation(self):
        ...

class TestRemapTorch:

    def test_single_sample(self):
        ...

    def test_batch(self):
        ...

    def test_batched_torch_agrees_with_numpy(self):
        ...

    def test_single_torch_agrees_with_numpy(self):
        ...
