"""
Stage 2 mixed-data training tests.

Covers the three things that were silently broken before:
    1. RealDataset ignored the train/val split, so the held-out frames were
       inside the training set.
    2. The default DataLoader collate raised KeyError on any mixed batch, so
       Stage 2 never actually trained on real data.
    3. MixedDataset's sampler weights were never used, so real frames would
       have been ~4% of an epoch.

Plus one forward + loss + backward step through the exact masked-loss helper
that train_stage2 uses.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import pytest
import torch
from shark_pose.core.constants import NUM_BETAS, NUM_JOINTS, POSE_DIM
from shark_pose.data.mixed_dataset import MixedDataset, make_mixed_sampler, mixed_collate
from shark_pose.data.real_dataset import RealDataset
from shark_pose.data.synthetic_dataset import SyntheticDataset

def _load_train_spin():
    """Import scripts/train_spin.py (not a package module) for its helper."""
    ...

@pytest.fixture(scope='module')
def synthetic_item():
    ...

@pytest.fixture(scope='module')
def real_item():
    ...

class _StubDataset(torch.utils.data.Dataset):
    """Minimal fixed-length dataset — the sampler only needs len()."""

    def __init__(self, n: int):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> int:
        ...

class TestRealDatasetSplit:

    def test_split_counts(self):
        ...

    def test_splits_are_disjoint_and_ordered(self):
        ...

    def test_missing_split_index_falls_back_to_scan(self):
        ...

class TestMixedCollate:

    def test_uniform_batch_from_mixed_items(self, synthetic_item, real_item):
        ...

    def test_synthetic_row_gets_unit_confidence(self, synthetic_item, real_item):
        ...

    def test_real_row_gets_zero_filled_3d_fields(self, synthetic_item, real_item):
        ...

    def test_default_collate_would_have_crashed(self, synthetic_item, real_item):
        ...

class TestMixedSampler:

    def test_draw_proportions_match_synthetic_weight(self):
        ...

    def test_without_sampler_real_would_be_swamped(self):
        ...

class TestStage2BatchLosses:

    def test_step_on_mixed_batch(self, shark_model, synthetic_item, real_item):
        ...

    def test_3d_terms_vanish_without_3d_gt(self, shark_model, real_item):
        ...
