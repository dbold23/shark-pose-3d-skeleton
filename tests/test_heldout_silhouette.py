"""S7 stage 1: the held-out silhouette scorer and the extractor's ``--offset`` gate.

The extractor's default (``offset 0``) must be bit-identical to the original
``frame_idx % stride == 0`` gate; ``offset 1`` at stride 2 must be exactly its
complement.  The scorer, fed masks that ARE the mesh's own renders, must report
IoU 1.0 on both sets and a gap of 0; fed a perturbed held-out mask, a positive
gap.  Writing the block into a record must change nothing else.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

def _load(name: str):
    ...

def test_offset_zero_is_the_original_stride_gate():
    ...

def test_offset_one_at_stride_two_is_exactly_the_complement():
    ...

def test_extractor_cli_default_offset_is_zero():
    ...

def test_partition_is_disjoint_and_counts_missing_masks(tmp_path):
    ...

@pytest.fixture(scope='module')
def assets():
    ...

def _record(n_frames: int) -> dict:
    ...

def _write_masks(sc, assets, record, bundle: Path, perturb_heldout: bool):
    ...

def test_own_renders_score_one_and_gap_zero(assets, tmp_path):
    ...

def test_perturbed_heldout_masks_open_a_positive_gap(assets, tmp_path):
    ...

def test_missing_heldout_dir_is_refused(assets, tmp_path):
    ...

def test_undistorted_record_is_refused():
    ...

def test_write_record_adds_one_key_and_nothing_else(assets, tmp_path):
    ...

def test_every_n_mismatch_with_the_record_is_refused(assets, tmp_path):
    ...

def test_mask_geometry_is_on_axis_for_a_centred_blob_and_at_the_corner_for_a_corner_pixel():
    ...

def test_score_block_carries_per_frame_covariates(assets, tmp_path):
    ...

def test_trunk_band_iou_is_one_on_own_renders_and_ignores_a_fin_perturbation(assets, tmp_path):
    """A mask perturbed only OUTSIDE the trunk band (a fin-sized blob far from the body) must
    lower the full IoU but leave the trunk IoU at 1.0 (prereg A3)."""
    ...

def test_score_block_carries_trunk_gap(assets, tmp_path):
    ...

def test_a_mask_larger_than_the_fitted_trunk_is_flagged_band_limited(assets, tmp_path):
    """Masks dilated well beyond the trunk (the too-small-mesh signature): the narrow band hides
    the excess, the wide band sees it, the record is flagged and the wide value is quoted (A4)."""
    ...
