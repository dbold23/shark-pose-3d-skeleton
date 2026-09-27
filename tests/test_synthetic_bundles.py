"""Tests for ``scripts/generate_synthetic_bundles.py`` (S7 stage 2).

Everything here is CPU, tiny (12-14 frames at 192x108 or 320x180) and runs no
model: the only network evaluated is ``SharkSMPL`` forward on known parameters,
which is what the generator itself does.  No YOLO, no SAM, no fit.

The corpus draw pool is built once per session (it reads 42 fit records) and
shared through a module-scoped fixture, so a test costs one generator run.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture(scope='module')
def corpus_cache(tmp_path_factory) -> Path:
    """One pooled-draws cache for the whole module (42 records read once)."""
    ...

def run(out: Path, corpus_cache: Path, *extra: str) -> Path:
    ...

def only_bundle(out: Path) -> Path:
    ...

def test_same_seed_gives_identical_population_and_detections(tmp_path, corpus_cache):
    ...

def test_girth_draws_in_range_with_bounded_first_differences():
    ...

def test_written_girth_field_obeys_the_same_bounds(tmp_path, corpus_cache):
    ...

def test_noise_off_mask_equals_hard_silhouette_of_the_truth(tmp_path, corpus_cache):
    ...

def test_noise_off_keypoints_are_the_projected_landmarks(tmp_path, corpus_cache):
    ...

def test_mask_directories_are_even_and_odd_and_agree_with_their_index(tmp_path, corpus_cache):
    ...

def test_clip_meta_passes_the_contract(tmp_path, corpus_cache):
    ...

def test_truth_record_loads_and_rebuilds_the_truth_mesh(tmp_path, corpus_cache):
    ...

def test_lens_arm_moves_the_periphery_more_than_the_centre():
    ...

def test_lens_and_pinhole_arms_differ_in_the_written_keypoints(tmp_path, corpus_cache):
    ...

@pytest.mark.parametrize('classes,target', [(('oblique', 'broadside'), gsb.CONTRAST_OBLIQUE), (('broadside', 'broadside'), gsb.CONTRAST_BROADSIDE)])
def test_pair_sampler_lands_in_the_requested_contrast_band(tmp_path, corpus_cache, classes, target):
    ...

def test_bundle_carries_every_file_the_fitter_discovers(tmp_path, corpus_cache):
    ...

def test_manifest_and_population_report_denominators(tmp_path, corpus_cache):
    ...

def test_conf_curve_schema_and_draw():
    ...

def test_truncated_normal_respects_its_interval():
    ...

def test_achieved_keypoint_visibility_matches_the_corpus_targets(tmp_path, corpus_cache):
    """prereg A3: the synthetic DETECTIONS reproduce the corpus's per-keypoint
    visibility rate (conf >= 0.5) on a 40-window set, within +-0.1."""
    ...

def test_generator_runs_on_model_assets_v2(tmp_path, corpus_cache):
    """prereg A3.1: the box run passes --assets data/model_assets_v2, which is
    what the shipped fitter and the morph extractor load
    (fit_video.MODEL_ASSET_DIRS['v2']).  v2 has no joint regressor -- its rest
    joints come from bone_positions.npy and do NOT follow betas -- so it is a
    different code path through SharkSMPL and is exercised here."""
    ...

def test_noise_on_injects_the_pre_registered_keypoint_sigma(tmp_path, corpus_cache):
    """The injected error is the prereg's 12 px in the proxy, and the written
    confidence is not a constant."""
    ...
