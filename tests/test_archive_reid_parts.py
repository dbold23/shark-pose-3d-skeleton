"""Tests for the S8 re-ID v2 front end: body parts cut in one canonical frame, frame quality and diverse selection,
per-part identity features, part-grouped fusion with cue admission, and the whole chain from a synthetic video
to a ranked animal (``part_crops`` -> ``part_features_run`` -> ``reid`` loaders -> ``fuse_parts``).

Synthetic sharks (``synth_body``) have known identity on the dorsal fin, the caudal upper lobe and the pigment
line of each flank, and NONE on the pectoral fin or the caudal lower lobe, so those are the built-in negatives.
Thresholds guard regressions; they are not accuracy claims for real sharks."""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from shark_pose.archive import part_features as pf
from shark_pose.archive.fin_edge import match_score
from shark_pose.archive.frame_quality import Candidate, coverage, score_candidates, select_diverse, sharpness
from shark_pose.archive.parts import PARTS, body_frame, crop_part, part_affine, to_crop
from shark_pose.archive.reid_bench import Truth, build_truth
from shark_pose.archive.reid_system import Cue, admit, chance_rank10, crossfit_folds, fuse, fuse_parts, notch_cue, profile_cue, single
from shark_pose.archive.synth_body import random_body, render_frame

def _part(f, name):
    ...

@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('rot', [-25.0, 0.0, 30.0])
def test_body_frame_names_the_side_and_crops_are_canonical(side, rot):
    ...

def test_sharpness_falls_with_blur_and_ignores_gain():
    ...

def test_occlusion_lowers_caudal_coverage():
    ...

def test_select_diverse_spreads_over_sides_and_views():
    ...

def test_gates_drop_cut_and_clipped_frames():
    ...

def test_notches_found_and_pooled_across_frames():
    ...

def test_pigment_line_is_individual_and_differs_left_to_right():
    ...

def test_caudal_upper_margin_is_individual():
    ...

def test_sided_cues_never_compare_left_with_right():
    ...

def _cue(name, n=30, seed=0):
    ...

def test_fuse_parts_counts_a_part_once_however_many_cues_describe_it():
    ...

def test_fuse_is_unchanged_by_the_v2_refactor():
    ...

def test_admission_rule():
    ...

@pytest.fixture(scope='module')
def synth_run(tmp_path_factory):
    ...

def test_part_crops_write_every_part_from_both_sides(synth_run):
    ...

def test_video_to_ranked_animal(synth_run):
    ...

def test_cli_v2_bench_admits_only_informative_cues_and_identify_follows(tmp_path):
    ...
