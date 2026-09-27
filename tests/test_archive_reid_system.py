"""Tests for the S8 re-ID system (shark_pose.archive.reid_system + s8/reid): cue rules, fusion, ranking,
confirmed links, flank crop geometry, and the CLI on the real shape table. Synthetic worlds have known identity;
thresholds guard regressions, they are not accuracy claims for real sharks."""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from shark_pose.archive.reid_bench import Truth, build_truth, evaluate_similarity
from shark_pose.archive.reid_system import Cue, Scar, fin_cue, fuse, pigment_cue, rank_animals, scar_pair_score, shape_cue, single
from shark_pose.archive.body_coords import N_THETA, N_U

def _world(seed=0, n_linked=40, n_single=40):
    ...

def _mrr(fz, truth):
    """Mean reciprocal rank over EVERY linked encounter; one the ranker cannot score counts 0."""
    ...

def test_fusion_beats_every_single_cue():
    ...

def test_fused_scores_have_no_nan_and_no_shared_cue_ranks_last():
    ...

def test_pigment_never_compares_left_with_right():
    ...

def test_shape_is_a_tie_breaker_not_an_override():
    ...

def test_shape_cue_is_scale_invariant_by_its_within_encounter_covariance():
    ...

def _cov(u0=0.0, u1=1.0, t0=-180.0, t1=180.0):
    ...

def test_scar_rules():
    ...

def test_rank_animals_rules():
    ...

def test_confirmed_links_grow_the_truth_and_none_changes_nothing():
    ...

def test_flank_crop_is_canonical_and_names_the_side():
    ...

def test_cli_bench_and_identify_on_the_real_shape_table(tmp_path):
    ...

def test_cli_bench_reads_pigment_and_scar_tables(tmp_path):
    """Crop dirs, an embedding npz and an S1-style scar run for real ledger videos, identity planted."""
    ...
