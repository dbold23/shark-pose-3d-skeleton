"""Tests for S8 M1: pelvic crop geometry, the target video list, and the pre-registered sex probe on synthetic
embeddings with known truth (a real signal passes, no signal fails, a site-only signal is caught by the baseline)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import sex_probe as sp
from clasper_crops import ANAL, PELVIC, pelvic_quad, span_score
from m1_videos import select

def _kp(p, q):
    ...

def test_span_score_gates_confidence_and_span():
    ...

def test_pelvic_quad_rectifies_along_the_pelvic_anal_line():
    ...

def test_video_list_caps_per_key_longest_first():
    ...

def test_folds_are_grouped_and_stratified():
    ...

def test_logistic_fit_is_at_its_optimum():
    ...

def test_key_descriptor_needs_two_crops():
    ...

def _world(seed, signal, site_signal=0.0, n_animals=320, n_unknown=60, dim=96):
    """Animals with 1-2 keys; key descriptor = signal * sex direction + site_signal * site direction + noise.
    Sex is 60 % M; with site_signal the sex ratio also differs by site (the crews' sampling confound)."""
    ...

def test_a_real_signal_passes_and_publishes_only_confident_unknowns():
    ...

def test_no_signal_fails_and_writes_nothing():
    ...

def test_a_site_only_signal_is_caught_by_the_baseline_gate():
    """The embedding carries only the site; sex differs by site. Pooled accuracy looks useful, but the metadata
    baseline does as well and the within-site strata fall to chance."""
    ...

def test_too_few_animals_fails_even_with_a_perfect_signal():
    ...
