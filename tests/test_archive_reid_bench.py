"""Tests for the S8 re-ID benchmark scorer on synthetic known-truth data.

A synthetic ledger holds linked animals (several encounters each) and unlinked
single-encounter animals. Descriptors are an identity vector plus noise, so the
expected behaviour is known: a noiseless descriptor ranks every true mate first,
a random one sits near chance, and a same-encounter duplicate can never count.
"""
from __future__ import annotations
import csv
import numpy as np
import pytest
from shark_pose.archive.reid_bench import build_truth, encounter_descriptors, evaluate, evaluate_similarity, load_embeddings

def _write_ledger(tmp_path, n_linked=8, enc_per=3, n_single=40, vids_per_enc=2, seed=0):
    ...

def _embed(truth, videos, noise, dim=32, seed=1):
    ...

def test_truth_counts(tmp_path):
    ...

def test_noiseless_descriptor_ranks_every_mate_first(tmp_path):
    ...

def test_random_descriptor_is_near_chance(tmp_path):
    ...

def test_same_encounter_duplicate_never_counts(tmp_path):
    """A second video of the probe's own encounter is identical to the probe: it must not be a match."""
    ...

def test_sex_filter_only_removes_conflicting_sex(tmp_path):
    ...

def test_embedding_csv_roundtrip_and_unknown_rows(tmp_path):
    ...

def test_ties_rank_against_the_probe(tmp_path):
    """prereg_v2: ranks are pessimistic. A matcher that scores every pair alike (two fins without notches)
    must not put a mate at rank 1, and AP must count the tied non-mates ahead of the mates."""
    ...
