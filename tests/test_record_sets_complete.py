"""A run directory that holds fit records must be declared as a re-aggregation arm.

Regression guard for the defect s0/consolidate found on 2026-09-15 and s1/fold_patches
folded into the shipped scripts: `run_box2` -- a whole A100 box arm, 143 record
directories and 147 window records -- sat on disk with fit records and a shipped `morph/`
aggregation while being named nowhere in

    outputs/.../individuals/reaggregate/src/reagg_common.py :: RECORD_SETS
    outputs/.../individuals/ledger/src/common.py            :: P

so every re-aggregation and every ledger build silently skipped it.  The ledger published
107 individuals when 249 had records.  Nothing failed; the arm was simply invisible.

This test discovers the record-bearing run directories from the FILESYSTEM and fails if
any of them is absent from those two declarations.  It is skipped when the programme
artefact tree is not present (the tree lives under `outputs/`, which is not in git).
"""
from __future__ import annotations
import glob
import importlib.util
import os
import pytest

def _load(path, name):
    ...

def record_bearing_runs():
    """Run directories under WORK that hold at least one fit record, from disk alone.

    A record is `<run>/<record dir>/<fit dir>/individual_*.json` -- the shape every arm
    writes (run1 `fit/` and `fit_stamped/`, run_box and run_box2 `fit/`).
    """
    ...

def test_every_record_bearing_run_is_a_reaggregation_arm():
    ...

def test_every_record_bearing_run_is_a_ledger_path():
    ...

def test_run_box2_specifically_is_declared():
    """The arm that was actually lost, pinned by name."""
    ...
