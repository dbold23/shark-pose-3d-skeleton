"""S7 stage 3 CPU tools: the C11 best_span producer and the grid todo builder.

Contract under test:
  * `make_grid_todo.py` re-targets ONLY the OUTDIR of the box4 stage A todo --
    a generated C00 todo differs from `box4/stageA_todo_n1.txt` in field 5 and
    nowhere else;
  * a config whose manifest `runner_extra.clip_frames == "best_span"` (C11)
    carries the `--clip-frames` flag in a 7th field for exactly those windows
    that have an EMITTED span, and no 7th field for the rest;
  * `best_span_replicates.py` emits the pre-registered row schema, and its span
    and refusal come from the shipped `window_gate` functions rather than from
    anything re-implemented in the lane.

No model, no fit and no detector runs here: the synthetic sidecars are built
from the corpus scan's own proportion table so that `rescore.my_cue` sees a
clean along-axis fit.
"""
import importlib.util
import json
import os
import subprocess
import sys
import numpy as np
import pytest

def _load(name):
    ...

@pytest.fixture(scope='module')
def todo_mod():
    ...

@pytest.fixture(scope='module')
def span_mod():
    ...

@pytest.fixture
def fake_inputs(tmp_path):
    ...

def _lines(path):
    ...

def test_outdir_is_the_only_rewrite(todo_mod, fake_inputs):
    ...

def test_c11_flags_only_the_emitted_spans(todo_mod, fake_inputs):
    ...

def test_c00_differs_from_stage_a_only_in_outdir(todo_mod, tmp_path):
    """The real manifest and the real 42-job stage A todo."""
    ...

def _synthetic_npz(path, window_id, video, start, end, detected, stride=2):
    """A sidecar whose frames sit exactly on the scan's proportion table.

    `detected` is a per-scored-frame bool; a False frame has n_det = 0, which is
    what makes a frame unscored (window_gate.FrameStats docstring).
    """
    ...

def _bundle(root, ind, window, video, start, n_frames):
    ...

@pytest.fixture
def synthetic_corpus(tmp_path):
    """Three windows: a mid-window span, an all-good window, one with no sidecar."""
    ...

def test_best_span_json_schema_and_denominators(span_mod, synthetic_corpus):
    ...

def test_span_and_refusal_come_from_window_gate(span_mod, synthetic_corpus):
    """The span is window_gate's, not the lane's: recompute it from the module."""
    ...

def test_sidecar_name_follows_the_stage_c_pattern(span_mod):
    ...

@pytest.mark.parametrize('script', ['run_grid_box.sh', 'sync_grid_home.sh'])
def test_box_scripts_parse(script):
    ...

def test_config_digest_audit_reproduces_the_fitter_stamp_and_flags_a_clobbered_cell(tmp_path):
    """The audit's digest must equal fit_individual's stamp on the shipped config
    (ed89f4e438edfccd on every journalled fit, ledger/fits.csv), and a record
    fitted under another YAML must be reported as a mismatch."""
    ...
