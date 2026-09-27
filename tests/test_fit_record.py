"""TASKS C1-C5 -- what a fit RECORD has to carry to be readable afterwards.

Five defects, one file:

* **C1 / R9** -- a lateral-axis refusal has two causes that no published record
  can currently tell apart: the animal was not rolled (GEOMETRY) or the mask was
  too small to see the roll it had (RESOLUTION).  The five scan measurements
  that separate them are dropped on the way into the fit; ``window_attribution``
  carries them in, and ``gate_rejections`` says how many frames survived the
  frame gate and what took the rest.
* **C2 / R3** -- covered in ``test_joint_fit.py``: the pairwise separation.
* **C3 / R6** -- an absolute read off a record whose gate bars, commit and
  wells mode are unknown is unreadable.
* **C4 / R8** -- ``--clip-frames ID=start:end``, so a window whose useful span
  does not start at frame 0 no longer has to be physically re-cut.
* **C5 / R7** -- ``--no-video``, and the record is untouched by it.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest

def _clip_dir(tmp_path: Path, quality: dict | None) -> Path:
    ...

def test_window_attribution_reads_the_scan_row_off_the_clip_sidecar(tmp_path):
    ...

def test_window_attribution_falls_back_to_window_type(tmp_path):
    """Every corpus sidecar has carried the class as ``window_type`` since the
    first window was cut; only the four scan numbers are genuinely new."""
    ...

def test_window_attribution_override_wins_over_the_sidecar(tmp_path):
    ...

def test_window_attribution_says_so_when_it_has_nothing(tmp_path):
    """A null that is not explained is the defect, not the fix."""
    ...

def _synthetic_record(path: Path, spine: list[int], n_joints: int) -> Path:
    """A record with one frame per gate outcome, and one that survives.

    Frame 0 survives; 1 is undetected; 2 has a residual far outside the clip's
    robust spread; 3 has its snout inside the edge margin; 4 is 45 deg oblique.
    """
    ...

def test_gate_rejections_scores_every_clause_off_the_record(tmp_path):
    ...

@pytest.mark.skipif(not E1.exists(), reason='the shipped E1 records are absent')
def test_gate_rejections_reproduces_the_shipped_e1_counts():
    """C1's acceptance test, without a refit.

    The frame gate is a pure function of the record, so the counts the E1
    morphometrics record published -- ``{detection 24, residual 46,
    truncation_edge 113, obliqueness_broadside_only 343}`` over
    ``n_frames 545 -> n_gated 362`` -- must come back off the two records
    themselves.
    """
    ...

def test_bars_stamp_is_the_gate_the_fit_will_be_judged_by():
    ...

def test_resource_stamp_names_its_own_unit():
    ...

def test_git_stamp_keeps_the_first_dirty_path_whole(tmp_path):
    """``git status --porcelain`` writes "XY path", and X is a SPACE for an
    unstaged modification -- so stripping the output eats the first line's
    first character and the first path comes back as "cripts/...".
    """
    ...

def test_git_stamp_carries_the_dirty_flag():
    """The gate constants were UNCOMMITTED when E1-E4 were written, so a commit
    alone names a gate the fit did not run under."""
    ...

def test_clip_frames_spans_are_half_open():
    ...

@pytest.mark.parametrize('spec, msg', [('a', 'ID=start:end'), ('a=245', 'start:end'), ('a=x:y', 'integers'), ('a=10:10', 'start < end'), ('a=-1:5', 'start < end')])
def test_clip_frames_refuses_a_span_it_cannot_honour(spec, msg):
    ...

def test_clip_frames_refuses_a_clip_that_is_not_in_the_fit():
    ...

def test_a_span_is_the_frame_cap(tmp_path):
    """``0:245`` must fit the same 245 frames a bare ``--max-frames 245`` does,
    and a span must beat the sidecar's own ``fit_trim``."""
    ...

def _fake_fit(n_frames: int, n_joints: int=31):
    """The smallest pair of objects ``world_frame_record`` actually consumes.

    It needs a fitter with a ``root_joint``, a model it can call for a rest
    mesh, and one result dict per frame.  Everything else in the function is
    numpy over those results -- which is the part ``--clip-frames`` moves.
    """
    ...

def test_a_start_offset_keeps_every_index_in_the_windows_own_numbering():
    """C4 / R8, the claim the sidecars depend on.

    ``start_frame`` shifts what is DECODED, never how a frame is NUMBERED: the
    mask files, the centreline records and the width / bend / length series are
    all keyed on the window's own frame index, so a fit from 50 must report
    frames 50..211, count 162 of them, key the centreline gate on 50..211 and
    put ``t_s`` at ``(clip_meta.start_frame + 50 + i) / fps``.

    Drives ``world_frame_record`` itself; the arithmetic under test is the
    ``frame_offset`` that ``--clip-frames`` sets.
    """
    ...

def test_the_start_offset_is_a_time_base_not_only_a_decoder_skip():
    """C4, stated rather than assumed: ``--clip-frames ID=start:end`` is NOT
    equivalent to physically re-cutting the window.

    ``fit_request`` builds the swimming wave's ``times`` from the WINDOW index,
    so a fit from frame 50 starts at 50/fps and not at 0.  That keeps one phase
    across two windows of one video, and it means the initialiser's coarse
    (phase, frequency) grid is offset against a re-cut window's -- so the two
    fits need not settle in the same well.  Both CLIs say so.
    """
    ...

def test_no_video_turns_the_short_decode_check_off_and_says_so():
    """C5 / R7.  ``read_frames`` yields exactly ``limit`` placeholders when the
    writer is off, so the warning the caller feeds can never fire: the guard is
    OFF on precisely the flag the batch runs under, and that is now stated
    rather than reading as a passing check."""
    ...
