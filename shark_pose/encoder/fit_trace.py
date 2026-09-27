"""``--trace-json``: what the optimiser did, step by step.  DEFAULT OFF.

The fitter keeps only the CONVERGED result, so nothing on disk says how the
girth field (or the pose, or the shape) got there.  This module is the whole of
the opt-in that records it: a :class:`FitTracer` is handed to
:class:`~shark_pose.encoder.joint_fitting.JointSpineFit`, which calls
:meth:`FitTracer.armed` before a step and :meth:`FitTracer.record` after the
loss of an armed one.  Without a tracer the driver builds no hooks at all, so
the optimiser loop is the shipped loop, statement for statement.

A trace is a DIAGNOSTIC.  It is written for one fit, by hand, to be looked at;
it is not part of any record the morphometrics layer reads.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
FLUSH_EVERY = 50

def _as_list(value):
    """A torch tensor / numpy array / sequence as nested floats, or None."""
    ...

class FitTracer:
    """Collects one record every ``every`` optimiser steps and writes JSON.

    ``every`` <= 0 arms nothing, which is the same as having no tracer at all;
    :func:`make_tracer` returns ``None`` for that case so the caller never
    builds one.
    """

    def __init__(self, path, every: int=1, meta: Optional[dict]=None, flush_every: int=FLUSH_EVERY):
        ...

    def armed(self, step: int) -> bool:
        """Is ``step`` (0-based, within its stage) a step to record?"""
        ...

    def record(self, *, stage: str, step: int, pass_index: int, total_loss: float, terms: dict, girth=None, ref: Optional[dict]=None) -> dict:
        """Append one step's record; rewrite the file every ``flush_every``.

        ``terms`` are the WEIGHTED contributions ``SpineSMPLify._loss`` added,
        under the names that code uses for them; ``girth`` is the (K, 2) girth
        leaf as the optimiser holds it -- the per-station ``(g_x, g_z)``
        cross-section scales themselves, which is exactly what
        ``summary.girth.fitted`` reports; ``ref`` is the reference frame's
        pose / betas / translation, enough to re-pose the mesh later.
        """
        ...

    def payload(self) -> dict:
        ...

    def write(self) -> Path:
        """Atomic: write a sibling temp file, then rename over the target."""
        ...

def make_tracer(path, every: int, meta: Optional[dict]=None) -> Optional[FitTracer]:
    """A tracer, or ``None`` -- the default -- when tracing is off."""
    ...
