"""``--trace-json``: the trace writer's schema, and that OFF is a no-op.

Two things are worth pinning here.  The first is the RECORD: a trace is read by
an offline viewer, so its shape is an interface.  The second is that the flag
OFF changes nothing -- the helper that builds the optimiser's hooks returns
``(None, None)`` without a tracer, so ``JointSpineFit._run`` runs the statement
sequence it always ran, and with the hooks ON the loop's arithmetic is still
untouched (the hook is called between the loss and the backward and only reads).
"""
from __future__ import annotations
import json
from types import SimpleNamespace
import pytest
from shark_pose.encoder.fit_trace import FitTracer, make_tracer
from shark_pose.encoder.joint_fitting import JointSpineFit

def test_trace_writer_schema(tmp_path):
    ...

def test_trace_writer_flushes_partial(tmp_path):
    """A crash mid-fit still leaves the records written so far."""
    ...

def test_tracer_accepts_a_torch_girth_leaf(tmp_path):
    ...

def test_make_tracer_is_none_when_off(tmp_path):
    ...

def test_trace_hooks_are_none_without_a_tracer():
    """No tracer -> no hooks, so ``_run`` is the shipped loop."""
    ...

def test_trace_hooks_arm_only_the_recorded_steps(tmp_path):
    """With a tracer the hooks arm the fitter's term dict on those steps only."""
    ...

def test_run_is_unchanged_by_the_hooks(tmp_path):
    """The traced loop takes the same steps as the untraced one, bit for bit."""
    ...

def test_the_fitter_ships_with_tracing_off():
    """Both trace attributes are class-level and default to None."""
    ...
