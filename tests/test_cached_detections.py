"""Tests for the cached-detections flag (``detector.cached_detections``).

Contract under test:
  * the flag is ABSENT from every shipped config and defaults to false, and
    with it off nothing changes -- the live YOLO backend is still constructed,
    even when a ``detections.npz`` path is handed in;
  * with it on, one window frame's evidence is EXACTLY the cached row (proxy
    px and confidence, not re-gated), a frame past the end of the cache is the
    all-zero row (= undetected), and the native-px row is the exact letterbox
    inverse of the proxy row;
  * with it on NO detector backend is constructed, so the fit runs on a box
    with no detector weights;
  * ``discover_clip`` finds ``detections.npz`` and ``fit_individual`` hands the
    path to the fitter only when the flag is on.

Nothing here decodes a video or runs a model: the per-frame logic is the
importable :func:`cached_detection_row`, and the two generator probes stop at
the detector-construction branch (the next statement loads assets off disk and
raises, which is what the probes assert they reach).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml
from shark_pose.core.skeleton import NUM_JOINTS
from shark_pose.sim2real.cached_detections import DETECTIONS_FILENAME, cached_detection_row, load_cached_detections
from shark_pose.sim2real.proxy_geometry import image_to_proxy_xy, proxy_to_image_xy
H, W, RES = (1080, 1920, 256)

def make_npz(tmp_path: Path, n_frames: int=5, name: str=DETECTIONS_FILENAME):
    """A detections cache with a distinct, exactly representable value per cell."""
    ...

def run_gen_to_detector(tmp_path: Path, cfg: dict, detections_path):
    """Advance ``_fit_video_gen`` up to (and through) its detector branch.

    Returns ``"BACKEND"`` if it tried to construct the YOLO backend, or the
    exception it raised at the NEXT step (loading the model assets, which do
    not exist here) -- i.e. proof that it got past the branch without one.
    """
    ...

def base_cfg(**detector):
    ...

@pytest.mark.parametrize('cfg_path', SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_do_not_carry_the_key(cfg_path):
    """The key is absent from the shipped configs and reads false there."""
    ...

def test_off_still_constructs_the_live_backend(tmp_path):
    """OFF is the shipped path: the detector is built, cache or no cache."""
    ...

def test_off_ignores_a_detections_path(tmp_path):
    """A stray --detections cannot switch the cached path on; only the flag can."""
    ...

def test_cli_default_is_no_detections(monkeypatch):
    """An existing fit_video.py command line is unchanged: --detections is None."""
    ...

def test_row_is_exactly_the_cached_row(tmp_path):
    ...

def test_row_is_not_re_gated_per_keypoint(tmp_path):
    """The live conf gate is at BOX level, so cached rows are already post-gate.

    A cached confidence below ``detector.yolo_conf`` (0.25) must survive
    verbatim -- zeroing it here would throw away keypoints the live path keeps.
    """
    ...

def test_past_the_end_is_the_zero_row(tmp_path):
    ...

def test_native_row_round_trips_the_proxy_row(tmp_path):
    """``proxy_to_image_xy`` inverts ``image_to_proxy_xy`` on the cached rows."""
    ...

def test_on_needs_no_yolo_backend(tmp_path, monkeypatch):
    """The per-frame path works with the backend poisoned: no weights needed."""
    ...

def test_on_skips_the_backend_in_the_fitter(tmp_path):
    """And the fitter itself: ON reaches asset loading without building one."""
    ...

def test_on_without_a_path_refuses(tmp_path):
    ...

@pytest.mark.parametrize('kw,msg', [(dict(kp=np.zeros((2, 5, 2), np.float32), conf=np.zeros((2, 5), np.float32)), 'kp must be'), (dict(kp=np.zeros((2, NUM_JOINTS, 2), np.float32), conf=np.zeros((3, NUM_JOINTS), np.float32)), 'conf must be'), (dict(kp=np.zeros((2, NUM_JOINTS, 2), np.float32)), 'missing conf')])
def test_malformed_cache_is_refused(tmp_path, kw, msg):
    ...

def test_missing_cache_is_refused(tmp_path):
    ...

def test_discover_clip_finds_the_cache(tmp_path):
    ...

def test_discover_clip_without_a_cache(tmp_path):
    ...
