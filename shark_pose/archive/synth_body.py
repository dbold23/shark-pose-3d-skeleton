"""Synthetic known-truth white sharks, whole body, for testing the S8 re-ID front end.

An individual carries several independent identity features, as real white sharks do:

  dorsal fin     trailing-edge notches (``synth_fins.FinIdentity``)
  caudal fin     notches on the upper lobe's trailing margin
  pigment line   the jagged countershading boundary on the flank, DIFFERENT on the left and right side
  head spots     dark marks between eye and gill slits, different per side

A frame renders one side of the animal (a shark swimming to the image left, dorsal up, shows its LEFT flank)
under a view: scale, in-plane rotation, yaw foreshortening, lighting gain/offset, Gaussian defocus, sensor
noise, an optional occluder (water/glare over a region) and optional truncation by the frame edge. It returns
the image, the body mask a segmenter would return (occluded pixels excluded), and the 16 YOLO keypoints with
confidences (low where occluded or outside the frame), so every step of the pipeline sees what it would see
on a real frame and nothing more.

Canonical body coordinates: x along the body in body lengths (snout 0, caudal fork 1), y toward the dorsal side.
The keypoint stations follow ``shark_pose.corpus.view_yaw.PROPORTIONS``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from shark_pose.archive.synth_fins import FinIdentity, outline, random_identity

def top(x):
    ...

def bot(x):
    ...

@dataclass
class BodyIdentity:
    fin: FinIdentity
    caudal_notches: list[tuple[float, float, float]]
    pigment: dict[str, list[tuple[float, float, float]]]
    spots: dict[str, list[tuple[float, float, float]]]
    pigment_base: float = 0.45

def random_body(rng: np.random.Generator, n_fin_notches: int | None=None) -> BodyIdentity:
    ...

def pigment_rel(ident: BodyIdentity, side: str, x):
    """Relative height (0 ventral outline, 1 dorsal outline) of the countershading boundary at x."""
    ...

def _keypoints(ident: BodyIdentity, fin_poly_tip):
    ...

def _dorsal_fin(ident: BodyIdentity):
    ...

def _caudal(ident: BodyIdentity, n=160):
    ...

def _small_fin(base_a, base_b, tip):
    ...

def _body_outline(n=200):
    ...

@dataclass
class SynthFrame:
    img: np.ndarray
    mask: np.ndarray
    kxy: np.ndarray
    kconf: np.ndarray
    side: str

def render_frame(ident: BodyIdentity, side: str, rng: np.random.Generator, *, length_px: float=700.0, yaw_deg: float=0.0, rot_deg: float=0.0, blur_sigma: float=0.0, noise_sd: float=3.0, gain: float=1.0, offset: float=0.0, occluder: tuple[float, float, float, float] | None=None, shift=(0.0, 0.0), size=(1280, 720), kp_noise_frac: float=0.004) -> SynthFrame:
    """``occluder``: canonical box (x0, x1, y0, y1) painted over with water; ``shift``: image offset (px)."""
    ...
