"""Body surface coordinates from a stage 7 fit record, for mapping scars and marks (S8 S1).

A scar seen in one frame must be found again in another pass, another year, from the other
side. Image coordinates cannot do that; a coordinate fixed to the body can. This module builds,
per fitted frame, a tube around the fitted spine and casts image pixels onto it, returning

    u      fraction of the 13-joint spine arc, 0 at spine_00_cranium, 1 at spine_12_caudal_axis_2
    theta  degrees around the body, 0 dorsal, +90 the shark's LEFT flank, -90 its right, +-180 ventral

Everything comes from the record itself: the camera-frame spine joints (``joints_3d[:13]``,
which project to ``joints_2d_proxy`` by the proxy pinhole to 1e-5 px), the body attitude
``R_wb`` (columns: lateral, longitudinal = snout, dorsal), the per-frame girth multipliers
(``girth`` = 7 x [g_x, g_z]) and the template's section semi-axes at the 7 girth stations
(``morph_v5/template_reference.json`` -> station_comparison: a lateral, b dorsal). The tube is an
elliptical-section approximation of the fitted mesh, without fins: u is robust (it is set by the
spine), theta is approximate where fins or the head depart from an ellipse. Pixels that miss the
tube are reported as misses, never snapped.

Native pixels map to the proxy pinhole by the letterbox of ``shark_pose.sim2real.proxy_geometry``
(re-implemented here without torch; ``test_archive_body_coords`` checks the two agree).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
N_SPINE = 13
N_U = 160
N_THETA = 72

@dataclass
class Sections:
    fractions: np.ndarray
    a: np.ndarray
    b: np.ndarray

def sections_from_template_reference(tref: dict) -> Sections:
    ...

@dataclass
class Camera:
    width: int
    height: int
    proxy_res: int
    proxy_focal: float
    proxy_pp: np.ndarray

    @classmethod
    def from_summary(cls, cam: dict) -> 'Camera':
        ...

    def _letterbox(self):
        ...

    def native_to_proxy(self, xy):
        ...

    def proxy_to_native(self, xy):
        ...

    def project(self, X):
        """Camera-frame points -> native pixels."""
        ...

    def rays(self, xy_native):
        """Unit ray directions in the camera frame for native pixels."""
        ...

@dataclass
class Tube:
    V: np.ndarray
    u: np.ndarray
    theta: np.ndarray
    centre: np.ndarray
    dorsal: np.ndarray
    left: np.ndarray

def _arc_interp(P: np.ndarray, u: np.ndarray):
    ...

def build_tube(joints_3d, R_wb, girth, sections: Sections, n_u: int=N_U, n_theta: int=N_THETA) -> Tube:
    ...

def _triangles(t: Tube):
    ...

def cast(tube: Tube, rays: np.ndarray):
    """Nearest tube hit per ray (camera at the origin). Returns u, theta (deg), depth, hit mask."""
    ...

def surface_point(tube: Tube, u: float, theta_deg: float) -> np.ndarray:
    """Point on the tube at (u, theta), by the same construction as build_tube (for tests and overlays)."""
    ...

def visible_cells(tube: Tube, cam: Camera) -> np.ndarray:
    """(N_U, N_THETA) cells whose outward normal faces the camera and whose centre projects inside the frame.

    Needed to say a region was SEEN without a scar, rather than not seen at all.
    """
    ...

def tube_silhouette(tube: Tube, cam: Camera) -> np.ndarray:
    """Native-resolution 0/1 silhouette of the tube: union of the convex hulls of consecutive rings."""
    ...
