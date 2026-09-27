"""
Underwater shader setup for Blender.

Creates a physically-motivated underwater environment using Blender's
shader node system. Supports volumetric scattering, wavelength-dependent
color absorption, projected caustic patterns, and backscatter particle
effects. All parameters are exposed so that ``domain_randomization.py``
can drive them.

Requires Blender's ``bpy`` module (runs inside Blender Python).
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

def _ensure_bpy() -> None:
    """Raise if not running inside Blender."""
    ...

def setup_world_volume(absorption_coefficients: Tuple[float, float, float]=(0.4, 0.1, 0.05), turbidity: float=0.3, algae_tint: Tuple[float, float, float]=(0.0, 0.1, 0.0), algae_density: float=0.1, fog_start: float=5.0, fog_end: float=50.0) -> None:
    """Configure the Blender world shader for underwater volumetric medium.

    Creates a Principled Volume node tree in the world shader that
    simulates depth-dependent light absorption and scattering.

    Args:
        absorption_coefficients: (r, g, b) per-channel absorption. Higher
            values mean faster attenuation. Red is absorbed most in water.
        turbidity: overall scattering density [0, 1].
        algae_tint: additive color tint from algae/phytoplankton.
        algae_density: amount of algae tinting [0, 1].
        fog_start: near distance where volumetric fog begins (meters).
        fog_end: far distance where fog reaches full opacity.
    """
    ...

def setup_caustics(caustic_scale: float=1.5, caustic_intensity: float=0.5, sun_elevation: float=45.0, sun_azimuth: float=180.0) -> None:
    """Create a caustic light projector in the scene.

    Uses a spotlight with a procedural Voronoi texture to approximate
    the dancing light patterns caused by water surface refraction.

    Args:
        caustic_scale: scale of the caustic pattern texture.
        caustic_intensity: brightness of the caustic light [0, 1].
        sun_elevation: sun angle above horizon (degrees).
        sun_azimuth: sun compass direction (degrees).
    """
    ...

def setup_backscatter_particles(backscatter_density: float=0.2, turbidity: float=0.3) -> None:
    """Create floating particle system to simulate backscatter.

    Adds small icosphere particles in the camera frustum to represent
    suspended sediment, plankton, and marine snow.

    Args:
        backscatter_density: controls particle count [0, 1].
        turbidity: overall water turbidity, influences particle size.
    """
    ...

def setup_shark_skin_material(shark_obj: Any, skin_params: Dict[str, Any]) -> None:
    """Apply domain-randomized skin material to the shark mesh.

    Args:
        shark_obj: Blender mesh object for the shark.
        skin_params: skin randomization dict from domain_randomization.
    """
    ...

def setup_underwater_environment(params: Optional[Dict[str, Dict[str, Any]]]=None) -> None:
    """Set up the complete underwater environment in Blender.

    This is the main entry point called by the render pipeline. It
    configures the world volume, caustic projector, and backscatter
    particles using the provided domain randomization parameters.

    Args:
        params: nested dict from ``domain_randomization.randomize_all()``.
            Expected keys: 'water', 'lighting'. If None, uses defaults.
    """
    ...
