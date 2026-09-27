"""
Domain randomization parameter generators for synthetic shark rendering.

Each function produces a dictionary of Blender-compatible parameters
that can be passed directly to Blender material/lighting/world nodes.
The ``randomize_all()`` function calls every sub-randomizer and returns
a single merged configuration dictionary.

Categories:
    skin_pattern  - shark skin base color, pattern, counter-shading
    lighting      - sun angle, intensity, caustics, ambient temperature
    water         - turbidity, backscatter, absorption, algae tint
    background    - seafloor texture, distractors, occluders
    camera        - focal length, motion blur, distortion, sensor noise
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np

def randomize_skin_pattern(rng: Optional[np.random.Generator]=None) -> Dict[str, Any]:
    """Generate randomized shark skin material parameters.

    Args:
        rng: numpy random generator. Created fresh if None.

    Returns:
        Dict with keys:
            base_color_dorsal: (r, g, b) linear RGB for dorsal surface.
            base_color_ventral: (r, g, b) linear RGB for ventral surface.
            pattern_type: one of 'spots', 'stripes', 'plain', 'gradient'.
            pattern_scale: float, UV-space scale of the pattern texture.
            pattern_contrast: float in [0, 1], contrast of the pattern overlay.
            counter_shading_blend: float in [0, 1], dorsal-to-ventral gradient blend.
            roughness: float in [0.3, 0.9], surface micro-roughness.
            specular: float in [0.0, 0.5], specular reflection intensity.
    """
    ...

def randomize_lighting(rng: Optional[np.random.Generator]=None) -> Dict[str, Any]:
    """Generate randomized lighting parameters.

    Models both the sun (directional) and underwater caustic patterns
    projected through the water surface.

    Args:
        rng: numpy random generator.

    Returns:
        Dict with keys:
            sun_elevation: float, degrees above horizon [5, 85].
            sun_azimuth: float, degrees [0, 360].
            sun_intensity: float, energy value [0.5, 5.0].
            sun_color_temperature: float, Kelvin [4500, 7500].
            caustic_scale: float, caustic pattern scale [0.5, 3.0].
            caustic_intensity: float, [0.0, 1.0].
            ambient_color_temperature: float, Kelvin [3500, 8000].
            ambient_intensity: float, [0.1, 1.0].
            num_point_lights: int, additional point lights [0, 4].
            point_light_intensity_range: (min, max) energy for point lights.
            point_light_color_temp_range: (min, max) Kelvin for point lights.
    """
    ...

def jerlov_kd(water_type: str, blend: float=0.0, rng: Optional[np.random.Generator]=None) -> np.ndarray:
    """Return the RGB diffuse attenuation coefficients Kd for a Jerlov type.

    Args:
        water_type: one of ``JERLOV_WATER_TYPES`` keys.
        blend: in [0, 1], fraction to interpolate toward the *next* murkier
            type in ``JERLOV_TYPE_ORDER`` (0 = pure ``water_type``). Gives
            continuous coverage between the discrete anchors.
        rng: unused; accepted for signature symmetry.

    Returns:
        (3,) float array of Kd_RGB in m^-1.
    """
    ...

def randomize_water(rng: Optional[np.random.Generator]=None, distance_range: Tuple[float, float]=(0.5, 8.0), water_type: Optional[str]=None) -> Dict[str, Any]:
    """Generate randomized underwater medium parameters (Jerlov-based).

    Controls how light is scattered and absorbed by the water column,
    directly affecting visibility and color cast. Rather than sampling
    turbidity/absorption independently, this samples a physically-grounded
    Jerlov water type and derives every downstream quantity from its
    Kd(lambda) spectrum, so color cast and visibility co-vary the way they
    do in real footage.

    Args:
        rng: numpy random generator.
        distance_range: (min, max) camera->subject path length in metres.
            The per-channel transmission ``exp(-Kd * distance)`` sets the
            water color cast, so this couples subject range to appearance.
        water_type: force a specific Jerlov type (one of
            ``JERLOV_WATER_TYPES``). If None, one is sampled at random and
            blended toward the next murkier type.

    Returns:
        Dict with keys (backward-compatible superset of the previous
        implementation; ``underwater_shader.py`` consumes ``turbidity``,
        ``backscatter_density`` and ``absorption_coefficients``):
            turbidity: float in [0, 1], normalised from Kd_G.
            backscatter_density: float in [0, 1], scales with turbidity.
            absorption_profile: str, ``"jerlov_<type>"``.
            absorption_coefficients: (r, g, b) attenuation [0, 1] (Kd clipped).
            algae_tint: (r, g, b) additive green-yellow algae tint.
            algae_density: float in [0, 1], higher in coastal water.
            fog_start / fog_end: float metres, tied to physical visibility.
            jerlov_type: str, the sampled base type.
            jerlov_blend: float in [0, 1], interpolation toward next type.
            kd_rgb: (r, g, b) diffuse attenuation coefficients [m^-1].
            path_length_m: float, sampled camera->subject distance.
            transmission_rgb: (r, g, b) fraction of light surviving the path.
            water_color_rgb: (r, g, b) resulting medium color cast [0, 1].
    """
    ...

def randomize_background(rng: Optional[np.random.Generator]=None) -> Dict[str, Any]:
    """Generate randomized background and scene clutter parameters.

    Args:
        rng: numpy random generator.

    Returns:
        Dict with keys:
            floor_texture_type: str, seafloor material type.
            floor_texture_scale: float, UV tiling scale [0.5, 5.0].
            floor_roughness: float in [0.3, 1.0].
            floor_depth: float, depth below shark (meters) [0.5, 30.0].
            distractor_count: int, number of non-shark objects (fish, etc.) [0, 15].
            distractor_size_range: (min, max) scale factor for distractors.
            occluder_count: int, partial occluders (kelp, reef, etc.) [0, 5].
            occluder_opacity_range: (min, max) transparency [0.3, 1.0].
            background_hdri_rotation: float, degrees [0, 360].
    """
    ...

def randomize_camera(rng: Optional[np.random.Generator]=None) -> Dict[str, Any]:
    """Generate randomized camera parameters.

    Args:
        rng: numpy random generator.

    Returns:
        Dict with keys:
            focal_length: float, mm [18, 135].
            sensor_width: float, mm [23.5, 36.0].
            motion_blur_shutter: float, shutter angle in degrees [0, 180].
            motion_blur_enabled: bool.
            lens_distortion_k1: float, radial distortion [-0.3, 0.3].
            lens_distortion_k2: float, radial distortion [-0.1, 0.1].
            chromatic_aberration: float in [0, 0.02].
            noise_level: float in [0, 0.1], additive Gaussian noise sigma.
            exposure_compensation: float in [-2.0, 2.0] EV.
            white_balance_shift: float in [-0.3, 0.3].
    """
    ...

def randomize_all(seed: Optional[int]=None, rng: Optional[np.random.Generator]=None) -> Dict[str, Dict[str, Any]]:
    """Generate a complete set of domain randomization parameters.

    Calls all sub-randomizers and returns a nested dictionary suitable
    for passing into the render pipeline.

    Args:
        seed: random seed. Ignored if rng is provided.
        rng: numpy random generator. If None, created from seed.

    Returns:
        Dict with top-level keys: 'skin', 'lighting', 'water',
        'background', 'camera', each mapping to the corresponding
        sub-randomizer's output dict.
    """
    ...
