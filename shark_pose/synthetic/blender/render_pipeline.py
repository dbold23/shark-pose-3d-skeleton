"""
Render orchestration for synthetic shark imagery using Blender.

Provides a ``RenderConfig`` dataclass for all render settings and a
``render_single_frame`` function that produces RGB, depth, normals,
segmentation mask, projected 2D keypoints, 3D joints, theta, and beta
annotations for each frame.

Designed to run inside Blender's Python interpreter. The ``bpy`` import
is conditional so that the module can be imported outside Blender for
testing or configuration purposes.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .domain_randomization import randomize_all
from .pose_sampler import sample_pose

@dataclass
class RenderConfig:
    """Configuration for a single render or batch of renders.

    Attributes:
        output_dir: root directory for all render outputs.
        resolution: (width, height) in pixels.
        samples: number of path-tracing samples per pixel.
        engine: Blender render engine ('CYCLES' or 'EEVEE').
        use_gpu: whether to use GPU compute for Cycles.
        device_type: GPU backend ('CUDA', 'OPTIX', 'METAL', 'OPENCL').
        film_transparent: render with transparent background.
        color_depth: bit depth for output images (8 or 16).
        file_format: image format ('PNG', 'OPENEXR', 'JPEG').
        export_depth: write depth pass.
        export_normals: write surface normals pass.
        export_segmentation: write object-index segmentation mask.
        export_keypoints_2d: project 3D joints to 2D and save.
        export_keypoints_3d: save 3D joint locations.
        shark_blend_file: path to the .blend file with the rigged shark model.
        hdri_dir: directory of HDRI environment maps for backgrounds.
        seed: global random seed for reproducibility.
    """
    resolution: Tuple[int, int] = (512, 512)
    samples: int = 128
    use_gpu: bool = True
    film_transparent: bool = False
    color_depth: int = 16
    export_depth: bool = True
    export_normals: bool = True
    export_segmentation: bool = True
    export_keypoints_2d: bool = True
    export_keypoints_3d: bool = True
    seed: int = 42

def _build_annotation(render_id: str, pose_params: Dict[str, Any], randomization_params: Dict[str, Dict[str, Any]], keypoints_2d: Optional[np.ndarray], keypoints_3d: Optional[np.ndarray], beta: Optional[np.ndarray], camera_matrix: Optional[np.ndarray], image_paths: Dict[str, str]) -> Dict[str, Any]:
    """Build a JSON-serialisable annotation dict for a single render.

    Args:
        render_id: unique identifier for this render.
        pose_params: output from pose_sampler.sample_pose().
        randomization_params: output from domain_randomization.randomize_all().
        keypoints_2d: (16, 2) projected pixel coordinates or None.
        keypoints_3d: (16, 3) world-space joint locations or None.
        beta: (10,) shape parameters or None.
        camera_matrix: (4, 4) camera world matrix or None.
        image_paths: mapping of pass name to file path.

    Returns:
        Annotation dictionary.
    """
    ...

def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    ...

def _configure_render_settings(config: RenderConfig) -> None:
    """Apply render settings to the current Blender scene."""
    ...

def _setup_render_passes(config: RenderConfig) -> None:
    """Enable compositor render passes for depth, normals, segmentation."""
    ...

def _apply_pose_to_armature(armature_obj: Any, pose_params: Dict[str, Any]) -> None:
    """Apply sampled pose angles to the shark armature.

    Maps the 16-joint axis-angle vector to Blender bone rotations.

    Args:
        armature_obj: Blender armature object.
        pose_params: output from pose_sampler.sample_pose().
    """
    ...

def _get_keypoints_3d(armature_obj: Any) -> np.ndarray:
    """Extract 3D joint positions from the posed armature.

    Args:
        armature_obj: Blender armature object.

    Returns:
        (16, 3) array of joint world-space positions.
    """
    ...

def _project_to_2d(keypoints_3d: np.ndarray, camera_obj: Any, scene: Any) -> np.ndarray:
    """Project 3D keypoints to 2D pixel coordinates using the scene camera.

    Args:
        keypoints_3d: (16, 3) world-space joint positions.
        camera_obj: Blender camera object.
        scene: Blender scene.

    Returns:
        (16, 2) pixel coordinates.
    """
    ...

def render_single_frame(config: RenderConfig, pose_params: Optional[Dict[str, Any]]=None, randomization_params: Optional[Dict[str, Dict[str, Any]]]=None, render_id: Optional[str]=None, beta: Optional[np.ndarray]=None) -> Dict[str, Any]:
    """Render a single synthetic frame with full annotations.

    Orchestrates pose application, domain randomization, rendering, and
    annotation export. When called outside Blender (HAS_BPY is False),
    produces a dry-run annotation without actual rendering.

    Args:
        config: render configuration.
        pose_params: output from sample_pose(). Sampled if None.
        randomization_params: output from randomize_all(). Sampled if None.
        render_id: unique ID. Auto-generated if None.
        beta: (10,) shape parameters. Random if None.

    Returns:
        Annotation dictionary for this render.
    """
    ...
