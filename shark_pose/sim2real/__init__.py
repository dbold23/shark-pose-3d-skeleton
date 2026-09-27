"""Sim2Real proxy representation extraction.

The network NEVER sees raw RGB. All inputs are 19-channel proxy tensors:
  Ch  0-15: Confidence-weighted keypoint heatmaps
  Ch 16-17: Optical flow (u, v)
  Ch    18: Edge / silhouette map
"""
from shark_pose.sim2real.proxy_extractor import ProxyExtractor
from shark_pose.sim2real.keypoint_tracks import KeypointDetector, KeypointTracker, generate_heatmaps, generate_heatmaps_batch, soft_argmax_2d
from shark_pose.sim2real.optical_flow import OpticalFlowExtractor
from shark_pose.sim2real.edge_features import EdgeExtractor
from shark_pose.sim2real.underwater_correction import UnderwaterCorrector
from shark_pose.sim2real.camera_calib import CameraIntrinsics, load_camera, preset_intrinsics
from shark_pose.sim2real.mask_centerline import bend_metrics, bend_of_projected_stations, chord_bend_deg, extract_centerline, resample_polyline, tail_deflection_pct
from shark_pose.sim2real.proxy_geometry import letterbox_params, letterbox_image, image_to_proxy_xy, proxy_to_image_xy
__all__ = ['ProxyExtractor', 'KeypointDetector', 'KeypointTracker', 'generate_heatmaps', 'generate_heatmaps_batch', 'soft_argmax_2d', 'OpticalFlowExtractor', 'EdgeExtractor', 'UnderwaterCorrector', 'CameraIntrinsics', 'load_camera', 'preset_intrinsics', 'letterbox_params', 'letterbox_image', 'image_to_proxy_xy', 'proxy_to_image_xy', 'extract_centerline', 'chord_bend_deg', 'tail_deflection_pct', 'bend_metrics', 'bend_of_projected_stations', 'resample_polyline']
