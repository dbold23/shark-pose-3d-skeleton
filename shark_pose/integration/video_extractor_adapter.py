"""
Adapter wrapping SharkScarAnnotator's VideoFrameExtractor for the temporal pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

class VideoExtractorAdapter:
    """Wraps SharkScarAnnotator's VideoFrameExtractor for pose estimation pipeline.

    Provides quality-based frame extraction with multiple strategies.
    """

    def __init__(self, annotator_root: str | Path):
        """
        Args:
            annotator_root: path to the shark-scar-annotator/ repo directory
        """
        ...

    @property
    def available(self) -> bool:
        ...

    def extract_frames(self, video_path: str | Path, output_dir: str | Path, strategy: str='quality', max_frames: int | None=None, min_quality_score: float=30.0, n: int=30) -> list[str]:
        """Extract frames from video using quality-based filtering.

        Args:
            video_path: path to input video
            output_dir: directory for extracted frames
            strategy: "every_n", "uniform", "quality", or "keyframe"
            max_frames: maximum frames to extract
            min_quality_score: minimum quality threshold (0-100)
            n: frame interval for "every_n" strategy

        Returns:
            List of extracted frame file paths.
        """
        ...

    def extract_for_temporal(self, video_path: str | Path, output_dir: str | Path, fps: int=10) -> list[str]:
        """Extract frames at a fixed FPS for temporal processing.

        For temporal models (BiLSTM, Skeletor), we need evenly-spaced frames.

        Args:
            video_path: input video
            output_dir: output directory
            fps: desired frames per second

        Returns:
            List of extracted frame paths.
        """
        ...
