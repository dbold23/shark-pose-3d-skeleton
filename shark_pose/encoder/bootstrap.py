"""
Bootstrap manager for SPIN self-training loop.

Runs inference on unlabeled real-world images, refines predictions with
SharkSMPLify, then filters through a 5-gate rejection system to produce
high-quality pseudo ground-truth annotations. These are accumulated into
a growing pseudo-GT pool and merged with synthetic training data for
the next training round (data evolution).

The 5-gate rejection system ensures only reliable pseudo-labels enter
the training set:
    1. Reprojection gate: mean 2D error below pixel threshold
    2. Discriminator gate: adversarial plausibility score above threshold
    3. Pose prior gate: GMM log-probability above threshold
    4. Keypoint visibility gate: at least 10/16 joints visible
    5. Uncertainty gate: mean kappa concentration above threshold
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset, TensorDataset
from ..core.constants import NUM_JOINTS

@dataclass
class GateThresholds:
    """Thresholds for the 5-gate rejection system."""
    reproj_max_px: float = 15.0
    discriminator_min: float = 0.5
    pose_prior_min: float = -60.0
    min_visible_joints: int = 6
    kappa_min: float = 5.0

@dataclass
class PseudoGTSample:
    """A single pseudo ground-truth annotation produced by the bootstrap loop."""
    proxy_input: Tensor
    pose: Tensor
    shape: Tensor
    camera: Tensor
    keypoints_2d: Tensor
    keypoint_conf: Tensor
    kappa: Tensor
    reproj_error: float

class BootstrapManager:
    """Manages the SPIN self-training bootstrap loop.

    Workflow per epoch:
        1. Run SPINModel inference on unlabeled batch.
        2. Refine with SharkSMPLify fitting.
        3. Apply 5-gate rejection to filter high-quality pseudo-GT.
        4. Accumulate accepted samples into the pseudo-GT pool.
        5. Merge pseudo-GT pool with synthetic dataset for next training round.
    """

    def __init__(self, thresholds: Optional[GateThresholds]=None, discriminator: Optional[nn.Module]=None, pose_prior: Optional[Callable[[Tensor], Tensor]]=None, max_pool_size: int=50000):
        """
        Args:
            thresholds: Gate thresholds. If None, uses defaults.
            discriminator: Optional adversarial discriminator module.
                Should accept (B, 61) parameter vectors and return
                (B,) plausibility scores in [0, 1].
            pose_prior: Optional GMM pose prior function.
                Should accept (B, 48) pose vectors and return
                (B,) log-probabilities.
            max_pool_size: Maximum number of pseudo-GT samples to retain.
        """
        ...

    def process_batch(self, proxy_inputs: Tensor, model_outputs: Dict[str, Tensor], smplify_outputs: Dict[str, Tensor], keypoints_2d: Tensor, keypoint_conf: Tensor) -> List[PseudoGTSample]:
        """Process a batch of predictions through the 5-gate rejection system.

        Args:
            proxy_inputs: (B, 19, H, W) proxy representations.
            model_outputs: Dict from SPINModel.forward() containing at least
                'rotation_uncertainty' (B, 16) kappa values.
            smplify_outputs: Dict from SharkSMPLify.fit() containing
                'pose' (B, 48), 'shape' (B, 10), 'camera' (B, 3),
                'reprojection_error' (B,).
            keypoints_2d: (B, 16, 2) detected 2D keypoints.
            keypoint_conf: (B, 16) keypoint confidence scores.

        Returns:
            List of PseudoGTSample instances that passed all 5 gates.
        """
        ...

    def _add_to_pool(self, new_samples: List[PseudoGTSample]) -> None:
        """Add samples to the pseudo-GT pool, evicting oldest if full.

        Maintains a FIFO eviction policy: oldest samples are removed
        first when the pool exceeds max_pool_size.
        """
        ...

    def get_pseudo_gt_dataset(self) -> Dataset:
        """Convert the current pseudo-GT pool to a PyTorch Dataset.

        Returns:
            TensorDataset with (proxy_input, pose, shape, camera) tuples.
        """
        ...

    def get_evolved_dataset(self, synthetic_dataset: Dataset) -> Dataset:
        """Merge pseudo-GT pool with synthetic dataset for training.

        This implements the data evolution strategy: each bootstrap round
        grows the training set by adding verified pseudo-GT from real images
        to the existing synthetic data.

        Args:
            synthetic_dataset: The base synthetic training dataset.

        Returns:
            ConcatDataset combining synthetic and pseudo-GT data.
        """
        ...

    def get_pool_stats(self) -> Dict[str, float]:
        """Return summary statistics of the current pseudo-GT pool.

        Returns:
            Dictionary with pool size, mean reprojection error,
            mean kappa, etc.
        """
        ...

    def clear_pool(self) -> None:
        """Clear the pseudo-GT pool (e.g., at the start of a new training phase)."""
        ...

    @property
    def pool_size(self) -> int:
        """Current number of samples in the pseudo-GT pool."""
        ...
