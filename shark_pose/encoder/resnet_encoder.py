"""
ResNet-50 backbone encoder modified for 19-channel proxy input.

Instead of RGB images, the encoder consumes a proxy representation with
19 channels: 16 keypoint heatmaps + 2 optical flow + 1 edge/silhouette.
Every input channel of the first conv is seeded from the ImageNet-pretrained
RGB filters (see `ProxyEncoder` for the history of this choice).
"""
from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights
from ..core.constants import NUM_KEYPOINT_CHANNELS, TOTAL_PROXY_CHANNELS

def conv1_channel_balance(conv_weight: Tensor) -> Dict[str, object]:
    """Summarise how much weight each input channel of conv1 carries.

    Args:
        conv_weight: (out_channels, in_channels, kH, kW) first-conv weight.

    Returns:
        Dict with:
            per_channel_abs_mean: list of per-input-channel mean |w|
                                  (length = in_channels)
            max_abs_mean / min_abs_mean: over the keypoint channels only
                                  (0 .. NUM_KEYPOINT_CHANNELS - 1)
            argmax / argmin:      the channel indices those came from
            ratio:                max_abs_mean / min_abs_mean over the same
                                  keypoint channels (inf when a channel is
                                  exactly zero)

    The ratio is the audit statistic: a healthy encoder keeps it near 1,
    while the old zero-init produced 19x (round_004) to 608x (stage 1).
    """
    ...

class ProxyEncoder(nn.Module):
    """ResNet-50 encoder adapted for 19-channel proxy input.

    Architecture:
        - Conv1 modified: Conv2d(19, 64, 7, stride=2, padding=3)
        - Layers 1-4 from standard ResNet-50
        - Global average pooling -> 2048-D feature vector

    conv1 initialisation (changed 2026-09-03)
    -----------------------------------------
    The original implementation zeroed the whole conv1 weight and then copied
    the ImageNet RGB filters into input channels 0-2 only -- in BOTH the
    pretrained and the randomly-initialised case.  Channels 3-18 therefore
    started at exactly zero and could only grow through gradients that were
    themselves proportional to the (zero) weights of neighbouring layers.
    They never caught up.  Measured on the trained checkpoints:

        checkpoint                 mean |w| ratio, channels 0-2 : 3-15
        stage 1                    608x
        stage 2                    217x
        active-learning round_004   19x

    with channels 16-18 still exactly 0.  The encoder was consequently almost
    blind to the keypoint heatmaps it is supposed to read: shifting every
    heatmap by +40 px moved the predicted 2D joints by only ~13 px (gain 0.34),
    which is the mean-pose collapse seen on real data.

    Current behaviour:
        pretrained=True   every input channel receives the MEAN of the three
                          ImageNet RGB filters, scaled by 3 / in_channels.
                          A spatially uniform input then produces exactly the
                          same pre-activation as the original RGB conv would,
                          so bn1 and the rest of the pretrained trunk stay in
                          their calibrated regime, and every channel starts
                          with identical (non-zero) weight.
        pretrained=False  PyTorch's default Kaiming init on the fresh conv is
                          kept untouched -- notably it is NOT zeroed.

    Channels 16-18 (flow u, flow v, edge) receive the same initialisation as
    the heatmap channels.  By project convention they are zero at input in
    both training and inference (`ProxyExtractor(keypoints_only=True)`), so
    their weights are harmless; giving them a normal init means the layer can
    be reused unchanged if those channels are ever filled.
    """

    def __init__(self, pretrained: bool=True, in_channels: int=TOTAL_PROXY_CHANNELS):
        """
        Args:
            pretrained: If True, seed every conv1 input channel from the
                ImageNet RGB filters (mean filter, scaled by 3/in_channels)
                and load ImageNet weights for the rest of the trunk.
            in_channels: Number of input channels (default 19).
        """
        ...

    def channel_balance(self) -> Dict[str, object]:
        """conv1_channel_balance of this encoder's current conv1 weight."""
        ...

    def forward(self, proxy_input: Tensor) -> Tensor:
        """Extract features from proxy representation.

        Args:
            proxy_input: (B, 19, H, W) proxy representation tensor.
                Channels 0-15: keypoint heatmaps (one per joint)
                Channels 16-17: optical flow (u, v)
                Channel 18: edge / silhouette map

        Returns:
            features: (B, 2048) global feature vector.
        """
        ...

    @property
    def output_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        ...
