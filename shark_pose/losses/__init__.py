"""Loss functions for all training stages."""
from shark_pose.losses.keypoint_loss import Keypoint3DLoss, KeypointReprojectionLoss, StructureAwareLoss
from shark_pose.losses.mesh_loss import CollisionLoss, MeshEdgeLoss, MeshVertexLoss
from shark_pose.losses.joint_limits import JointLimitLoss, default_joint_limits_rad
from shark_pose.losses.pose_prior_loss import PosePriorLoss
from shark_pose.losses.shape_prior_loss import ShapePriorLoss
from shark_pose.losses.adversarial_loss import AdversarialLoss, gradient_penalty
from shark_pose.losses.temporal_loss import AccelerationLoss, DCTSmoothnessLoss, RotationalSmoothnessLoss, SmoothnessLoss, TemporalFrictionLoss, make_temporal_loss_fn
from shark_pose.losses.saa_loss import SemanticAlignmentLoss, gaussian_kernel
__all__ = ['KeypointReprojectionLoss', 'Keypoint3DLoss', 'StructureAwareLoss', 'MeshVertexLoss', 'MeshEdgeLoss', 'CollisionLoss', 'JointLimitLoss', 'default_joint_limits_rad', 'PosePriorLoss', 'ShapePriorLoss', 'AdversarialLoss', 'gradient_penalty', 'AccelerationLoss', 'SmoothnessLoss', 'DCTSmoothnessLoss', 'RotationalSmoothnessLoss', 'make_temporal_loss_fn', 'TemporalFrictionLoss', 'SemanticAlignmentLoss', 'gaussian_kernel']
