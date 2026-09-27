"""SPIN encoder, regressor, and optimization."""
from .resnet_encoder import ProxyEncoder
from .iterative_regressor import IterativeRegressor
from .spin_model import SPINModel
from .smplify_fitting import SharkSMPLify, NeuralDescentRNN, geman_mcclure
from .bootstrap import BootstrapManager, GateThresholds, PseudoGTSample
__all__ = ['ProxyEncoder', 'IterativeRegressor', 'SPINModel', 'SharkSMPLify', 'NeuralDescentRNN', 'geman_mcclure', 'BootstrapManager', 'GateThresholds', 'PseudoGTSample']
