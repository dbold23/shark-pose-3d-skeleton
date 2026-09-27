"""Corpus-scale scan helpers (window trust gate)."""
from .window_gate import GATE_THRESHOLDS, TRUNK_GATE, WindowStats, gate_failures, trunk_frame_ok, trunk_bracket_fraction
__all__ = ['GATE_THRESHOLDS', 'TRUNK_GATE', 'WindowStats', 'gate_failures', 'trunk_frame_ok', 'trunk_bracket_fraction']
