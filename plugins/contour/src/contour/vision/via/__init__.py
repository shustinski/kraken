"""Via (transition hole) detection: composable strategies + legacy bridge."""

from __future__ import annotations

from .bright_tophat_dog import (
    BrightViaDetection,
    BrightViaDetectionResult,
    BrightViaDetectorConfig,
    bright_center_score,
    detect_bright_vias,
    edge_likeness_score,
    line_likeness_score,
    mask_fraction,
    radial_symmetry_score,
    suppress_close_points,
)
from .orchestrator import CompositeViaDetector, ViaRunConfig
from .primary_sem import SemPrimaryViaConfig, SemPrimaryViaDetector, ViaPolarityScan

__all__ = [
    "BrightViaDetection",
    "BrightViaDetectionResult",
    "BrightViaDetectorConfig",
    "CompositeViaDetector",
    "SemPrimaryViaConfig",
    "SemPrimaryViaDetector",
    "ViaPolarityScan",
    "ViaRunConfig",
    "bright_center_score",
    "detect_bright_vias",
    "edge_likeness_score",
    "line_likeness_score",
    "mask_fraction",
    "radial_symmetry_score",
    "suppress_close_points",
]
