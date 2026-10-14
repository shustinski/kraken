"""benchmark_metal_segmentation: module state."""
from __future__ import annotations

STRATEGIES = (
    "auto",
    "legacy_otsu",
    "local_adaptive",
    "gradient_watershed",
    "random_walker",
    "graph_cut",
    "reconstruction",
    "closed_boundary",
    "structural_watershed",
    "owt_ucm",
    "graph_multi_separator",
    "gasp",
    "mutex_watershed",
    "multicut",
    "lifted_multicut",
)

# Evaluation-only ROI. Recognition always runs on the full SEM frame.
EVALUATION_BORDER_CROP_PX = 50

BORDER_AUDIT_FRAMES = ("1514", "2497", "0284")
