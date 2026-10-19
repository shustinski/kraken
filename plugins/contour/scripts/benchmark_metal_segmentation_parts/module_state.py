"""benchmark_metal_segmentation: module state."""
from __future__ import annotations
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PLUGIN_ROOT / "src"
REAL_DATASET_ROOT = PLUGIN_ROOT / "tests" / "test_metal"

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
