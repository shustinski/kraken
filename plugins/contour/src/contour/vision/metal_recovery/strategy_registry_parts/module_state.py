"""strategy_registry: module state."""
from __future__ import annotations
from typing import Literal

ParameterKind = Literal["bool", "int", "float", "choice"]

ParameterValue = bool | int | float | str

_EXISTING = (
    ("auto", "Auto (topology control)"),
    ("legacy_otsu", "Legacy Otsu"),
    ("local_adaptive", "Local Adaptive"),
    ("gradient_watershed", "Gradient Watershed"),
    ("random_walker", "Random Walker"),
    ("graph_cut", "Graph Cut"),
    ("reconstruction", "Reconstruction"),
    ("closed_boundary", "Closed Boundary"),
    ("structural_watershed", "Structural Watershed"),
)

IMPLEMENTED_NEW_STRATEGIES = frozenset(
    {"owt_ucm", "graph_multi_separator", "gasp", "mutex_watershed", "multicut", "lifted_multicut"}
)
