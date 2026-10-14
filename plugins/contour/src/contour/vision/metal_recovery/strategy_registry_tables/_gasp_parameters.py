"""_GASP_PARAMETERS."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal
from ..strategy_registry_parts.module_state import (
    ParameterKind,
    ParameterValue,
    _EXISTING,
    IMPLEMENTED_NEW_STRATEGIES,
)
from ..strategy_registry_parts.parameter_spec import (
    ParameterSpec,
    StrategySpec,
    _p,
    MetalStrategyConfigs,
    normalize_strategy_parameters,
    strategy_spec,
    visible_strategy_specs,
)
from ._classification_parameters import _CLASSIFICATION_PARAMETERS
from ._signed_graph_parameters import _SIGNED_GRAPH_PARAMETERS

_GASP_PARAMETERS = (
    *_SIGNED_GRAPH_PARAMETERS,
    _p(
        "linkage_criterion",
        "GASP linkage criterion",
        "choice",
        "average",
        choices=(
            ("average", "Average linkage"),
            ("mutex_abs_max", "Mutex/absolute-max linkage"),
            ("sum", "Sum linkage"),
        ),
        tooltip="Average is size-stable, sum favors broad evidence, and mutex/absolute-max preserves the strongest signed conflict.",
    ),
    _p(
        "use_signed_edges",
        "Use signed edges",
        "bool",
        True,
        tooltip="When enabled, repulsive edges veto or penalize merges. Disabling them reduces GASP to attractive agglomeration.",
    ),
    _p(
        "minimum_merge_affinity",
        "Minimum merge affinity",
        "float",
        0.05,
        -1.0,
        1.0,
        0.01,
        tooltip="Clusters merge only above this signed affinity. Higher values increase region count.",
    ),
    _p(
        "maximum_repulsive_conflict",
        "Maximum repulsive conflict",
        "float",
        0.45,
        0.0,
        1.0,
        0.01,
        tooltip="Merges with stronger accumulated repulsion are rejected. Lower values preserve more boundaries.",
    ),
    _p(
        "merge_stopping_threshold",
        "Merge stopping threshold",
        "float",
        0.0,
        -1.0,
        1.0,
        0.01,
        tooltip="Stops once the best remaining linkage is at or below this value.",
    ),
    _p(
        "maximum_operations",
        "Maximum operations",
        "int",
        200000,
        1,
        5000000,
        1000,
        tooltip="Safety cap for agglomeration operations. Too low a value leaves an incomplete partition.",
        advanced=True,
    ),
    *_CLASSIFICATION_PARAMETERS,
)
