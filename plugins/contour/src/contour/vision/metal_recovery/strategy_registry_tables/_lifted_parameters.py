"""_LIFTED_PARAMETERS."""
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
from ._multicut_parameters import _MULTICUT_PARAMETERS

_LIFTED_PARAMETERS = (
    *_MULTICUT_PARAMETERS,
    _p(
        "lifted_edges_enabled",
        "Lifted edges enabled",
        "bool",
        True,
        tooltip="Adds sparse non-local relations while preserving the same local graph and cost model as Multicut.",
    ),
    _p(
        "minimum_lifted_distance",
        "Minimum lifted distance",
        "int",
        4,
        2,
        256,
        1,
        tooltip="Minimum centroid distance for a non-local relation. Smaller values duplicate local edges.",
        units="px",
    ),
    _p(
        "maximum_lifted_distance",
        "Maximum lifted distance",
        "int",
        24,
        3,
        512,
        1,
        tooltip="Maximum non-local distance. Large values increase graph density and can connect unrelated traces.",
        units="px",
    ),
    _p(
        "lifted_distance_step",
        "Lifted distance step",
        "int",
        4,
        1,
        128,
        1,
        tooltip="Sampling step for sparse lifted candidates. Smaller steps add more edges and cost memory.",
        units="px",
    ),
    _p(
        "lifted_attraction_weight",
        "Lifted attraction weight",
        "float",
        0.45,
        0.0,
        8.0,
        0.05,
        tooltip="Supports same-trace continuity across short missing evidence. Too high a value increases merges.",
    ),
    _p(
        "lifted_repulsion_weight",
        "Lifted repulsion weight",
        "float",
        0.75,
        0.0,
        8.0,
        0.05,
        tooltip="Preserves separation across short local boundary failures. Too high a value increases splits.",
    ),
    _p(
        "orientation_aligned_lifted_edges",
        "Orientation-aligned lifted edges",
        "bool",
        True,
        tooltip="Restricts lifted candidates using the local orientation field rather than hard-coded image axes.",
    ),
    _p(
        "cross_boundary_lifted_repulsion",
        "Cross-boundary lifted repulsion",
        "bool",
        True,
        tooltip="Adds repulsion when the line between regions crosses persistent boundary evidence.",
    ),
    _p(
        "same_trace_lifted_attraction",
        "Same-trace lifted attraction",
        "bool",
        True,
        tooltip="Adds attraction between orientation-compatible regions with similar material evidence.",
    ),
    _p(
        "lifted_confidence_threshold",
        "Lifted confidence threshold",
        "float",
        0.6,
        0.0,
        1.0,
        0.01,
        tooltip="Only non-local relations above this confidence are added. Higher values make the lifted graph sparser.",
    ),
    _p(
        "maximum_lifted_edges",
        "Maximum lifted edges",
        "int",
        200000,
        0,
        5000000,
        1000,
        tooltip="Hard sparse-density cap preventing quadratic graph construction.",
        advanced=True,
    ),
)
