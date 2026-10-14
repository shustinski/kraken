"""_MWS_PARAMETERS."""
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

_MWS_PARAMETERS = (
    *_SIGNED_GRAPH_PARAMETERS,
    _p(
        "attractive_neighborhood_offsets",
        "Attractive neighborhood offsets",
        "choice",
        "local",
        choices=(("local", "Local graph offsets"), ("local_plus_diagonal", "Local plus diagonal")),
        tooltip="Defines which nearby relations compete as attractive edges.",
    ),
    _p(
        "mutex_neighborhood_offsets",
        "Mutex neighborhood offsets",
        "choice",
        "local_plus_long_range",
        choices=(("local", "Local"), ("local_plus_long_range", "Local plus long-range")),
        tooltip="Adds non-local mutual-exclusion evidence without changing attractive graph support.",
    ),
    _p(
        "attractive_weight_scale",
        "Attractive weight scale",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Scales attractive edge ordering. Higher values let merges act before competing mutex edges.",
    ),
    _p(
        "mutex_weight_scale",
        "Mutex weight scale",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Scales mutex edge ordering. Higher values establish exclusions earlier and reduce merges.",
    ),
    _p(
        "minimum_mutex_confidence",
        "Minimum mutex confidence",
        "float",
        0.55,
        0.0,
        1.0,
        0.01,
        tooltip="Repulsive relations below this confidence do not become mutex constraints.",
    ),
    _p(
        "long_range_mutex_distance",
        "Long-range mutex distance",
        "int",
        8,
        1,
        128,
        1,
        tooltip="Maximum sparse orientation-aware mutex distance. Too large a value can exclude unrelated structures.",
        units="px",
    ),
    _p(
        "edge_ordering",
        "Edge ordering / normalization",
        "choice",
        "descending_confidence",
        choices=(("descending_confidence", "Descending confidence"), ("signed_margin", "Signed margin")),
        tooltip="Controls the deterministic order in which attractive and mutex relations are processed.",
    ),
    *_CLASSIFICATION_PARAMETERS,
)
