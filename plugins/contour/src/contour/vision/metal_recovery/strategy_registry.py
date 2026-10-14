"""Declarative registry for production conductor-segmentation strategies.

The registry is deliberately independent from Qt.  Backend validation, the
settings serializer, presets and the UI all consume the same parameter
metadata, so defaults and ranges cannot silently diverge.
"""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal

from .strategy_registry_parts.module_state import (
    ParameterKind,
    ParameterValue,
    _EXISTING,
    IMPLEMENTED_NEW_STRATEGIES,
)
from .strategy_registry_parts.parameter_spec import (
    ParameterSpec,
    StrategySpec,
    _p,
    MetalStrategyConfigs,
    normalize_strategy_parameters,
    strategy_spec,
    visible_strategy_specs,
)

from .strategy_registry_tables._classification_parameters import _CLASSIFICATION_PARAMETERS
from .strategy_registry_tables._signed_graph_parameters import _SIGNED_GRAPH_PARAMETERS
from .strategy_registry_tables._owt_parameters import _OWT_PARAMETERS
from .strategy_registry_tables._msp_parameters import _MSP_PARAMETERS
from .strategy_registry_tables._gasp_parameters import _GASP_PARAMETERS
from .strategy_registry_tables._mws_parameters import _MWS_PARAMETERS
from .strategy_registry_tables._multicut_parameters import _MULTICUT_PARAMETERS
from .strategy_registry_tables._lifted_parameters import _LIFTED_PARAMETERS

STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    strategy_id: StrategySpec(strategy_id, display_name, "Existing production strategy.")
    for strategy_id, display_name in _EXISTING
}
STRATEGY_REGISTRY.update(
    {
        "owt_ucm": StrategySpec(
            "owt_ucm",
            "OWT-UCM",
            "Hierarchical segmentation from oriented contour strength. Best suited when physical boundaries are clear but locally incomplete.",
            "contour.vision.metal_recovery.owt_ucm:segment_owt_ucm",
            _OWT_PARAMETERS,
            True,
            True,
        ),
        "graph_multi_separator": StrategySpec(
            "graph_multi_separator",
            "Graph Multi-Separator",
            "Graph partition with explicit separator pixels and optional long-range interactions.",
            "contour.vision.metal_recovery.graph_multi_separator:segment_graph_multi_separator",
            _MSP_PARAMETERS,
            True,
            True,
        ),
        "gasp": StrategySpec(
            "gasp",
            "GASP",
            "Agglomerative clustering of a signed attraction/repulsion graph.",
            "contour.vision.metal_recovery.graph_strategies:segment_gasp",
            _GASP_PARAMETERS,
            True,
            True,
        ),
        "mutex_watershed": StrategySpec(
            "mutex_watershed",
            "Mutex Watershed",
            "Fast signed graph partition using attractive and mutually exclusive relations.",
            "contour.vision.metal_recovery.graph_strategies:segment_mutex_watershed",
            _MWS_PARAMETERS,
            True,
            True,
        ),
        "multicut": StrategySpec(
            "multicut",
            "Multicut",
            "Global graph partition minimizing attractive and repulsive edge costs.",
            "contour.vision.metal_recovery.graph_strategies:segment_multicut",
            _MULTICUT_PARAMETERS,
            True,
            True,
        ),
        "lifted_multicut": StrategySpec(
            "lifted_multicut",
            "Lifted Multicut",
            "Multicut with additional non-local relations for long-range topology constraints.",
            "contour.vision.metal_recovery.graph_strategies:segment_lifted_multicut",
            _LIFTED_PARAMETERS,
            True,
            True,
        ),
        "ic_sem_expert": StrategySpec(
            "ic_sem_expert",
            "IC-SEM Expert Knowledge",
            "Reserved extension point for documented IC-SEM design-rule priors. No unverified algorithm is implemented.",
        ),
    }
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    ParameterKind,
    ParameterValue,
    _EXISTING,
    IMPLEMENTED_NEW_STRATEGIES,
    ParameterSpec,
    StrategySpec,
    _p,
    MetalStrategyConfigs,
    normalize_strategy_parameters,
    strategy_spec,
    visible_strategy_specs,
)

__all__ = [
    "IMPLEMENTED_NEW_STRATEGIES",
    "STRATEGY_REGISTRY",
    "MetalStrategyConfigs",
    "ParameterSpec",
    "ParameterValue",
    "StrategySpec",
    "normalize_strategy_parameters",
    "strategy_spec",
    "visible_strategy_specs",
]
