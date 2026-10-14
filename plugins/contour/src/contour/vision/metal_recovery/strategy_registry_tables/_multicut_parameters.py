"""_MULTICUT_PARAMETERS."""
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

_MULTICUT_PARAMETERS = (
    *_SIGNED_GRAPH_PARAMETERS,
    _p(
        "cost_transform",
        "Cost transform",
        "choice",
        "log_odds",
        choices=(("log_odds", "Log odds"), ("signed_linear", "Signed linear")),
        tooltip="Log odds emphasizes confident probabilities; signed linear keeps a bounded directly interpretable cost.",
    ),
    _p(
        "affinity_bias",
        "Probability / affinity bias",
        "float",
        0.5,
        0.01,
        0.99,
        0.01,
        tooltip="Neutral probability for the cost transform. Higher values require stronger attraction before a merge is profitable.",
    ),
    _p(
        "attraction_cost_scale",
        "Attraction cost scale",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Scales the reward for keeping attractive edges uncut.",
    ),
    _p(
        "repulsion_cost_scale",
        "Repulsion cost scale",
        "float",
        1.0,
        0.0,
        8.0,
        0.05,
        tooltip="Scales the penalty for keeping repulsive edges inside one component.",
    ),
    _p(
        "solver",
        "Solver",
        "choice",
        "greedy_additive",
        choices=(("greedy_additive", "Greedy Additive"),),
        tooltip="Only the deterministic greedy-additive solver is shipped; unavailable exact or fusion solvers are not advertised.",
    ),
    _p(
        "initialization",
        "Initialization",
        "choice",
        "singletons",
        choices=(("singletons", "Singleton regions"), ("positive_components", "Positive-edge components")),
        tooltip="Singletons are conservative; positive components start from confident attractive connectivity.",
    ),
    _p(
        "maximum_iterations",
        "Maximum iterations",
        "int",
        200000,
        1,
        5000000,
        1000,
        tooltip="Maximum greedy contractions. Too low a value returns an unfinished partition.",
        advanced=True,
    ),
    _p(
        "time_limit_seconds",
        "Time limit",
        "float",
        30.0,
        0.1,
        3600.0,
        0.5,
        tooltip="Deterministic wall-clock safety limit. A reached limit is reported explicitly in debug metadata.",
        advanced=True,
        units="s",
    ),
    _p(
        "convergence_tolerance",
        "Convergence tolerance",
        "float",
        0.0,
        0.0,
        1.0,
        0.001,
        tooltip="Minimum positive objective gain required for another contraction.",
        advanced=True,
    ),
    *_CLASSIFICATION_PARAMETERS,
)
