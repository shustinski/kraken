"""_CLASSIFICATION_PARAMETERS."""
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

_CLASSIFICATION_PARAMETERS = (
    _p(
        "core_evidence_weight",
        "Core metal evidence weight",
        "float",
        1.0,
        0.0,
        8.0,
        0.1,
        tooltip="Higher values favor regions containing reliable conductor-core seeds. Too high a value can retain isolated bright texture.",
    ),
    _p(
        "substrate_evidence_weight",
        "Substrate evidence weight",
        "float",
        1.0,
        0.0,
        8.0,
        0.1,
        tooltip="Higher values reject regions supported by substrate or groove evidence. Too high a value can lose dark conductor interiors.",
    ),
    _p(
        "intensity_evidence_weight",
        "Intensity evidence weight",
        "float",
        0.45,
        0.0,
        8.0,
        0.05,
        tooltip="Weights robust region intensity without assuming that every metal pixel is bright. Too high a value makes illumination dominate structure.",
    ),
    _p(
        "local_contrast_evidence_weight",
        "Local contrast evidence weight",
        "float",
        0.8,
        0.0,
        8.0,
        0.05,
        tooltip="Higher values favor regions that differ from their local background. Too high a value may fragment textured plates.",
    ),
    _p(
        "boundary_side_evidence_weight",
        "Boundary-side evidence weight",
        "float",
        0.6,
        0.0,
        8.0,
        0.05,
        tooltip="Weights the intensity transition across a region boundary. Too high a value can reject border-touching conductors.",
    ),
    _p(
        "minimum_metal_confidence",
        "Minimum metal confidence",
        "float",
        0.6,
        0.0,
        1.0,
        0.01,
        tooltip="Minimum normalized material score classified as metal. Higher values reduce false metal but increase misses.",
    ),
    _p(
        "minimum_background_confidence",
        "Minimum background confidence",
        "float",
        0.45,
        0.0,
        1.0,
        0.01,
        tooltip="Scores below this level are background. Keep it no greater than the metal threshold to leave an explicit ambiguous interval.",
    ),
    _p(
        "ambiguous_region_policy",
        "Ambiguous region policy",
        "choice",
        "background",
        choices=(("metal", "Metal"), ("background", "Background"), ("preserve", "Preserve seed evidence")),
        tooltip="Controls regions between the background and metal confidence thresholds. Preserve uses high-confidence core evidence only.",
    ),
)
