"""processing: pipeline step config."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any

from contour.application.processing_parts.module_state import (
    ALGORITHM_BACKEND_LEGACY,
    ALGORITHM_BACKEND_SEM,
    RECOGNITION_MODE_CONDUCTORS,
    RECOGNITION_MODE_DISABLED,
    RECOGNITION_MODE_METAL,
    RECOGNITION_MODE_VIA,
    VIA_CHANNEL_MODE_COLUMNS,
    VIA_CHANNEL_MODE_GRAYSCALE,
    VIA_CHANNEL_MODE_RED_BLUE,
    VIA_DISPLAY_MODE_CIRCLE,
    VIA_DISPLAY_MODE_RECTANGLE,
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
    VIA_SIZE_MODE_FIXED,
    VIA_SIZE_MODE_RANGE,
)

def normalize_via_display_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    return VIA_DISPLAY_MODE_RECTANGLE if text == VIA_DISPLAY_MODE_RECTANGLE else VIA_DISPLAY_MODE_CIRCLE

def normalize_via_size_mode(value: Any) -> str:
    return VIA_SIZE_MODE_FIXED if str(value).strip().lower() == VIA_SIZE_MODE_FIXED else VIA_SIZE_MODE_RANGE

def normalize_via_search_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return VIA_SEARCH_MODE_HEURISTIC
    if text == VIA_SEARCH_MODE_HEURISTIC:
        return VIA_SEARCH_MODE_HEURISTIC
    if text == VIA_SEARCH_MODE_TEMPLATE:
        return VIA_SEARCH_MODE_TEMPLATE
    if text == VIA_SEARCH_MODE_HYBRID:
        return VIA_SEARCH_MODE_HYBRID
    if text == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG:
        return VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG
    return VIA_SEARCH_MODE_HEURISTIC

def normalize_via_channel_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return VIA_CHANNEL_MODE_GRAYSCALE
    if text == VIA_CHANNEL_MODE_COLUMNS:
        return VIA_CHANNEL_MODE_COLUMNS
    if text in {"gray", "grey", "grayscale"}:
        return VIA_CHANNEL_MODE_GRAYSCALE
    if text in {"rb", "red_blue", "red-blue", "redblue"}:
        return VIA_CHANNEL_MODE_RED_BLUE
    return VIA_CHANNEL_MODE_GRAYSCALE

def _normalize_bright_via_mask_mode(value: Any) -> str:
    return "AND" if str(value or "").strip().upper() == "AND" else "OR"

def _normalize_bright_via_metal_constraint_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"disabled", "off", "none", "false", "0"}:
        return "disabled"
    if text in {"strict", "hard"}:
        return "strict"
    return "soft"

def _odd_positive(value: Any, *, minimum: int) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = int(minimum)
    size = max(int(minimum), size)
    if size % 2 == 0:
        size += 1
    return size

def normalize_algorithm_backend(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {ALGORITHM_BACKEND_SEM, "new", "sem_auto", "auto_sem", "legacy_via"}:
        return ALGORITHM_BACKEND_SEM
    return ALGORITHM_BACKEND_LEGACY

def normalize_recognition_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {RECOGNITION_MODE_DISABLED, "off", "none", "false", "0"}:
        return RECOGNITION_MODE_DISABLED
    if text in {RECOGNITION_MODE_METAL, "metal", "metallization", "металл"}:
        return RECOGNITION_MODE_CONDUCTORS
    if text in {RECOGNITION_MODE_VIA, "vias", "contact", "contacts"}:
        return RECOGNITION_MODE_VIA
    if text in {RECOGNITION_MODE_CONDUCTORS, "conductor", "wires", "проводники", "traces"}:
        return RECOGNITION_MODE_CONDUCTORS
    return RECOGNITION_MODE_CONDUCTORS

def normalize_metal_segmentation_strategy(value: Any) -> str:
    from ...vision.metal_recovery.segmentation import normalize_metal_segmentation_strategy as _norm

    return _norm(value)

def normalize_metal_segmentation_method(value: Any) -> str:
    """Deprecated alias — maps to segmentation strategy."""
    return normalize_metal_segmentation_strategy(value)

def normalize_metal_sensitivity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "низкая", "низк"}:
        return "low"
    if text in {"high", "высокая", "высок"}:
        return "high"
    return "medium"

def parse_integer_value_list(payload: Any) -> list[int]:
    if payload in (None, ""):
        return []
    if isinstance(payload, str):
        raw_values = re.split(r"[\s,;]+", payload.strip())
    elif isinstance(payload, (list, tuple, set)):
        raw_values = list(payload)
    else:
        raw_values = [payload]

    values: set[int] = set()
    for raw_value in raw_values:
        if raw_value in (None, ""):
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        try:
            parsed = int(float(text))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            values.add(parsed)
    return sorted(values)

def _serialize_template_images(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        return []
    serialized: list[Any] = []
    for item in payload:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, list):
            serialized.append(item)
    return serialized

def _parse_template_images(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, list)]

@dataclass(frozen=True, slots=True)
class OperationParameterSpec:
    name: str
    label: str
    kind: str
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    decimals: int = 3
    options: list[str] = field(default_factory=list)
    tooltip: str = ""

@dataclass(slots=True)
class PipelineStepConfig:
    operation: str
    name: str
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> PipelineStepConfig:
        return PipelineStepConfig(
            operation=self.operation,
            name=self.name,
            enabled=self.enabled,
            parameters=dict(self.parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "name": self.name,
            "enabled": self.enabled,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PipelineStepConfig:
        return cls(
            operation=str(payload["operation"]),
            name=str(payload.get("name") or payload["operation"]),
            enabled=bool(payload.get("enabled", True)),
            parameters=dict(payload.get("parameters", {})),
        )

@dataclass(slots=True)
class ContourDebugCandidate:
    contour_index: int
    bbox: tuple[int, int, int, int]
    area: float = 0.0
    perimeter: float = 0.0
    roundness: float = 0.0
    accepted: bool = False
    reason: str = ""
    source: str = ""
    score: float = 0.0
    effective_width: float = 0.0
    width_metric: str = ""
    template_index: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
