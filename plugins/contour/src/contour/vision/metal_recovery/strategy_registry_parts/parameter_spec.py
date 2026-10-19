"""strategy_registry: parameter spec."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from contour.vision.metal_recovery.strategy_registry_parts.module_state import IMPLEMENTED_NEW_STRATEGIES

@dataclass(frozen=True, slots=True)
class ParameterSpec:
    key: str
    label: str
    kind: ParameterKind
    default: ParameterValue
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    tooltip: str = ""
    advanced: bool = False
    units: str = ""

    def normalize(self, value: Any) -> ParameterValue:
        if self.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if self.kind == "choice":
            text = str(value)
            allowed = {choice_value for choice_value, _label in self.choices}
            return text if text in allowed else self.default
        if self.kind == "int":
            parsed: float = float(int(value))
        else:
            parsed = float(value)
        if self.minimum is not None:
            parsed = max(float(self.minimum), parsed)
        if self.maximum is not None:
            parsed = min(float(self.maximum), parsed)
        return int(parsed) if self.kind == "int" else float(parsed)

@dataclass(frozen=True, slots=True)
class StrategySpec:
    strategy_id: str
    display_name: str
    description: str
    backend_path: str | None = None
    parameters: tuple[ParameterSpec, ...] = ()
    partition_based: bool = False
    preserves_instances: bool = False

    def load_backend(self) -> Callable[..., Any] | None:
        if self.backend_path is None:
            return None
        module_name, attribute = self.backend_path.split(":", 1)
        return getattr(import_module(module_name), attribute)

def _p(
    key: str,
    label: str,
    kind: ParameterKind,
    default: ParameterValue,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    *,
    choices: tuple[tuple[str, str], ...] = (),
    tooltip: str,
    advanced: bool = False,
    units: str = "",
) -> ParameterSpec:
    return ParameterSpec(
        key=key,
        label=label,
        kind=kind,
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        choices=choices,
        tooltip=tooltip,
        advanced=advanced,
        units=units,
    )

@dataclass(slots=True)
class MetalStrategyConfigs:
    """Validated nested strategy settings with backward-compatible defaults."""

    values: dict[str, dict[str, ParameterValue]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> MetalStrategyConfigs:
        source = raw or {}
        return cls(
            values={
                strategy_id: normalize_strategy_parameters(
                    strategy_id,
                    source.get(strategy_id) if isinstance(source.get(strategy_id), Mapping) else None,
                )
                for strategy_id in sorted(IMPLEMENTED_NEW_STRATEGIES)
            }
        )

    def for_strategy(self, strategy_id: str) -> dict[str, ParameterValue]:
        return normalize_strategy_parameters(strategy_id, self.values.get(strategy_id))

    def to_dict(self) -> dict[str, dict[str, ParameterValue]]:
        return {key: dict(value) for key, value in self.values.items()}

def normalize_strategy_parameters(
    strategy_id: str,
    raw: Mapping[str, Any] | None,
) -> dict[str, ParameterValue]:
    from contour.vision.metal_recovery.strategy_registry import STRATEGY_REGISTRY

    spec = STRATEGY_REGISTRY.get(strategy_id)
    if spec is None:
        raise ValueError(f"Unknown metal segmentation strategy: {strategy_id}")
    source = raw or {}
    normalized: dict[str, ParameterValue] = {}
    for parameter in spec.parameters:
        try:
            normalized[parameter.key] = parameter.normalize(source.get(parameter.key, parameter.default))
        except (TypeError, ValueError):
            normalized[parameter.key] = parameter.default
    if strategy_id == "lifted_multicut":
        minimum = int(normalized["minimum_lifted_distance"])
        maximum = int(normalized["maximum_lifted_distance"])
        if minimum > maximum:
            maximum = minimum
            normalized["maximum_lifted_distance"] = maximum
    background = float(normalized.get("minimum_background_confidence", 0.0))
    metal = float(normalized.get("minimum_metal_confidence", 1.0))
    if background > metal:
        normalized["minimum_background_confidence"] = metal
    return normalized

def strategy_spec(strategy_id: str) -> StrategySpec:
    from contour.vision.metal_recovery.strategy_registry import STRATEGY_REGISTRY

    try:
        return STRATEGY_REGISTRY[strategy_id]
    except KeyError as exc:
        raise ValueError(f"Unknown metal segmentation strategy: {strategy_id}") from exc

def visible_strategy_specs() -> tuple[StrategySpec, ...]:
    from contour.vision.metal_recovery.strategy_registry import STRATEGY_REGISTRY

    return tuple(spec for spec in STRATEGY_REGISTRY.values() if spec.strategy_id != "ic_sem_expert")

if TYPE_CHECKING:
    from contour.vision.metal_recovery.strategy_registry_parts.module_state import ParameterKind, ParameterValue
