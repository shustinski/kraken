"""Versioned project-scoped configuration for Karakal analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kraken_core.analysis_protocol import (
    AnalysisParameter,
    AnalysisProfileKind,
    AnalysisScaleMode,
    AnalysisSourceRole,
)


KARAKAL_PROFILE_SCHEMA = "karakal.analysis-profile.v1"


def _object_list(value: object, field_name: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} entries must be objects")
    return tuple(value)


class SourceBindingKind(StrEnum):
    FILESYSTEM = "filesystem"
    REPRESENTATION = "representation"


@dataclass(frozen=True, slots=True)
class AnalysisSourceBinding:
    binding_key: str
    role: AnalysisSourceRole
    kind: SourceBindingKind
    source_id: str
    source_version_id: str = ""
    display_name: str = ""

    def __post_init__(self) -> None:
        for field_name in ("binding_key", "source_id"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} is required")

    def to_payload(self) -> dict[str, object]:
        return {
            "binding_key": self.binding_key,
            "role": self.role.value,
            "kind": self.kind.value,
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "display_name": self.display_name,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisSourceBinding":
        return cls(
            binding_key=str(payload.get("binding_key", "")),
            role=AnalysisSourceRole(str(payload.get("role", ""))),
            kind=SourceBindingKind(str(payload.get("kind", ""))),
            source_id=str(payload.get("source_id", "")),
            source_version_id=str(payload.get("source_version_id", "")),
            display_name=str(payload.get("display_name", "")),
        )


@dataclass(frozen=True, slots=True)
class KarakalAnalysisProfileV1:
    profile: AnalysisProfileKind
    bindings: tuple[AnalysisSourceBinding, ...]
    project_id: str = ""
    object_type: str = "polygon"
    metric_key: str = "overall_polygon_score"
    scale_mode: AnalysisScaleMode = AnalysisScaleMode.ABSOLUTE
    gradient_name: str = "accessible_blue_amber"
    visible_layers: tuple[str, ...] = ("quality", "status", "reference")
    parameters: tuple[AnalysisParameter, ...] = ()
    schema: str = KARAKAL_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "visible_layers", tuple(str(layer) for layer in self.visible_layers))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if self.schema != KARAKAL_PROFILE_SCHEMA:
            raise ValueError(f"Unsupported Karakal profile schema: {self.schema}")
        binding_keys = [binding.binding_key for binding in self.bindings]
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError("Profile binding keys must be unique")
        parameter_keys = [parameter.key for parameter in self.parameters]
        if len(parameter_keys) != len(set(parameter_keys)):
            raise ValueError("Profile parameter keys must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "project_id": self.project_id,
            "profile": self.profile.value,
            "bindings": [binding.to_payload() for binding in self.bindings],
            "object_type": self.object_type,
            "metric_key": self.metric_key,
            "scale_mode": self.scale_mode.value,
            "gradient_name": self.gradient_name,
            "visible_layers": list(self.visible_layers),
            "parameters": [parameter.to_payload() for parameter in self.parameters],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "KarakalAnalysisProfileV1":
        raw_bindings = _object_list(payload.get("bindings", []), "bindings")
        raw_parameters = _object_list(payload.get("parameters", []), "parameters")
        raw_layers = payload.get("visible_layers", [])
        if not isinstance(raw_layers, list):
            raise ValueError("visible_layers must be a list")
        if any(not isinstance(item, str) for item in raw_layers):
            raise ValueError("visible_layers entries must be strings")
        return cls(
            schema=str(payload.get("schema", "")),
            project_id=str(payload.get("project_id", "")),
            profile=AnalysisProfileKind(str(payload.get("profile", ""))),
            bindings=tuple(AnalysisSourceBinding.from_payload(item) for item in raw_bindings),
            object_type=str(payload.get("object_type", "polygon")),
            metric_key=str(payload.get("metric_key", "overall_polygon_score")),
            scale_mode=AnalysisScaleMode(str(payload.get("scale_mode", AnalysisScaleMode.ABSOLUTE.value))),
            gradient_name=str(payload.get("gradient_name", "accessible_blue_amber")),
            visible_layers=tuple(str(item) for item in raw_layers),
            parameters=tuple(AnalysisParameter.from_payload(item) for item in raw_parameters),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "KarakalAnalysisProfileV1":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Karakal profile must contain an object")
        return cls.from_payload(payload)


__all__ = [
    "KARAKAL_PROFILE_SCHEMA",
    "AnalysisSourceBinding",
    "KarakalAnalysisProfileV1",
    "SourceBindingKind",
]
