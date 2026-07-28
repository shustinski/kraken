"""Versioned, dependency-free contracts for Kraken plugin jobs.

The protocol deliberately transports metadata only.  Large assets are
materialized into a per-job staging directory by Kraken Agent and are
referenced by safe POSIX relative paths.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Mapping


PLUGIN_PROTOCOL_VERSION = "1.0"
PLUGIN_JOB_SCHEMA = "kraken.plugin-job.v1"
PLUGIN_RESULT_SCHEMA = "kraken.plugin-result.v1"
PLUGIN_JOB_SCHEMA_V2 = "kraken.plugin-job.v2"
PLUGIN_RESULT_SCHEMA_V2 = "kraken.plugin-result.v2"


class PluginOperation(StrEnum):
    VECTORIZE_FRAMES = "frames.vectorize.v1"
    BINARY_SEGMENT_FRAMES = "frames.binary-segment.v1"
    PREPARE_DATASET = "frames.dataset.prepare.v1"
    TRAIN_MODEL = "dataset.model.train.v1"
    BINARY_SEGMENT_FRAMES_V2 = "frames.binary-segment.v2"
    ANALYZE_LAYER_CONFIDENCE = "layer.confidence.analyze.v1"


class PluginRunMode(StrEnum):
    INTERACTIVE = "interactive"
    HEADLESS = "headless"


class PluginJobOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def safe_relative_path(value: object) -> str:
    """Return a normalized safe transport path or raise ``ValueError``."""

    text = _required_text(value, "relative_path").replace("\\", "/")
    raw_parts = text.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"Unsafe relative path: {text}")
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
    }
    for part in raw_parts:
        if (
            any(ord(char) < 32 for char in part)
            or ":" in part
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].casefold() in reserved
        ):
            raise ValueError(f"Unsafe relative path: {text}")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"Unsafe relative path: {text}")
    return candidate.as_posix()


def _sha256(value: object) -> str:
    text = _required_text(value, "sha256").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    return text


@dataclass(frozen=True, slots=True)
class PluginCapability:
    operation: str
    modes: tuple[str, ...] = (PluginRunMode.INTERACTIVE.value,)
    input_media_types: tuple[str, ...] = ()
    output_media_types: tuple[str, ...] = ()
    maximum_batch_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _required_text(self.operation, "operation"))
        if not self.modes:
            raise ValueError("At least one plugin mode is required")
        if self.maximum_batch_size is not None and self.maximum_batch_size <= 0:
            raise ValueError("maximum_batch_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "operation": self.operation,
            "modes": list(self.modes),
            "input_media_types": list(self.input_media_types),
            "output_media_types": list(self.output_media_types),
        }
        if self.maximum_batch_size is not None:
            result["maximum_batch_size"] = self.maximum_batch_size
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginCapability":
        maximum = payload.get("maximum_batch_size")
        return cls(
            operation=str(payload.get("operation", "")),
            modes=tuple(str(item) for item in payload.get("modes", (PluginRunMode.INTERACTIVE.value,))),
            input_media_types=tuple(str(item) for item in payload.get("input_media_types", ())),
            output_media_types=tuple(str(item) for item in payload.get("output_media_types", ())),
            maximum_batch_size=None if maximum is None else int(maximum),
        )


@dataclass(frozen=True, slots=True)
class PluginFrameInput:
    frame_id: str
    x: int
    y: int
    artifact_version_id: str
    sha256: str
    media_type: str
    relative_path: str

    def __post_init__(self) -> None:
        for field_name in ("frame_id", "artifact_version_id", "media_type"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        if self.x < 1 or self.y < 1:
            raise ValueError("Frame coordinates are one-based positive integers")
        object.__setattr__(self, "sha256", _sha256(self.sha256))
        object.__setattr__(self, "relative_path", safe_relative_path(self.relative_path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "x": self.x,
            "y": self.y,
            "artifact_version_id": self.artifact_version_id,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginFrameInput":
        return cls(
            frame_id=str(payload.get("frame_id", "")),
            x=int(payload.get("x", 0)),
            y=int(payload.get("y", 0)),
            artifact_version_id=str(payload.get("artifact_version_id", "")),
            sha256=str(payload.get("sha256", "")),
            media_type=str(payload.get("media_type", "")),
            relative_path=str(payload.get("relative_path", "")),
        )


@dataclass(frozen=True, slots=True)
class PluginJobManifest:
    job_id: str
    operation: str
    project_id: str
    layer_id: str
    actor_id: str
    target_representation_id: str
    inputs: tuple[PluginFrameInput, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema: str = PLUGIN_JOB_SCHEMA
    protocol_version: str = PLUGIN_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "job_id",
            "operation",
            "project_id",
            "layer_id",
            "actor_id",
            "target_representation_id",
        ):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        if self.schema != PLUGIN_JOB_SCHEMA:
            raise ValueError(f"Unsupported job schema: {self.schema}")
        if self.protocol_version != PLUGIN_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported plugin protocol: {self.protocol_version}")
        if not self.inputs:
            raise ValueError("Plugin job requires at least one frame")
        frame_ids = [item.frame_id for item in self.inputs]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Plugin job contains duplicate frame IDs")
        object.__setattr__(self, "parameters", dict(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "operation": self.operation,
            "project_id": self.project_id,
            "layer_id": self.layer_id,
            "actor_id": self.actor_id,
            "target_representation_id": self.target_representation_id,
            "created_at": self.created_at,
            "parameters": dict(self.parameters),
            "inputs": [item.to_dict() for item in self.inputs],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginJobManifest":
        raw_inputs = payload.get("inputs", ())
        if not isinstance(raw_inputs, (list, tuple)):
            raise ValueError("inputs must be an array")
        raw_parameters = payload.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("parameters must be an object")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            job_id=str(payload.get("job_id", "")),
            operation=str(payload.get("operation", "")),
            project_id=str(payload.get("project_id", "")),
            layer_id=str(payload.get("layer_id", "")),
            actor_id=str(payload.get("actor_id", "")),
            target_representation_id=str(payload.get("target_representation_id", "")),
            created_at=str(payload.get("created_at", "")),
            parameters=dict(raw_parameters),
            inputs=tuple(PluginFrameInput.from_dict(item) for item in raw_inputs if isinstance(item, Mapping)),
        )

    @classmethod
    def from_json(cls, raw: str) -> "PluginJobManifest":
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("Plugin job manifest must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class PluginFrameOutput:
    output_id: str
    frame_id: str
    relative_path: str
    sha256: str
    media_type: str
    role: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("output_id", "frame_id", "media_type", "role"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        object.__setattr__(self, "relative_path", safe_relative_path(self.relative_path))
        object.__setattr__(self, "sha256", _sha256(self.sha256))

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_id": self.output_id,
            "frame_id": self.frame_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "role": self.role,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginFrameOutput":
        return cls(
            output_id=str(payload.get("output_id", "")),
            frame_id=str(payload.get("frame_id", "")),
            relative_path=str(payload.get("relative_path", "")),
            sha256=str(payload.get("sha256", "")),
            media_type=str(payload.get("media_type", "")),
            role=str(payload.get("role", "")),
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
        )


class PluginAssetScope(StrEnum):
    FRAME = "frame"
    LAYER = "layer"


@dataclass(frozen=True, slots=True)
class PluginAsset:
    """Role-aware V2 asset; several roles may belong to the same frame."""

    asset_id: str
    role: str
    scope: PluginAssetScope | str
    relative_path: str
    sha256: str
    media_type: str
    frame_id: str | None = None
    representation_id: str | None = None
    x: int | None = None
    y: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("asset_id", "role", "media_type"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "scope", PluginAssetScope(str(self.scope)))
        object.__setattr__(self, "relative_path", safe_relative_path(self.relative_path))
        object.__setattr__(self, "sha256", _sha256(self.sha256))
        if self.scope is PluginAssetScope.FRAME:
            if not self.frame_id or self.x is None or self.y is None or self.x < 1 or self.y < 1:
                raise ValueError("frame assets require frame_id and positive x/y")
        elif self.frame_id is not None or self.x is not None or self.y is not None:
            raise ValueError("layer assets cannot declare frame coordinates")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "role": self.role,
            "scope": self.scope.value,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "frame_id": self.frame_id,
            "representation_id": self.representation_id,
            "x": self.x,
            "y": self.y,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginAsset":
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("asset metadata must be an object")
        return cls(
            asset_id=str(payload.get("asset_id", "")),
            role=str(payload.get("role", "")),
            scope=str(payload.get("scope", "")),
            relative_path=str(payload.get("relative_path", "")),
            sha256=str(payload.get("sha256", "")),
            media_type=str(payload.get("media_type", "")),
            frame_id=None if payload.get("frame_id") is None else str(payload["frame_id"]),
            representation_id=None if payload.get("representation_id") is None else str(payload["representation_id"]),
            x=None if payload.get("x") is None else int(payload["x"]),
            y=None if payload.get("y") is None else int(payload["y"]),
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class PluginJobManifestV2:
    job_id: str
    operation: str
    project_id: str
    layer_id: str
    actor_id: str
    inputs: tuple[PluginAsset, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    target_representation_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema: str = PLUGIN_JOB_SCHEMA_V2
    protocol_version: str = "2.0"

    def __post_init__(self) -> None:
        for name in ("job_id", "operation", "project_id", "layer_id", "actor_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.schema != PLUGIN_JOB_SCHEMA_V2 or self.protocol_version != "2.0":
            raise ValueError("Unsupported V2 plugin manifest")
        if not self.inputs:
            raise ValueError("V2 plugin job requires at least one asset")
        paths = [item.relative_path.casefold() for item in self.inputs]
        if len(paths) != len(set(paths)):
            raise ValueError("V2 input paths must be unique")
        object.__setattr__(self, "parameters", dict(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "protocol_version": self.protocol_version,
            "job_id": self.job_id, "operation": self.operation,
            "project_id": self.project_id, "layer_id": self.layer_id,
            "actor_id": self.actor_id, "created_at": self.created_at,
            "parameters": dict(self.parameters),
            "target_representation_ids": list(self.target_representation_ids),
            "inputs": [item.to_dict() for item in self.inputs],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginJobManifestV2":
        raw_inputs = payload.get("inputs", ())
        raw_parameters = payload.get("parameters", {})
        if not isinstance(raw_inputs, (list, tuple)):
            raise ValueError("inputs must be an array")
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("parameters must be an object")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            job_id=str(payload.get("job_id", "")),
            operation=str(payload.get("operation", "")),
            project_id=str(payload.get("project_id", "")),
            layer_id=str(payload.get("layer_id", "")),
            actor_id=str(payload.get("actor_id", "")),
            created_at=str(payload.get("created_at", "")),
            parameters=dict(raw_parameters),
            target_representation_ids=tuple(str(item) for item in payload.get("target_representation_ids", ())),
            inputs=tuple(PluginAsset.from_dict(item) for item in raw_inputs if isinstance(item, Mapping)),
        )

    @classmethod
    def from_json(cls, raw: str) -> "PluginJobManifestV2":
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("V2 plugin job manifest must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class PluginResultPublicationV2:
    job_id: str
    publication_id: str
    sequence: int
    plugin_id: str
    plugin_version: str
    outputs: tuple[PluginAsset, ...]
    outcome: str = PluginJobOutcome.SUCCEEDED.value
    applied_parameters: Mapping[str, Any] = field(default_factory=dict)
    frame_values: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    report: Mapping[str, Any] = field(default_factory=dict)
    final: bool = False
    schema: str = PLUGIN_RESULT_SCHEMA_V2
    protocol_version: str = "2.0"

    def __post_init__(self) -> None:
        for name in ("job_id", "publication_id", "plugin_id", "plugin_version"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        if self.sequence < 1:
            raise ValueError("publication sequence starts at one")
        if self.schema != PLUGIN_RESULT_SCHEMA_V2 or self.protocol_version != "2.0":
            raise ValueError("Unsupported V2 result publication")
        try:
            PluginJobOutcome(self.outcome)
        except ValueError as exc:
            raise ValueError(f"Unsupported plugin outcome: {self.outcome}") from exc
        object.__setattr__(self, "applied_parameters", dict(self.applied_parameters))
        normalized_values: dict[str, dict[str, float]] = {}
        for frame_id, measurements in self.frame_values.items():
            if not isinstance(measurements, Mapping):
                raise ValueError("frame_values entries must be objects")
            normalized_values[_required_text(frame_id, "frame_id")] = {
                _required_text(name, "measurement"): float(value)
                for name, value in measurements.items()
            }
        object.__setattr__(self, "frame_values", normalized_values)
        object.__setattr__(self, "report", dict(self.report))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "protocol_version": self.protocol_version,
            "job_id": self.job_id, "publication_id": self.publication_id,
            "sequence": self.sequence, "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version, "outcome": self.outcome,
            "final": self.final,
            "applied_parameters": dict(self.applied_parameters),
            "frame_values": {
                frame_id: dict(measurements)
                for frame_id, measurements in self.frame_values.items()
            },
            "report": dict(self.report),
            "outputs": [item.to_dict() for item in self.outputs],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginResultPublicationV2":
        raw_outputs = payload.get("outputs", ())
        parameters = payload.get("applied_parameters", {})
        frame_values = payload.get("frame_values", {})
        report = payload.get("report", {})
        if not isinstance(raw_outputs, (list, tuple)):
            raise ValueError("outputs must be an array")
        if not all(isinstance(value, Mapping) for value in (parameters, frame_values, report)):
            raise ValueError("publication parameters, frame_values and report must be objects")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            job_id=str(payload.get("job_id", "")),
            publication_id=str(payload.get("publication_id", "")),
            sequence=int(payload.get("sequence", 0)),
            plugin_id=str(payload.get("plugin_id", "")),
            plugin_version=str(payload.get("plugin_version", "")),
            outcome=str(payload.get("outcome", PluginJobOutcome.SUCCEEDED.value)),
            final=bool(payload.get("final", False)),
            applied_parameters=dict(parameters),
            frame_values={
                str(frame_id): dict(measurements)
                for frame_id, measurements in frame_values.items()
                if isinstance(measurements, Mapping)
            },
            report=dict(report),
            outputs=tuple(
                PluginAsset.from_dict(item)
                for item in raw_outputs
                if isinstance(item, Mapping)
            ),
        )

    @classmethod
    def from_json(cls, raw: str) -> "PluginResultPublicationV2":
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("V2 plugin result publication must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class PluginResultManifest:
    job_id: str
    outcome: str
    plugin_id: str
    plugin_version: str
    outputs: tuple[PluginFrameOutput, ...] = ()
    applied_parameters: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    completed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema: str = PLUGIN_RESULT_SCHEMA
    protocol_version: str = PLUGIN_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for field_name in ("job_id", "outcome", "plugin_id", "plugin_version"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        try:
            PluginJobOutcome(self.outcome)
        except ValueError as exc:
            raise ValueError(f"Unsupported plugin outcome: {self.outcome}") from exc
        if self.schema != PLUGIN_RESULT_SCHEMA:
            raise ValueError(f"Unsupported result schema: {self.schema}")
        if self.protocol_version != PLUGIN_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported plugin protocol: {self.protocol_version}")
        output_ids = [item.output_id for item in self.outputs]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("Plugin result contains duplicate output IDs")
        object.__setattr__(self, "applied_parameters", dict(self.applied_parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "outcome": self.outcome,
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "completed_at": self.completed_at,
            "applied_parameters": dict(self.applied_parameters),
            "outputs": [item.to_dict() for item in self.outputs],
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginResultManifest":
        raw_outputs = payload.get("outputs", ())
        if not isinstance(raw_outputs, (list, tuple)):
            raise ValueError("outputs must be an array")
        raw_parameters = payload.get("applied_parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("applied_parameters must be an object")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            job_id=str(payload.get("job_id", "")),
            outcome=str(payload.get("outcome", "")),
            plugin_id=str(payload.get("plugin_id", "")),
            plugin_version=str(payload.get("plugin_version", "")),
            completed_at=str(payload.get("completed_at", "")),
            applied_parameters=dict(raw_parameters),
            outputs=tuple(PluginFrameOutput.from_dict(item) for item in raw_outputs if isinstance(item, Mapping)),
            errors=tuple(str(item) for item in payload.get("errors", ())),
        )

    @classmethod
    def from_json(cls, raw: str) -> "PluginResultManifest":
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("Plugin result manifest must be a JSON object")
        return cls.from_dict(payload)


def plugin_supports(
    capabilities: tuple[PluginCapability, ...] | list[PluginCapability],
    operation: str,
    *,
    mode: str | None = None,
    batch_size: int | None = None,
) -> bool:
    for capability in capabilities:
        if capability.operation != operation:
            continue
        if mode is not None and mode not in capability.modes:
            continue
        if (
            batch_size is not None
            and capability.maximum_batch_size is not None
            and batch_size > capability.maximum_batch_size
        ):
            continue
        return True
    return False


def parse_plugin_job(payload: Mapping[str, Any]) -> PluginJobManifest | PluginJobManifestV2:
    schema = str(payload.get("schema", ""))
    if schema == PLUGIN_JOB_SCHEMA:
        return PluginJobManifest.from_dict(payload)
    if schema == PLUGIN_JOB_SCHEMA_V2:
        return PluginJobManifestV2.from_dict(payload)
    raise ValueError(f"Unsupported job schema: {schema}")


def parse_plugin_job_json(raw: str) -> PluginJobManifest | PluginJobManifestV2:
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("Plugin job manifest must be a JSON object")
    return parse_plugin_job(payload)


def parse_plugin_result_json(raw: str) -> PluginResultManifest | PluginResultPublicationV2:
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("Plugin result manifest must be a JSON object")
    schema = str(payload.get("schema", ""))
    if schema == PLUGIN_RESULT_SCHEMA:
        return PluginResultManifest.from_dict(payload)
    if schema == PLUGIN_RESULT_SCHEMA_V2:
        return PluginResultPublicationV2.from_dict(payload)
    raise ValueError(f"Unsupported result schema: {schema}")


__all__ = [
    "PLUGIN_JOB_SCHEMA",
    "PLUGIN_PROTOCOL_VERSION",
    "PLUGIN_RESULT_SCHEMA",
    "PLUGIN_JOB_SCHEMA_V2",
    "PLUGIN_RESULT_SCHEMA_V2",
    "PluginAsset",
    "PluginAssetScope",
    "PluginCapability",
    "PluginFrameInput",
    "PluginFrameOutput",
    "PluginJobManifest",
    "PluginJobManifestV2",
    "PluginJobOutcome",
    "PluginOperation",
    "PluginResultManifest",
    "PluginResultPublicationV2",
    "PluginRunMode",
    "plugin_supports",
    "parse_plugin_job",
    "parse_plugin_job_json",
    "parse_plugin_result_json",
    "safe_relative_path",
]
