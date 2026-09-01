"""Versioned contracts for multi-source project analysis jobs."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .plugin_protocol import safe_relative_path


ANALYSIS_PROTOCOL_VERSION = "1.0"
ANALYSIS_JOB_SCHEMA = "kraken.analysis-job.v1"
ANALYSIS_RESULT_SCHEMA = "kraken.analysis-result.v1"


class AnalysisProfileKind(StrEnum):
    MODEL_COMPARISON = "model_comparison"
    GROUND_TRUTH_VALIDATION = "ground_truth_validation"
    CONFIDENCE_AUDIT = "confidence_audit"
    GRID_DEFECTS = "grid_defects"


class AnalysisSourceRole(StrEnum):
    ORIGINAL = "original"
    GROUND_TRUTH = "ground_truth"
    MODEL_OUTPUT = "model_output"
    CONFIDENCE = "confidence"
    REFERENCE = "reference"


class AnalysisScaleMode(StrEnum):
    ABSOLUTE = "absolute"
    WITHIN_RUN = "within_run"


class AnalysisOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _finite_number(value: object, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _object_list(value: object, field_name: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} entries must be objects")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class AnalysisParameter:
    key: str
    value: str | int | float | bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _required_text(self.key, "parameter.key"))

    def to_payload(self) -> dict[str, object]:
        return {"key": self.key, "value": self.value}

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisParameter":
        value = payload.get("value")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("parameter.value must be a scalar")
        return cls(key=str(payload.get("key", "")), value=value)


@dataclass(frozen=True, slots=True)
class AnalysisArtifactInput:
    """One role-bound artifact staged for a frame."""

    binding_key: str
    role: AnalysisSourceRole
    artifact_id: str
    artifact_version_id: str
    relative_path: str
    media_type: str
    sha256: str
    display_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_key", _required_text(self.binding_key, "artifact.binding_key"))
        object.__setattr__(self, "artifact_id", _required_text(self.artifact_id, "artifact.artifact_id"))
        object.__setattr__(self, "artifact_version_id", _required_text(self.artifact_version_id, "artifact.artifact_version_id"))
        object.__setattr__(self, "relative_path", safe_relative_path(self.relative_path))
        object.__setattr__(self, "media_type", _required_text(self.media_type, "artifact.media_type"))
        digest = _required_text(self.sha256, "artifact.sha256").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("artifact.sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "sha256", digest)

    def to_payload(self) -> dict[str, object]:
        return {
            "binding_key": self.binding_key,
            "role": self.role.value,
            "artifact_id": self.artifact_id,
            "artifact_version_id": self.artifact_version_id,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "display_name": self.display_name,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisArtifactInput":
        return cls(
            binding_key=str(payload.get("binding_key", "")),
            role=AnalysisSourceRole(str(payload.get("role", ""))),
            artifact_id=str(payload.get("artifact_id", "")),
            artifact_version_id=str(payload.get("artifact_version_id", "")),
            relative_path=str(payload.get("relative_path", "")),
            media_type=str(payload.get("media_type", "")),
            sha256=str(payload.get("sha256", "")),
            display_name=str(payload.get("display_name", "")),
        )


@dataclass(frozen=True, slots=True)
class AnalysisFrameInput:
    frame_id: str
    x: int
    y: int
    artifacts: tuple[AnalysisArtifactInput, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _required_text(self.frame_id, "frame.frame_id"))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if not self.artifacts:
            raise ValueError("frame.artifacts must not be empty")
        binding_keys = [artifact.binding_key for artifact in self.artifacts]
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError(f"Duplicate artifact binding in frame {self.frame_id}")

    def to_payload(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "x": self.x,
            "y": self.y,
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisFrameInput":
        raw_artifacts = _object_list(payload.get("artifacts", []), "frame.artifacts")
        return cls(
            frame_id=str(payload.get("frame_id", "")),
            x=int(payload.get("x", 0)),
            y=int(payload.get("y", 0)),
            artifacts=tuple(AnalysisArtifactInput.from_payload(item) for item in raw_artifacts),
        )


@dataclass(frozen=True, slots=True)
class AnalysisJobManifest:
    job_id: str
    project_id: str
    profile: AnalysisProfileKind
    frames: tuple[AnalysisFrameInput, ...]
    parameters: tuple[AnalysisParameter, ...] = ()
    protocol_version: str = ANALYSIS_PROTOCOL_VERSION
    schema: str = ANALYSIS_JOB_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _required_text(self.job_id, "job_id"))
        object.__setattr__(self, "project_id", _required_text(self.project_id, "project_id"))
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if self.protocol_version != ANALYSIS_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported analysis protocol: {self.protocol_version}")
        if self.schema != ANALYSIS_JOB_SCHEMA:
            raise ValueError(f"Unsupported analysis job schema: {self.schema}")
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Analysis frame ids must be unique")
        parameter_keys = [parameter.key for parameter in self.parameters]
        if len(parameter_keys) != len(set(parameter_keys)):
            raise ValueError("Analysis parameter keys must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "project_id": self.project_id,
            "profile": self.profile.value,
            "frames": [frame.to_payload() for frame in self.frames],
            "parameters": [parameter.to_payload() for parameter in self.parameters],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisJobManifest":
        raw_frames = _object_list(payload.get("frames", []), "frames")
        raw_parameters = _object_list(payload.get("parameters", []), "parameters")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            job_id=str(payload.get("job_id", "")),
            project_id=str(payload.get("project_id", "")),
            profile=AnalysisProfileKind(str(payload.get("profile", ""))),
            frames=tuple(AnalysisFrameInput.from_payload(item) for item in raw_frames),
            parameters=tuple(AnalysisParameter.from_payload(item) for item in raw_parameters),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "AnalysisJobManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Analysis job manifest must contain an object")
        return cls.from_payload(payload)


@dataclass(frozen=True, slots=True)
class AnalysisMetricValue:
    key: str
    raw_value: float
    goodness: float
    percentile: float | None = None
    unit: str = ""
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _required_text(self.key, "metric.key"))
        object.__setattr__(self, "raw_value", _finite_number(self.raw_value, "metric.raw_value"))
        goodness = _finite_number(self.goodness, "metric.goodness")
        if not 0.0 <= goodness <= 1.0:
            raise ValueError("metric.goodness must be in the range 0..1")
        object.__setattr__(self, "goodness", goodness)
        if self.percentile is not None:
            percentile = _finite_number(self.percentile, "metric.percentile")
            if not 0.0 <= percentile <= 100.0:
                raise ValueError("metric.percentile must be in the range 0..100")
            object.__setattr__(self, "percentile", percentile)

    def to_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "raw_value": self.raw_value,
            "goodness": self.goodness,
            "percentile": self.percentile,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisMetricValue":
        percentile = payload.get("percentile")
        return cls(
            key=str(payload.get("key", "")),
            raw_value=float(payload.get("raw_value", 0.0)),
            goodness=float(payload.get("goodness", 0.0)),
            percentile=None if percentile is None else float(percentile),
            unit=str(payload.get("unit", "")),
            higher_is_better=bool(payload.get("higher_is_better", True)),
        )


@dataclass(frozen=True, slots=True)
class AnalysisAnomalyRegion:
    x: float
    y: float
    width: float
    height: float
    error_type: str
    severity: float

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "width", "height", "severity"):
            value = _finite_number(getattr(self, field_name), f"anomaly.{field_name}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"anomaly.{field_name} must be in the range 0..1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "error_type", _required_text(self.error_type, "anomaly.error_type"))

    def to_payload(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "error_type": self.error_type,
            "severity": self.severity,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisAnomalyRegion":
        return cls(
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
            width=float(payload.get("width", 0.0)),
            height=float(payload.get("height", 0.0)),
            error_type=str(payload.get("error_type", "")),
            severity=float(payload.get("severity", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class AnalysisFrameResult:
    frame_id: str
    x: int
    y: int
    status: str
    metrics: tuple[AnalysisMetricValue, ...]
    anomalies: tuple[AnalysisAnomalyRegion, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _required_text(self.frame_id, "result.frame_id"))
        object.__setattr__(self, "status", _required_text(self.status, "result.status"))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "anomalies", tuple(self.anomalies))
        metric_keys = [metric.key for metric in self.metrics]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError(f"Duplicate metric key in frame {self.frame_id}")

    def to_payload(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "x": self.x,
            "y": self.y,
            "status": self.status,
            "metrics": [metric.to_payload() for metric in self.metrics],
            "anomalies": [anomaly.to_payload() for anomaly in self.anomalies],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisFrameResult":
        raw_metrics = _object_list(payload.get("metrics", []), "result.metrics")
        raw_anomalies = _object_list(payload.get("anomalies", []), "result.anomalies")
        return cls(
            frame_id=str(payload.get("frame_id", "")),
            x=int(payload.get("x", 0)),
            y=int(payload.get("y", 0)),
            status=str(payload.get("status", "")),
            metrics=tuple(AnalysisMetricValue.from_payload(item) for item in raw_metrics),
            anomalies=tuple(AnalysisAnomalyRegion.from_payload(item) for item in raw_anomalies),
        )


@dataclass(frozen=True, slots=True)
class AnalysisScaleDefinition:
    metric_key: str
    mode: AnalysisScaleMode
    low: float
    high: float
    p05: float | None = None
    p50: float | None = None
    p95: float | None = None
    clipped_low: int = 0
    clipped_high: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_key", _required_text(self.metric_key, "scale.metric_key"))
        object.__setattr__(self, "low", _finite_number(self.low, "scale.low"))
        object.__setattr__(self, "high", _finite_number(self.high, "scale.high"))
        if self.high <= self.low:
            raise ValueError("scale.high must be greater than scale.low")
        if self.clipped_low < 0 or self.clipped_high < 0:
            raise ValueError("clipped counts must not be negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "metric_key": self.metric_key,
            "mode": self.mode.value,
            "low": self.low,
            "high": self.high,
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "clipped_low": self.clipped_low,
            "clipped_high": self.clipped_high,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisScaleDefinition":
        return cls(
            metric_key=str(payload.get("metric_key", "")),
            mode=AnalysisScaleMode(str(payload.get("mode", ""))),
            low=float(payload.get("low", 0.0)),
            high=float(payload.get("high", 1.0)),
            p05=None if payload.get("p05") is None else float(payload["p05"]),
            p50=None if payload.get("p50") is None else float(payload["p50"]),
            p95=None if payload.get("p95") is None else float(payload["p95"]),
            clipped_low=int(payload.get("clipped_low", 0)),
            clipped_high=int(payload.get("clipped_high", 0)),
        )


@dataclass(frozen=True, slots=True)
class AnalysisResultManifest:
    job_id: str
    project_id: str
    profile: AnalysisProfileKind
    outcome: AnalysisOutcome
    frames: tuple[AnalysisFrameResult, ...]
    scales: tuple[AnalysisScaleDefinition, ...]
    message: str = ""
    protocol_version: str = ANALYSIS_PROTOCOL_VERSION
    schema: str = ANALYSIS_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _required_text(self.job_id, "job_id"))
        object.__setattr__(self, "project_id", _required_text(self.project_id, "project_id"))
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "scales", tuple(self.scales))
        if self.protocol_version != ANALYSIS_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported analysis protocol: {self.protocol_version}")
        if self.schema != ANALYSIS_RESULT_SCHEMA:
            raise ValueError(f"Unsupported analysis result schema: {self.schema}")
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Analysis result frame ids must be unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "project_id": self.project_id,
            "profile": self.profile.value,
            "outcome": self.outcome.value,
            "frames": [frame.to_payload() for frame in self.frames],
            "scales": [scale.to_payload() for scale in self.scales],
            "message": self.message,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AnalysisResultManifest":
        raw_frames = _object_list(payload.get("frames", []), "result.frames")
        raw_scales = _object_list(payload.get("scales", []), "result.scales")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            job_id=str(payload.get("job_id", "")),
            project_id=str(payload.get("project_id", "")),
            profile=AnalysisProfileKind(str(payload.get("profile", ""))),
            outcome=AnalysisOutcome(str(payload.get("outcome", ""))),
            frames=tuple(AnalysisFrameResult.from_payload(item) for item in raw_frames),
            scales=tuple(AnalysisScaleDefinition.from_payload(item) for item in raw_scales),
            message=str(payload.get("message", "")),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "AnalysisResultManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Analysis result manifest must contain an object")
        return cls.from_payload(payload)


__all__ = [
    "ANALYSIS_JOB_SCHEMA",
    "ANALYSIS_PROTOCOL_VERSION",
    "ANALYSIS_RESULT_SCHEMA",
    "AnalysisAnomalyRegion",
    "AnalysisArtifactInput",
    "AnalysisFrameInput",
    "AnalysisFrameResult",
    "AnalysisJobManifest",
    "AnalysisMetricValue",
    "AnalysisOutcome",
    "AnalysisParameter",
    "AnalysisProfileKind",
    "AnalysisResultManifest",
    "AnalysisScaleDefinition",
    "AnalysisScaleMode",
    "AnalysisSourceRole",
]
