"""Scalable, deterministic contracts for partitioned Karakal analysis runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .analysis_protocol import (
    ANALYSIS_PROTOCOL_VERSION,
    AnalysisFrameInput,
    AnalysisOutcome,
    AnalysisParameter,
)
from .plugin_protocol import safe_relative_path


ANALYSIS_RUN_SCHEMA = "kraken.analysis-run.v1"
ANALYSIS_PARTITION_JOB_SCHEMA = "kraken.analysis-partition-job.v1"
ANALYSIS_PARTITION_RESULT_SCHEMA = "kraken.analysis-partition-result.v1"
ANALYSIS_PARTITION_SIZE = 1000
ANALYSIS_PARTITION_ALGORITHM = "ordered-frame-id.v1"
ANALYSIS_MAX_EXPRESSION_DEPTH = 8

_EXPRESSION_OPERATIONS = frozenset({"source", "xor", "subtract", "compare"})


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _object_list(value: object, field_name: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} entries must be objects")
    return tuple(value)


def canonical_json(payload: object) -> str:
    """Return the stable JSON representation used by run fingerprints."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisExpression:
    """A deliberately small expression tree; it never evaluates arbitrary code."""

    operation: str
    source_key: str = ""
    left: AnalysisExpression | None = None
    right: AnalysisExpression | None = None

    def __post_init__(self) -> None:
        operation = _required_text(self.operation, "expression.operation").lower()
        object.__setattr__(self, "operation", operation)
        if operation not in _EXPRESSION_OPERATIONS:
            raise ValueError(f"Unsupported analysis expression operation: {operation}")
        if operation == "source":
            object.__setattr__(self, "source_key", _required_text(self.source_key, "expression.source_key"))
            if self.left is not None or self.right is not None:
                raise ValueError("source expression must not contain operands")
        else:
            if self.source_key:
                raise ValueError(f"{operation} expression must not contain source_key")
            if self.left is None or self.right is None:
                raise ValueError(f"{operation} expression requires left and right operands")
        if self.depth > ANALYSIS_MAX_EXPRESSION_DEPTH:
            raise ValueError(f"Analysis expression depth must not exceed {ANALYSIS_MAX_EXPRESSION_DEPTH}")

    @property
    def depth(self) -> int:
        if self.operation == "source":
            return 1
        assert self.left is not None and self.right is not None
        return 1 + max(self.left.depth, self.right.depth)

    @property
    def source_keys(self) -> frozenset[str]:
        if self.operation == "source":
            return frozenset({self.source_key})
        assert self.left is not None and self.right is not None
        return self.left.source_keys | self.right.source_keys

    def to_payload(self) -> dict[str, object]:
        if self.operation == "source":
            return {"operation": self.operation, "source_key": self.source_key}
        assert self.left is not None and self.right is not None
        return {
            "operation": self.operation,
            "left": self.left.to_payload(),
            "right": self.right.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisExpression:
        operation = str(payload.get("operation", ""))
        if operation == "source":
            return cls(operation=operation, source_key=str(payload.get("source_key", "")))
        raw_left = payload.get("left")
        raw_right = payload.get("right")
        if not isinstance(raw_left, Mapping) or not isinstance(raw_right, Mapping):
            raise ValueError(f"{operation or 'binary'} expression requires object operands")
        return cls(
            operation=operation,
            left=cls.from_payload(raw_left),
            right=cls.from_payload(raw_right),
        )

    @classmethod
    def source(cls, key: str) -> AnalysisExpression:
        return cls("source", source_key=key)

    @classmethod
    def binary(cls, operation: str, left: AnalysisExpression, right: AnalysisExpression) -> AnalysisExpression:
        return cls(operation, left=left, right=right)


@dataclass(frozen=True, slots=True)
class AnalysisRecipe:
    expression: AnalysisExpression
    metric_keys: tuple[str, ...] = ()
    metric_registry_version: str = "karakal.metrics.v1"

    def __post_init__(self) -> None:
        if self.expression.operation != "compare":
            raise ValueError("Analysis recipe root operation must be compare")
        if self._contains_nested_compare(self.expression.left) or self._contains_nested_compare(self.expression.right):
            raise ValueError("compare is only allowed as the recipe root operation")
        normalized = tuple(_required_text(key, "recipe.metric_key") for key in self.metric_keys)
        if len(normalized) != len(set(normalized)):
            raise ValueError("Analysis metric keys must be unique")
        object.__setattr__(self, "metric_keys", normalized)
        object.__setattr__(
            self,
            "metric_registry_version",
            _required_text(self.metric_registry_version, "recipe.metric_registry_version"),
        )

    @staticmethod
    def _contains_nested_compare(expression: AnalysisExpression | None) -> bool:
        if expression is None:
            return False
        if expression.operation == "compare":
            return True
        return AnalysisRecipe._contains_nested_compare(expression.left) or AnalysisRecipe._contains_nested_compare(
            expression.right
        )

    @property
    def fingerprint(self) -> str:
        return payload_sha256(self.to_payload())

    def validate_bindings(self, available_keys: set[str] | frozenset[str]) -> None:
        missing = sorted(self.expression.source_keys - frozenset(available_keys))
        if missing:
            raise ValueError(f"Analysis recipe references missing bindings: {', '.join(missing)}")

    def to_payload(self) -> dict[str, object]:
        return {
            "expression": self.expression.to_payload(),
            "metric_keys": list(self.metric_keys),
            "metric_registry_version": self.metric_registry_version,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisRecipe:
        raw_expression = payload.get("expression")
        if not isinstance(raw_expression, Mapping):
            raise ValueError("recipe.expression must be an object")
        raw_metric_keys = payload.get("metric_keys", [])
        if not isinstance(raw_metric_keys, list):
            raise ValueError("recipe.metric_keys must be a list")
        return cls(
            expression=AnalysisExpression.from_payload(raw_expression),
            metric_keys=tuple(str(key) for key in raw_metric_keys),
            metric_registry_version=str(payload.get("metric_registry_version", "")),
        )


@dataclass(frozen=True, slots=True)
class AnalysisSourceBinding:
    binding_key: str
    source_id: str
    source_version: str
    display_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_key", _required_text(self.binding_key, "source.binding_key"))
        object.__setattr__(self, "source_id", _required_text(self.source_id, "source.source_id"))
        object.__setattr__(self, "source_version", _required_text(self.source_version, "source.source_version"))

    def to_payload(self) -> dict[str, object]:
        return {
            "binding_key": self.binding_key,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "display_name": self.display_name,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisSourceBinding:
        return cls(
            binding_key=str(payload.get("binding_key", "")),
            source_id=str(payload.get("source_id", "")),
            source_version=str(payload.get("source_version", "")),
            display_name=str(payload.get("display_name", "")),
        )


@dataclass(frozen=True, slots=True)
class AnalysisRuntimeIdentity:
    engine_version: str
    engine_build: str
    python_version: str
    numpy_version: str
    opencv_version: str
    operating_system: str
    partition_algorithm: str = ANALYSIS_PARTITION_ALGORITHM
    seed: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "engine_version",
            "engine_build",
            "python_version",
            "numpy_version",
            "opencv_version",
            "operating_system",
            "partition_algorithm",
        ):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), f"runtime.{field_name}"))

    def to_payload(self) -> dict[str, object]:
        return {
            "engine_version": self.engine_version,
            "engine_build": self.engine_build,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "opencv_version": self.opencv_version,
            "operating_system": self.operating_system,
            "partition_algorithm": self.partition_algorithm,
            "seed": self.seed,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisRuntimeIdentity:
        seed = payload.get("seed")
        return cls(
            engine_version=str(payload.get("engine_version", "")),
            engine_build=str(payload.get("engine_build", "")),
            python_version=str(payload.get("python_version", "")),
            numpy_version=str(payload.get("numpy_version", "")),
            opencv_version=str(payload.get("opencv_version", "")),
            operating_system=str(payload.get("operating_system", "")),
            partition_algorithm=str(payload.get("partition_algorithm", "")),
            seed=None if seed is None else int(seed),
        )


@dataclass(frozen=True, slots=True)
class AnalysisRunManifest:
    run_id: str
    project_id: str
    frame_ids: tuple[str, ...]
    source_bindings: tuple[AnalysisSourceBinding, ...]
    recipe: AnalysisRecipe
    runtime: AnalysisRuntimeIdentity
    parameters: tuple[AnalysisParameter, ...] = ()
    protocol_version: str = ANALYSIS_PROTOCOL_VERSION
    schema: str = ANALYSIS_RUN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _required_text(self.run_id, "run_id"))
        object.__setattr__(self, "project_id", _required_text(self.project_id, "project_id"))
        object.__setattr__(self, "frame_ids", tuple(_required_text(item, "frame_id") for item in self.frame_ids))
        object.__setattr__(self, "source_bindings", tuple(self.source_bindings))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if not self.frame_ids:
            raise ValueError("Analysis run must contain at least one frame")
        if len(self.frame_ids) != len(set(self.frame_ids)):
            raise ValueError("Analysis run frame ids must be unique")
        binding_keys = [binding.binding_key for binding in self.source_bindings]
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError("Analysis source binding keys must be unique")
        self.recipe.validate_bindings(set(binding_keys))
        parameter_keys = [parameter.key for parameter in self.parameters]
        if len(parameter_keys) != len(set(parameter_keys)):
            raise ValueError("Analysis parameter keys must be unique")
        if self.protocol_version != ANALYSIS_PROTOCOL_VERSION or self.schema != ANALYSIS_RUN_SCHEMA:
            raise ValueError("Unsupported analysis run contract")

    @property
    def fingerprint(self) -> str:
        payload = self.to_payload()
        payload.pop("run_id", None)
        return payload_sha256(payload)

    @property
    def partition_count(self) -> int:
        return math.ceil(len(self.frame_ids) / ANALYSIS_PARTITION_SIZE)

    def partition_frame_ids(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            self.frame_ids[offset : offset + ANALYSIS_PARTITION_SIZE]
            for offset in range(0, len(self.frame_ids), ANALYSIS_PARTITION_SIZE)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "project_id": self.project_id,
            "frame_ids": list(self.frame_ids),
            "source_bindings": [binding.to_payload() for binding in self.source_bindings],
            "recipe": self.recipe.to_payload(),
            "runtime": self.runtime.to_payload(),
            "parameters": [parameter.to_payload() for parameter in self.parameters],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisRunManifest:
        raw_frame_ids = payload.get("frame_ids", [])
        if not isinstance(raw_frame_ids, list):
            raise ValueError("frame_ids must be a list")
        raw_bindings = _object_list(payload.get("source_bindings", []), "source_bindings")
        raw_parameters = _object_list(payload.get("parameters", []), "parameters")
        raw_recipe = payload.get("recipe")
        raw_runtime = payload.get("runtime")
        if not isinstance(raw_recipe, Mapping) or not isinstance(raw_runtime, Mapping):
            raise ValueError("recipe and runtime must be objects")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            run_id=str(payload.get("run_id", "")),
            project_id=str(payload.get("project_id", "")),
            frame_ids=tuple(str(frame_id) for frame_id in raw_frame_ids),
            source_bindings=tuple(AnalysisSourceBinding.from_payload(item) for item in raw_bindings),
            recipe=AnalysisRecipe.from_payload(raw_recipe),
            runtime=AnalysisRuntimeIdentity.from_payload(raw_runtime),
            parameters=tuple(AnalysisParameter.from_payload(item) for item in raw_parameters),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> AnalysisRunManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Analysis run manifest must contain an object")
        return cls.from_payload(payload)


@dataclass(frozen=True, slots=True)
class AnalysisPartitionJobManifest:
    job_id: str
    run_id: str
    partition_id: str
    project_id: str
    partition_index: int
    partition_count: int
    run_fingerprint: str
    recipe: AnalysisRecipe
    frames: tuple[AnalysisFrameInput, ...]
    parameters: tuple[AnalysisParameter, ...] = ()
    protocol_version: str = ANALYSIS_PROTOCOL_VERSION
    schema: str = ANALYSIS_PARTITION_JOB_SCHEMA

    def __post_init__(self) -> None:
        for field_name in ("job_id", "run_id", "partition_id", "project_id", "run_fingerprint"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if not 0 <= self.partition_index < self.partition_count:
            raise ValueError("partition_index must be within partition_count")
        if not 1 <= len(self.frames) <= ANALYSIS_PARTITION_SIZE:
            raise ValueError(f"Analysis partition must contain 1..{ANALYSIS_PARTITION_SIZE} frames")
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Analysis partition frame ids must be unique")
        available_keys = {artifact.binding_key for frame in self.frames for artifact in frame.artifacts}
        self.recipe.validate_bindings(available_keys)
        required_keys = self.recipe.expression.source_keys
        for frame in self.frames:
            present = {artifact.binding_key for artifact in frame.artifacts}
            missing = sorted(required_keys - present)
            if missing:
                raise ValueError(f"Frame {frame.frame_id} is missing bindings: {', '.join(missing)}")
        if self.protocol_version != ANALYSIS_PROTOCOL_VERSION or self.schema != ANALYSIS_PARTITION_JOB_SCHEMA:
            raise ValueError("Unsupported analysis partition job contract")

    @property
    def fingerprint(self) -> str:
        payload = self.to_payload()
        payload.pop("job_id", None)
        payload.pop("partition_id", None)
        return payload_sha256(payload)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "partition_id": self.partition_id,
            "project_id": self.project_id,
            "partition_index": self.partition_index,
            "partition_count": self.partition_count,
            "run_fingerprint": self.run_fingerprint,
            "recipe": self.recipe.to_payload(),
            "frames": [frame.to_payload() for frame in self.frames],
            "parameters": [parameter.to_payload() for parameter in self.parameters],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisPartitionJobManifest:
        raw_recipe = payload.get("recipe")
        if not isinstance(raw_recipe, Mapping):
            raise ValueError("recipe must be an object")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            job_id=str(payload.get("job_id", "")),
            run_id=str(payload.get("run_id", "")),
            partition_id=str(payload.get("partition_id", "")),
            project_id=str(payload.get("project_id", "")),
            partition_index=int(payload.get("partition_index", -1)),
            partition_count=int(payload.get("partition_count", 0)),
            run_fingerprint=str(payload.get("run_fingerprint", "")),
            recipe=AnalysisRecipe.from_payload(raw_recipe),
            frames=tuple(
                AnalysisFrameInput.from_payload(item)
                for item in _object_list(payload.get("frames", []), "frames")
            ),
            parameters=tuple(
                AnalysisParameter.from_payload(item)
                for item in _object_list(payload.get("parameters", []), "parameters")
            ),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> AnalysisPartitionJobManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Analysis partition job manifest must contain an object")
        return cls.from_payload(payload)


@dataclass(frozen=True, slots=True)
class AnalysisRecordBundle:
    relative_path: str
    sha256: str
    compressed_size: int
    uncompressed_size: int
    frame_count: int
    media_type: str = "application/x-ndjson"
    compression: str = "gzip"

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", safe_relative_path(self.relative_path))
        digest = _required_text(self.sha256, "bundle.sha256").lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("bundle.sha256 must be a hexadecimal SHA-256 digest")
        object.__setattr__(self, "sha256", digest)
        if min(self.compressed_size, self.uncompressed_size, self.frame_count) < 0:
            raise ValueError("bundle sizes and frame_count must not be negative")
        if self.compression != "gzip":
            raise ValueError("Only gzip analysis record bundles are supported")

    def to_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "compressed_size": self.compressed_size,
            "uncompressed_size": self.uncompressed_size,
            "frame_count": self.frame_count,
            "media_type": self.media_type,
            "compression": self.compression,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisRecordBundle:
        return cls(
            relative_path=str(payload.get("relative_path", "")),
            sha256=str(payload.get("sha256", "")),
            compressed_size=int(payload.get("compressed_size", -1)),
            uncompressed_size=int(payload.get("uncompressed_size", -1)),
            frame_count=int(payload.get("frame_count", -1)),
            media_type=str(payload.get("media_type", "")),
            compression=str(payload.get("compression", "")),
        )


@dataclass(frozen=True, slots=True)
class AnalysisPartitionResultManifest:
    job_id: str
    run_id: str
    partition_id: str
    project_id: str
    outcome: AnalysisOutcome
    bundle: AnalysisRecordBundle | None
    message: str = ""
    protocol_version: str = ANALYSIS_PROTOCOL_VERSION
    schema: str = ANALYSIS_PARTITION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        for field_name in ("job_id", "run_id", "partition_id", "project_id"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        if self.outcome == AnalysisOutcome.SUCCEEDED and self.bundle is None:
            raise ValueError("Succeeded analysis partition requires a record bundle")
        if self.protocol_version != ANALYSIS_PROTOCOL_VERSION or self.schema != ANALYSIS_PARTITION_RESULT_SCHEMA:
            raise ValueError("Unsupported analysis partition result contract")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "partition_id": self.partition_id,
            "project_id": self.project_id,
            "outcome": self.outcome.value,
            "bundle": None if self.bundle is None else self.bundle.to_payload(),
            "message": self.message,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisPartitionResultManifest:
        raw_bundle = payload.get("bundle")
        if raw_bundle is not None and not isinstance(raw_bundle, Mapping):
            raise ValueError("bundle must be an object or null")
        return cls(
            schema=str(payload.get("schema", "")),
            protocol_version=str(payload.get("protocol_version", "")),
            job_id=str(payload.get("job_id", "")),
            run_id=str(payload.get("run_id", "")),
            partition_id=str(payload.get("partition_id", "")),
            project_id=str(payload.get("project_id", "")),
            outcome=AnalysisOutcome(str(payload.get("outcome", ""))),
            bundle=None if raw_bundle is None else AnalysisRecordBundle.from_payload(raw_bundle),
            message=str(payload.get("message", "")),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> AnalysisPartitionResultManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Analysis partition result manifest must contain an object")
        return cls.from_payload(payload)


__all__ = [
    "ANALYSIS_MAX_EXPRESSION_DEPTH",
    "ANALYSIS_PARTITION_ALGORITHM",
    "ANALYSIS_PARTITION_JOB_SCHEMA",
    "ANALYSIS_PARTITION_RESULT_SCHEMA",
    "ANALYSIS_PARTITION_SIZE",
    "ANALYSIS_RUN_SCHEMA",
    "AnalysisExpression",
    "AnalysisPartitionJobManifest",
    "AnalysisPartitionResultManifest",
    "AnalysisRecipe",
    "AnalysisRecordBundle",
    "AnalysisRunManifest",
    "AnalysisRuntimeIdentity",
    "AnalysisSourceBinding",
    "canonical_json",
    "payload_sha256",
]
