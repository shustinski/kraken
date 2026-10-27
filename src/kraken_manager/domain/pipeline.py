"""Executable definitions for the existing layer-pipeline graph.

Computations remain PluginJobs. This model only defines their dependencies,
typed artifact bindings and scheduling policy; it never manufactures results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Mapping
import hashlib
import json
import math

from .common import DomainValidationError, freeze_mapping, require_non_empty, validate_uuid


class PipelineDataKind(StrEnum):
    IMAGE = "image"
    MASK = "binary_mask"
    VECTOR = "vector"
    CONFIDENCE = "confidence"
    DATASET = "dataset"
    MODEL = "model"
    MOSAIC = "mosaic"


# These are the existing public Agent capabilities, not alternate algorithms.
CAPABILITY_PORTS = {
    "frames.vectorize.v1": ((PipelineDataKind.IMAGE, PipelineDataKind.MASK), PipelineDataKind.VECTOR),
    "frames.binary-segment.v1": ((PipelineDataKind.IMAGE,), PipelineDataKind.MASK),
    "frames.binary-segment.v2": ((PipelineDataKind.IMAGE,), PipelineDataKind.MASK),
    "frames.confidence.v1": ((PipelineDataKind.IMAGE,), PipelineDataKind.CONFIDENCE),
    "frames.dataset.prepare.v1": ((PipelineDataKind.IMAGE,), PipelineDataKind.DATASET),
    "dataset.model.train.v1": ((PipelineDataKind.DATASET,), PipelineDataKind.MODEL),
}


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise DomainValidationError(f"{name} must be an integer >= {0 if zero else 1}")


@dataclass(frozen=True, slots=True)
class PipelineResources:
    cpu: int = 1
    gpu: int = 0
    memory_mb: int = 256

    def __post_init__(self):
        positive(self.cpu, "cpu")
        positive(self.gpu, "gpu", zero=True)
        positive(self.memory_mb, "memory_mb")

    def fits(self, capacity: PipelineResources) -> bool:
        return all(getattr(self, k) <= getattr(capacity, k) for k in ("cpu", "gpu", "memory_mb"))

    def to_dict(self):
        return {k: getattr(self, k) for k in ("cpu", "gpu", "memory_mb")}


@dataclass(frozen=True, slots=True)
class PipelinePolicy:
    max_parallel: int = 1
    capacity: PipelineResources = field(default_factory=PipelineResources)
    max_attempts: int = 1
    retry_delay_seconds: float = 30
    recompute_stale: bool = False

    def __post_init__(self):
        positive(self.max_parallel, "max_parallel")
        positive(self.max_attempts, "max_attempts")
        if (
            isinstance(self.retry_delay_seconds, bool)
            or not isinstance(self.retry_delay_seconds, (int, float))
            or not math.isfinite(self.retry_delay_seconds)
            or self.retry_delay_seconds < 0
        ):
            raise DomainValidationError("retry delay must be finite and nonnegative")
        if not isinstance(self.recompute_stale, bool):
            raise DomainValidationError("recompute_stale must be boolean")

    def to_dict(self):
        return dict(
            max_parallel=self.max_parallel,
            capacity=self.capacity.to_dict(),
            max_attempts=self.max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            recompute_stale=self.recompute_stale,
        )


@dataclass(frozen=True, slots=True)
class PipelineModelInput:
    """Auxiliary immutable model port; frame inputs retain their V1 cardinality."""
    version_id: str | None = None
    producer_id: str | None = None

    def __post_init__(self):
        if (self.version_id is None) == (self.producer_id is None):
            raise DomainValidationError("model input requires exactly one version or producer")
        for value in (self.version_id, self.producer_id):
            if value is not None:
                validate_uuid(value)

    def to_dict(self):
        return {key: value for key in ("version_id", "producer_id") if (value := getattr(self, key)) is not None}


@dataclass(frozen=True, slots=True)
class PipelineOperation:
    id: str
    title: str
    capability: str
    algorithm_version: str
    source_representation_id: str
    target_representation_id: str
    input_kind: PipelineDataKind
    output_kind: PipelineDataKind
    coordinates: tuple[tuple[int, int], ...]
    dependencies: tuple[str, ...] = ()
    input_operation_id: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)
    resources: PipelineResources = field(default_factory=PipelineResources)
    approval_required: bool = False
    quality_gate: bool = True
    model_input: PipelineModelInput | None = None

    def __post_init__(self):
        for name in ("id", "source_representation_id", "target_representation_id"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        for name in ("title", "algorithm_version"):
            require_non_empty(getattr(self, name), field=name)
        object.__setattr__(self, "input_kind", PipelineDataKind(self.input_kind))
        object.__setattr__(self, "output_kind", PipelineDataKind(self.output_kind))
        ports = CAPABILITY_PORTS.get(self.capability)
        if ports is None or self.input_kind not in ports[0] or self.output_kind != ports[1]:
            raise DomainValidationError("incompatible pipeline capability ports")
        coordinates = tuple(tuple(xy) for xy in self.coordinates)
        if not coordinates or len(coordinates) != len(set(coordinates)):
            raise DomainValidationError("pipeline coordinates must be nonempty and unique")
        for xy in coordinates:
            if len(xy) != 2:
                raise DomainValidationError("pipeline coordinate needs x and y")
            for value in xy:
                positive(value, "coordinate", zero=True)
        object.__setattr__(self, "coordinates", coordinates)
        deps = tuple(validate_uuid(value) for value in self.dependencies)
        if len(set(deps)) != len(deps) or self.id in deps:
            raise DomainValidationError("duplicate or self pipeline dependency")
        object.__setattr__(self, "dependencies", deps)
        if self.model_input is not None:
            if not isinstance(self.model_input, PipelineModelInput):
                raise DomainValidationError("model_input requires a typed model binding")
            if self.capability not in {"frames.binary-segment.v2", "frames.confidence.v1"}:
                raise DomainValidationError("this capability has no model input port")
            if self.model_input.producer_id is not None and self.model_input.producer_id not in deps:
                raise DomainValidationError("model producer must be a dependency")
            if set(self.parameters) & {"model_source_path", "model_relative_path", "model_sha256", "model_filename", "model_version_id", "model_version"}:
                raise DomainValidationError("model binding metadata is resolved by Kraken")
        if self.input_operation_id is not None:
            validate_uuid(self.input_operation_id)
            if self.input_operation_id not in deps:
                raise DomainValidationError("input producer must be a dependency")
        if not isinstance(self.approval_required, bool):
            raise DomainValidationError("approval_required must be boolean")
        if not isinstance(self.quality_gate, bool):
            raise DomainValidationError("quality_gate must be boolean")
        if set(self.parameters) & {"pipeline_attempt_id", "pipeline_algorithm_version", "execution_resources"}:
            raise DomainValidationError("pipeline provenance parameters are reserved")
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters, field="pipeline.parameters"))

    def to_dict(self):
        def plain(v):
            if isinstance(v, Mapping):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (tuple, list)):
                return [plain(x) for x in v]
            return v

        return {
            **{
                k: getattr(self, k)
                for k in (
                    "id",
                    "title",
                    "capability",
                    "algorithm_version",
                    "source_representation_id",
                    "target_representation_id",
                    "input_kind",
                    "output_kind",
                    "coordinates",
                    "dependencies",
                    "input_operation_id",
                    "approval_required",
                    "quality_gate",
                )
            },
            "parameters": plain(self.parameters),
            "resources": self.resources.to_dict(),
            **({"model_input": self.model_input.to_dict()} if self.model_input is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class LayerPipelineDefinition:
    project_id: str
    layer_id: str
    operations: tuple[PipelineOperation, ...]
    policy: PipelinePolicy = field(default_factory=PipelinePolicy)
    version: int = 1

    def __post_init__(self):
        validate_uuid(self.project_id)
        validate_uuid(self.layer_id)
        positive(self.version, "pipeline version")
        object.__setattr__(self, "operations", tuple(self.operations))
        by_id = {op.id: op for op in self.operations}
        if not by_id or len(by_id) != len(self.operations):
            raise DomainValidationError("pipeline operations must be nonempty and unique")
        # Multiple writers to the same representation would make latest-version
        # binding ambiguous even when independent branches finish successfully.
        from .pipeline_layer import PipelineLayerOperation
        frames = tuple(op for op in self.operations if not isinstance(op, PipelineLayerOperation))
        if len({op.target_representation_id for op in frames}) != len(frames):
            raise DomainValidationError("pipeline operations need distinct output representations")
        for op in self.operations:
            if not op.resources.fits(self.policy.capacity):
                raise DomainValidationError("operation exceeds pipeline resource capacity")
            if set(op.dependencies) - by_id.keys():
                raise DomainValidationError("unknown pipeline dependency")
            if isinstance(op, PipelineLayerOperation):
                if op.input_operation_id:
                    producer = by_id[op.input_operation_id]
                    if isinstance(producer, PipelineLayerOperation) or producer.output_kind != op.content.kind:
                        raise DomainValidationError("layer content producer has an incompatible output kind")
                continue
            if op.model_input is not None and op.model_input.producer_id is not None:
                producer = by_id[op.model_input.producer_id]
                if producer.output_kind is not PipelineDataKind.MODEL:
                    raise DomainValidationError("model producer must publish a model")
            if op.input_operation_id:
                producer = by_id[op.input_operation_id]
                if (
                    producer.output_kind != op.input_kind
                    or producer.target_representation_id != op.source_representation_id
                    or not set(op.coordinates) <= set(producer.coordinates)
                ):
                    raise DomainValidationError("pipeline edge has incompatible type, representation or scope")
            elif op.source_representation_id in {item.target_representation_id for item in frames}:
                raise DomainValidationError("pipeline internal input must name its producer")
        remaining = {key: set(op.dependencies) for key, op in by_id.items()}
        while remaining:
            ready = {key for key, deps in remaining.items() if not deps}
            if not ready:
                raise DomainValidationError("pipeline dependency cycle")
            remaining = {key: deps - ready for key, deps in remaining.items() if key not in ready}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self):
        return dict(
            project_id=self.project_id,
            layer_id=self.layer_id,
            version=self.version,
            operations=[op.to_dict() for op in self.operations],
            policy=self.policy.to_dict(),
        )

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        policy = dict(data.pop("policy", {}))
        policy["capacity"] = PipelineResources(**policy.get("capacity", {}))
        operations = []
        for value in data.pop("operations"):
            op = dict(value)
            op["resources"] = PipelineResources(**op.get("resources", {}))
            if op.get("kind") == "layer":
                from .pipeline_layer import PipelineLayerContent, PipelineLayerOperation
                if op.get("content") is not None:
                    op["content"] = PipelineLayerContent(**op["content"])
                operations.append(PipelineLayerOperation(**op))
                continue
            if "model_input" in op:
                op["model_input"] = PipelineModelInput(**op["model_input"])
            operations.append(PipelineOperation(**op))
        return cls(**data, operations=tuple(operations), policy=PipelinePolicy(**policy))
