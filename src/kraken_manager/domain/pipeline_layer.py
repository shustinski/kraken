"""Typed layer operation in the existing graph; no manufactured frame IDs."""
from collections.abc import Mapping
from dataclasses import dataclass, field

from .common import DomainValidationError, freeze_mapping, require_non_empty, validate_uuid
from .pipeline import PipelineDataKind, PipelineResources, positive


@dataclass(frozen=True, slots=True)
class PipelineLayerContent:
    kind: PipelineDataKind
    representation_id: str | None = None
    producer_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "kind", PipelineDataKind(self.kind))
        if self.kind not in {PipelineDataKind.IMAGE, PipelineDataKind.MASK, PipelineDataKind.CONFIDENCE, PipelineDataKind.VECTOR}:
            raise DomainValidationError("mosaic content requires an image, mask, confidence or vector input port")
        if (self.representation_id is None) == (self.producer_id is None):
            raise DomainValidationError("layer content needs exactly one representation or producer")
        for value in (self.representation_id, self.producer_id):
            if value is not None:
                validate_uuid(value)

    def to_dict(self):
        return {"kind": self.kind.value, **{k: getattr(self, k) for k in ("representation_id", "producer_id") if getattr(self, k) is not None}}


@dataclass(frozen=True, slots=True)
class PipelineLayerOperation:
    id: str
    title: str
    capability: str
    algorithm_version: str
    stitch_id: str
    stitch_revision: int
    dependencies: tuple[str, ...] = ()
    content: PipelineLayerContent | None = None
    parameters: Mapping = field(default_factory=dict)
    resources: PipelineResources = field(default_factory=PipelineResources)
    approval_required: bool = False
    quality_gate: bool = True
    output_kind: PipelineDataKind = PipelineDataKind.MOSAIC
    kind: str = "layer"

    def __post_init__(self):
        for value in (self.id, self.stitch_id):
            validate_uuid(value)
        for name in ("title", "algorithm_version"):
            require_non_empty(getattr(self, name), field=name)
        positive(self.stitch_revision, "stitch_revision", zero=True)
        if self.kind != "layer" or self.capability != "layer.stitch.v1" or self.output_kind != PipelineDataKind.MOSAIC:
            raise DomainValidationError("unsupported layer operation ports")
        object.__setattr__(self, "output_kind", PipelineDataKind.MOSAIC)
        dependencies = tuple(validate_uuid(value) for value in self.dependencies)
        if self.id in dependencies or len(set(dependencies)) != len(dependencies):
            raise DomainValidationError("duplicate or self layer dependency")
        object.__setattr__(self, "dependencies", dependencies)
        if self.content is not None:
            if not isinstance(self.content, PipelineLayerContent):
                raise DomainValidationError("layer content requires a typed binding")
            if self.content.producer_id and self.content.producer_id not in dependencies:
                raise DomainValidationError("layer content producer must be a dependency")
        if not isinstance(self.approval_required, bool) or not isinstance(self.quality_gate, bool):
            raise DomainValidationError("layer gate flags must be boolean")
        if set(self.parameters) - {"mode", "block_size", "max_file_bytes"}:
            raise DomainValidationError("unknown layer operation parameters")
        if self.parameters.get("mode", "rgb") not in {"rgb", "gray", "labels"}:
            raise DomainValidationError("invalid mosaic pixel mode")
        positive(self.parameters.get("block_size", 256), "block_size")
        if self.parameters.get("block_size", 256) > 1024:
            raise DomainValidationError("mosaic block exceeds allocation limit")
        if "max_file_bytes" in self.parameters:
            positive(self.parameters["max_file_bytes"], "max_file_bytes")
            if self.parameters["max_file_bytes"] > 512 * 1024**3:
                raise DomainValidationError("mosaic disk budget exceeds its supported limit")
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters, field="pipeline.layer.parameters"))

    @property
    def input_operation_id(self):
        return None if self.content is None else self.content.producer_id

    @property
    def model_input(self):
        return None

    def to_dict(self):
        def plain(value):
            if isinstance(value, Mapping):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value
        return {"kind": self.kind, "id": self.id, "title": self.title, "capability": self.capability,
            "algorithm_version": self.algorithm_version, "stitch_id": self.stitch_id, "stitch_revision": self.stitch_revision,
            "dependencies": list(self.dependencies), "content": None if self.content is None else self.content.to_dict(),
            "parameters": plain(self.parameters), "resources": self.resources.to_dict(), "approval_required": self.approval_required,
            "quality_gate": self.quality_gate, "output_kind": self.output_kind.value}
