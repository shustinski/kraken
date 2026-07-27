"""Project structure aggregate and spatial value objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid5

from .common import (
    DomainValidationError,
    FrameId,
    LayerId,
    ProjectId,
    RepresentationId,
    as_utc,
    new_uuid,
    require_non_empty,
    utc_now,
    validate_uuid,
)


class GridOrientation(StrEnum):
    """Vertical direction of the project grid; X always grows to the right."""

    Y_DOWN = "y_down"
    Y_UP = "y_up"


class ProjectState(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class LayerType(StrEnum):
    METAL = "metal"
    CONTACT = "contact"
    GATE = "gate"
    DIFFUSION = "diffusion"


class RepresentationKind(StrEnum):
    IMAGE = "image"
    VECTOR = "vector"


class StructureState(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


def _next_revision(current: int, expected_revision: int | None) -> int:
    if expected_revision is not None and expected_revision != current:
        raise DomainValidationError(
            f"expected revision {expected_revision}, but aggregate is at revision {current}"
        )
    return current + 1


@dataclass(frozen=True, slots=True, order=True)
class FrameCoordinate:
    """A one-based coordinate. Bounds are checked by the owning project."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if isinstance(self.x, bool) or not isinstance(self.x, int) or self.x < 1:
            raise DomainValidationError("frame x must be an integer starting at 1")
        if isinstance(self.y, bool) or not isinstance(self.y, int) or self.y < 1:
            raise DomainValidationError("frame y must be an integer starting at 1")

    def frame_id(self, project_id: ProjectId | str) -> FrameId:
        """Return the stable UUIDv5 identifier for this project coordinate."""

        canonical_project_id = validate_uuid(str(project_id), field="project_id")
        return FrameId(str(uuid5(UUID(canonical_project_id), f"frame:{self.x}:{self.y}")))


def deterministic_frame_id(project_id: ProjectId | str, x: int, y: int) -> FrameId:
    """Convenience function for callers that do not yet have a coordinate."""

    return FrameCoordinate(x=x, y=y).frame_id(project_id)


@dataclass(frozen=True, slots=True)
class Project:
    """Project metadata with permanently fixed spatial dimensions.

    The dataclass is frozen and intentionally exposes no resize/reorient method.
    Mutating business operations return a new revision of the value.
    """

    id: ProjectId
    name: str
    width: int
    height: int
    orientation: GridOrientation
    storage_profile: str
    state: ProjectState
    revision: int
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", ProjectId(validate_uuid(str(self.id), field="project.id")))
        object.__setattr__(self, "name", require_non_empty(self.name, field="project.name"))
        object.__setattr__(
            self,
            "storage_profile",
            require_non_empty(self.storage_profile, field="project.storage_profile"),
        )
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise DomainValidationError("project.width must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height < 1:
            raise DomainValidationError("project.height must be a positive integer")
        if not isinstance(self.orientation, GridOrientation):
            object.__setattr__(self, "orientation", GridOrientation(self.orientation))
        if not isinstance(self.state, ProjectState):
            object.__setattr__(self, "state", ProjectState(self.state))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("project.revision must not be negative")
        object.__setattr__(self, "created_at", as_utc(self.created_at, field="project.created_at"))

    @classmethod
    def create(
        cls,
        *,
        name: str,
        width: int,
        height: int,
        orientation: GridOrientation = GridOrientation.Y_DOWN,
        storage_profile: str,
        project_id: ProjectId | str | None = None,
        created_at: datetime | None = None,
    ) -> Project:
        return cls(
            id=ProjectId(str(project_id) if project_id is not None else new_uuid()),
            name=name,
            width=width,
            height=height,
            orientation=orientation,
            storage_profile=storage_profile,
            state=ProjectState.ACTIVE,
            revision=0,
            created_at=created_at or utc_now(),
        )

    @property
    def frame_count(self) -> int:
        return self.width * self.height

    def coordinate(self, x: int, y: int) -> FrameCoordinate:
        coordinate = FrameCoordinate(x=x, y=y)
        if coordinate.x > self.width or coordinate.y > self.height:
            raise DomainValidationError(
                f"frame ({coordinate.x}, {coordinate.y}) is outside {self.width}x{self.height} project grid"
            )
        return coordinate

    def frame_id_at(self, x: int, y: int) -> FrameId:
        return self.coordinate(x, y).frame_id(self.id)

    def rename(self, name: str, *, expected_revision: int | None = None) -> Project:
        return replace(
            self,
            name=require_non_empty(name, field="project.name"),
            revision=_next_revision(self.revision, expected_revision),
        )

    def archive(self, *, expected_revision: int | None = None) -> Project:
        if self.state is ProjectState.ARCHIVED:
            return self
        return replace(
            self,
            state=ProjectState.ARCHIVED,
            revision=_next_revision(self.revision, expected_revision),
        )

    def restore(self, *, expected_revision: int | None = None) -> Project:
        if self.state is ProjectState.ACTIVE:
            return self
        return replace(
            self,
            state=ProjectState.ACTIVE,
            revision=_next_revision(self.revision, expected_revision),
        )


@dataclass(frozen=True, slots=True)
class Layer:
    id: LayerId
    project_id: ProjectId
    name: str
    type: LayerType
    order: int
    state: StructureState
    revision: int
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", LayerId(validate_uuid(str(self.id), field="layer.id")))
        object.__setattr__(
            self, "project_id", ProjectId(validate_uuid(str(self.project_id), field="layer.project_id"))
        )
        object.__setattr__(self, "name", require_non_empty(self.name, field="layer.name"))
        if not isinstance(self.type, LayerType):
            object.__setattr__(self, "type", LayerType(self.type))
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise DomainValidationError("layer.order must be a non-negative integer")
        if not isinstance(self.state, StructureState):
            object.__setattr__(self, "state", StructureState(self.state))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("layer.revision must not be negative")
        object.__setattr__(self, "created_at", as_utc(self.created_at, field="layer.created_at"))

    @classmethod
    def create(
        cls,
        *,
        project_id: ProjectId,
        name: str,
        type: LayerType,
        order: int,
        layer_id: LayerId | str | None = None,
        created_at: datetime | None = None,
    ) -> Layer:
        return cls(
            id=LayerId(str(layer_id) if layer_id is not None else new_uuid()),
            project_id=project_id,
            name=name,
            type=type,
            order=order,
            state=StructureState.ACTIVE,
            revision=0,
            created_at=created_at or utc_now(),
        )

    def rename(self, name: str, *, expected_revision: int | None = None) -> Layer:
        return replace(
            self,
            name=require_non_empty(name, field="layer.name"),
            revision=_next_revision(self.revision, expected_revision),
        )

    def reorder(self, order: int, *, expected_revision: int | None = None) -> Layer:
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise DomainValidationError("layer.order must be a non-negative integer")
        return replace(self, order=order, revision=_next_revision(self.revision, expected_revision))

    def archive(self, *, expected_revision: int | None = None) -> Layer:
        if self.state is StructureState.ARCHIVED:
            return self
        return replace(
            self,
            state=StructureState.ARCHIVED,
            revision=_next_revision(self.revision, expected_revision),
        )


@dataclass(frozen=True, slots=True)
class Representation:
    """A named sparse image or vector representation of a layer."""

    id: RepresentationId
    project_id: ProjectId
    layer_id: LayerId
    name: str
    kind: RepresentationKind
    note: str
    source: str | None
    active: bool
    state: StructureState
    revision: int
    created_at: datetime
    source_image_representation_id: RepresentationId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", RepresentationId(validate_uuid(str(self.id), field="representation.id")))
        object.__setattr__(
            self,
            "project_id",
            ProjectId(validate_uuid(str(self.project_id), field="representation.project_id")),
        )
        object.__setattr__(
            self, "layer_id", LayerId(validate_uuid(str(self.layer_id), field="representation.layer_id"))
        )
        object.__setattr__(self, "name", require_non_empty(self.name, field="representation.name"))
        if not isinstance(self.kind, RepresentationKind):
            object.__setattr__(self, "kind", RepresentationKind(self.kind))
        object.__setattr__(self, "note", self.note.strip())
        if self.source is not None:
            object.__setattr__(self, "source", require_non_empty(self.source, field="representation.source", maximum=2048))
        if self.source_image_representation_id is not None:
            object.__setattr__(
                self,
                "source_image_representation_id",
                RepresentationId(
                    validate_uuid(
                        str(self.source_image_representation_id),
                        field="representation.source_image_representation_id",
                    )
                ),
            )
        if self.kind is RepresentationKind.IMAGE and self.source_image_representation_id is not None:
            raise DomainValidationError("an image representation cannot belong to another image")
        if not isinstance(self.active, bool):
            raise DomainValidationError("representation.active must be boolean")
        if not isinstance(self.state, StructureState):
            object.__setattr__(self, "state", StructureState(self.state))
        if self.state is StructureState.ARCHIVED and self.active:
            raise DomainValidationError("an archived representation cannot be active")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("representation.revision must not be negative")
        object.__setattr__(self, "created_at", as_utc(self.created_at, field="representation.created_at"))

    @classmethod
    def create(
        cls,
        *,
        project_id: ProjectId,
        layer_id: LayerId,
        name: str,
        kind: RepresentationKind,
        note: str = "",
        source: str | None = None,
        source_image_representation_id: RepresentationId | str | None = None,
        active: bool = False,
        representation_id: RepresentationId | str | None = None,
        created_at: datetime | None = None,
    ) -> Representation:
        return cls(
            id=RepresentationId(str(representation_id) if representation_id is not None else new_uuid()),
            project_id=project_id,
            layer_id=layer_id,
            name=name,
            kind=kind,
            note=note,
            source=source,
            source_image_representation_id=(
                None
                if source_image_representation_id is None
                else RepresentationId(str(source_image_representation_id))
            ),
            active=active,
            state=StructureState.ACTIVE,
            revision=0,
            created_at=created_at or utc_now(),
        )

    def rename(self, name: str, *, expected_revision: int | None = None) -> Representation:
        return replace(
            self,
            name=require_non_empty(name, field="representation.name"),
            revision=_next_revision(self.revision, expected_revision),
        )

    def update_note(self, note: str, *, expected_revision: int | None = None) -> Representation:
        return replace(self, note=note.strip(), revision=_next_revision(self.revision, expected_revision))

    def activate(self, *, expected_revision: int | None = None) -> Representation:
        if self.state is StructureState.ARCHIVED:
            raise DomainValidationError("an archived representation cannot be activated")
        if self.active:
            return self
        return replace(self, active=True, revision=_next_revision(self.revision, expected_revision))

    def deactivate(self, *, expected_revision: int | None = None) -> Representation:
        if not self.active:
            return self
        return replace(self, active=False, revision=_next_revision(self.revision, expected_revision))

    def archive(self, *, expected_revision: int | None = None) -> Representation:
        if self.state is StructureState.ARCHIVED:
            return self
        return replace(
            self,
            state=StructureState.ARCHIVED,
            active=False,
            revision=_next_revision(self.revision, expected_revision),
        )


__all__ = [
    "FrameCoordinate",
    "GridOrientation",
    "Layer",
    "LayerType",
    "Project",
    "ProjectState",
    "Representation",
    "RepresentationKind",
    "StructureState",
    "deterministic_frame_id",
]
