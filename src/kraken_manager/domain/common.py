"""Small, framework-free primitives shared by the project-manager domain.

The domain deliberately represents identifiers as strings at its boundary.  The
``NewType`` aliases preserve useful static distinctions without coupling the
model to a database-specific UUID type.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import NewType, TypeAlias
from uuid import UUID, uuid4


ProjectId = NewType("ProjectId", str)
LayerId = NewType("LayerId", str)
RepresentationId = NewType("RepresentationId", str)
FrameId = NewType("FrameId", str)
ArtifactSeriesId = NewType("ArtifactSeriesId", str)
ArtifactVersionId = NewType("ArtifactVersionId", str)
PrincipalId = NewType("PrincipalId", str)
PerformerId = NewType("PerformerId", str)
PluginJobId = NewType("PluginJobId", str)
ReviewBatchId = NewType("ReviewBatchId", str)

JsonPrimitive: TypeAlias = str | int | float | bool | None
FrozenJson: TypeAlias = JsonPrimitive | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


class DomainValidationError(ValueError):
    """Raised when data cannot satisfy a domain invariant."""


class InvalidStateTransition(RuntimeError):
    """Raised when an aggregate state machine rejects a transition."""


def new_uuid() -> str:
    """Return a canonical random UUID string."""

    return str(uuid4())


def validate_uuid(value: str, *, field: str = "id") -> str:
    """Validate and canonicalise a UUID string."""

    try:
        parsed = UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise DomainValidationError(f"{field} must be a valid UUID") from exc
    return str(parsed)


def require_non_empty(value: str, *, field: str, maximum: int = 255) -> str:
    """Trim and validate a human-readable value."""

    normalized = value.strip()
    if not normalized:
        raise DomainValidationError(f"{field} must not be empty")
    if len(normalized) > maximum:
        raise DomainValidationError(f"{field} must be at most {maximum} characters")
    return normalized


def as_utc(value: datetime, *, field: str = "timestamp") -> datetime:
    """Require an aware timestamp and normalise it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainValidationError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def utc_now() -> datetime:
    """Return the current aware UTC time."""

    return datetime.now(timezone.utc)


def freeze_json(value: object, *, field: str = "value") -> FrozenJson:
    """Validate and recursively freeze JSON-compatible metadata.

    Booleans are checked before integers because ``bool`` subclasses ``int``.
    Mapping keys must be strings so the value can be serialized consistently by
    every storage adapter.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise DomainValidationError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJson] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DomainValidationError(f"{field} mapping keys must be strings")
            frozen[key] = freeze_json(item, field=f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, field=field) for item in value)
    raise DomainValidationError(f"{field} must contain JSON-compatible values")


def freeze_mapping(value: Mapping[str, object] | None, *, field: str = "metadata") -> Mapping[str, FrozenJson]:
    """Return an immutable, recursively frozen string-key mapping."""

    frozen = freeze_json({} if value is None else value, field=field)
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "ArtifactSeriesId",
    "ArtifactVersionId",
    "DomainValidationError",
    "FrameId",
    "FrozenJson",
    "InvalidStateTransition",
    "JsonPrimitive",
    "LayerId",
    "PerformerId",
    "PluginJobId",
    "PrincipalId",
    "ProjectId",
    "RepresentationId",
    "ReviewBatchId",
    "as_utc",
    "freeze_json",
    "freeze_mapping",
    "new_uuid",
    "require_non_empty",
    "utc_now",
    "validate_uuid",
]
