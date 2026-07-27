"""Storage-neutral thumbnail cache contracts.

This module deliberately contains no Qt, SQL, filesystem or archive concepts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Protocol


class StoreCapability(Flag):
    BATCH_READ = auto()
    BATCH_WRITE = auto()
    ATOMIC_REPLACE = auto()
    MULTIPROCESS = auto()
    ACCESS_METADATA = auto()
    QUOTA_EVICTION = auto()
    NAMESPACE_DELETE = auto()
    COMPACTION = auto()
    INTEGRITY_CHECK = auto()


class ThumbnailStoreError(RuntimeError):
    pass


class StoreUnavailable(ThumbnailStoreError):
    pass


class StoreReadOnly(ThumbnailStoreError):
    pass


class StoreFull(ThumbnailStoreError):
    pass


class CorruptEntry(ThumbnailStoreError):
    pass


class UnsupportedOperation(ThumbnailStoreError):
    pass


@dataclass(frozen=True, slots=True)
class StoreNamespace:
    plugin: str
    project: str = ""
    dataset: str = ""
    representation: str = ""
    generation: str = ""

    def canonical(self) -> str:
        return json.dumps(
            {
                "dataset": str(self.dataset),
                "generation": str(self.generation),
                "plugin": str(self.plugin),
                "project": str(self.project),
                "representation": str(self.representation),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StorePolicy:
    max_total_bytes: int | None = None
    max_namespace_bytes: int | None = None
    max_entry_bytes: int = 4 * 1024 * 1024
    durable: bool = False
    batch_size: int = 256
    batch_delay_ms: int = 50

    def __post_init__(self) -> None:
        if int(self.max_entry_bytes) <= 0:
            raise ValueError("max_entry_bytes must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class ThumbnailKey:
    source_key: str
    source_revision: str
    lod: int
    width: int
    height: int
    codec: str = "png"
    renderer_fingerprint: str = ""
    device_pixel_ratio: float = 1.0
    fit_mode: str = "cover"
    color_mode: str = "source"

    def canonical(self) -> bytes:
        payload = {
            "codec": str(self.codec).lower(),
            "color_mode": str(self.color_mode),
            "device_pixel_ratio": round(float(self.device_pixel_ratio), 6),
            "fit_mode": str(self.fit_mode),
            "height": int(self.height),
            "lod": int(self.lod),
            "renderer_fingerprint": str(self.renderer_fingerprint),
            "source_key": str(self.source_key),
            "source_revision": str(self.source_revision),
            "width": int(self.width),
        }
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def digest(self) -> bytes:
        return hashlib.sha256(self.canonical()).digest()

    def hex_digest(self) -> str:
        return self.digest().hex()


@dataclass(frozen=True, slots=True)
class ThumbnailRecord:
    key: ThumbnailKey
    payload: bytes
    codec: str = ""
    checksum: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = bytes(self.payload)
        object.__setattr__(self, "payload", raw)
        object.__setattr__(self, "codec", str(self.codec or self.key.codec).lower())
        digest = hashlib.sha256(raw).hexdigest()
        if self.checksum and str(self.checksum).lower() != digest:
            raise CorruptEntry("thumbnail checksum does not match payload")
        object.__setattr__(self, "checksum", digest)


@dataclass(frozen=True, slots=True)
class InvalidationSelector:
    source_key: str | None = None
    source_revision: str | None = None
    renderer_fingerprint: str | None = None

    def matches(self, key: ThumbnailKey) -> bool:
        return (
            (self.source_key is None or key.source_key == self.source_key)
            and (self.source_revision is None or key.source_revision == self.source_revision)
            and (
                self.renderer_fingerprint is None
                or key.renderer_fingerprint == self.renderer_fingerprint
            )
        )


@dataclass(frozen=True, slots=True)
class StoreStats:
    entries: int = 0
    bytes: int = 0
    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0
    maintenance_state: str = "idle"


class ThumbnailStore(Protocol):
    @property
    def capabilities(self) -> StoreCapability: ...

    def open(self, namespace: StoreNamespace, policy: StorePolicy | None = None) -> None: ...

    def get(self, key: ThumbnailKey) -> ThumbnailRecord | None: ...

    def get_many(self, keys: Iterable[ThumbnailKey]) -> Mapping[ThumbnailKey, ThumbnailRecord]: ...

    def put(self, record: ThumbnailRecord) -> None: ...

    def put_many(self, records: Iterable[ThumbnailRecord]) -> None: ...

    def delete(self, key: ThumbnailKey) -> bool: ...

    def invalidate(self, selector: InvalidationSelector) -> int: ...

    def clear_namespace(self) -> None: ...

    def stats(self) -> StoreStats: ...

    def compact(self) -> None: ...

    def close(self) -> None: ...


__all__ = [
    "CorruptEntry",
    "InvalidationSelector",
    "StoreCapability",
    "StoreFull",
    "StoreNamespace",
    "StorePolicy",
    "StoreReadOnly",
    "StoreStats",
    "StoreUnavailable",
    "ThumbnailKey",
    "ThumbnailRecord",
    "ThumbnailStore",
    "ThumbnailStoreError",
    "UnsupportedOperation",
]
