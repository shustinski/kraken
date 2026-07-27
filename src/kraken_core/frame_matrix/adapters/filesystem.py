"""Explicit file-per-thumbnail adapter.

This adapter intentionally favours inspectability and compatibility over the
inode efficiency of container stores.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..storage import (
    CorruptEntry,
    InvalidationSelector,
    StoreCapability,
    StoreFull,
    StoreNamespace,
    StorePolicy,
    StoreStats,
    StoreUnavailable,
    ThumbnailKey,
    ThumbnailRecord,
)

_CHECKSUM_TRAILER_MAGIC = b"KRAKTHM1"


def _key_payload(key: ThumbnailKey) -> dict[str, Any]:
    return {
        "source_key": key.source_key,
        "source_revision": key.source_revision,
        "lod": key.lod,
        "width": key.width,
        "height": key.height,
        "codec": key.codec,
        "renderer_fingerprint": key.renderer_fingerprint,
        "device_pixel_ratio": key.device_pixel_ratio,
        "fit_mode": key.fit_mode,
        "color_mode": key.color_mode,
    }


def _key_from_payload(payload: Mapping[str, Any]) -> ThumbnailKey:
    return ThumbnailKey(
        source_key=str(payload["source_key"]),
        source_revision=str(payload["source_revision"]),
        lod=int(payload["lod"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        codec=str(payload.get("codec", "png")),
        renderer_fingerprint=str(payload.get("renderer_fingerprint", "")),
        device_pixel_ratio=float(payload.get("device_pixel_ratio", 1.0)),
        fit_mode=str(payload.get("fit_mode", "cover")),
        color_mode=str(payload.get("color_mode", "source")),
    )


class FilesystemThumbnailStore:
    capabilities = (
        StoreCapability.BATCH_READ
        | StoreCapability.BATCH_WRITE
        | StoreCapability.ATOMIC_REPLACE
        | StoreCapability.MULTIPROCESS
        | StoreCapability.ACCESS_METADATA
        | StoreCapability.NAMESPACE_DELETE
        | StoreCapability.COMPACTION
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._namespace: StoreNamespace | None = None
        self._namespace_root: Path | None = None
        self._policy = StorePolicy()
        self._journal_path: Path | None = None
        self._index: dict[str, tuple[ThumbnailKey, str, int]] | None = None
        self._lock = threading.RLock()
        self._hits = self._misses = self._writes = self._errors = 0

    def open(self, namespace: StoreNamespace, policy: StorePolicy | None = None) -> None:
        self._namespace = namespace
        self._policy = policy or StorePolicy()
        self.root.mkdir(parents=True, exist_ok=True)
        namespace_root = (self.root / namespace.digest()).resolve()
        if not namespace_root.is_relative_to(self.root):
            raise StoreUnavailable("thumbnail namespace escapes store root")
        namespace_root.mkdir(parents=True, exist_ok=True)
        self._namespace_root = namespace_root
        self._journal_path = namespace_root / "namespace.journal"
        descriptor = namespace_root / "namespace.json"
        if not descriptor.exists():
            self._atomic_write(descriptor, namespace.canonical().encode("utf-8"))
        self._index = None

    def _require_open(self) -> Path:
        if self._namespace_root is None:
            raise StoreUnavailable("thumbnail store is not open")
        return self._namespace_root

    @staticmethod
    def _safe_extension(codec: str) -> str:
        normalized = "".join(character for character in str(codec).lower() if character.isalnum())
        return normalized[:12] or "bin"

    def _relative_path(self, key: ThumbnailKey) -> str:
        digest = key.hex_digest()
        return f"{digest[:2]}/{digest[2:4]}/{digest}.{self._safe_extension(key.codec)}"

    def _path(self, key: ThumbnailKey) -> Path:
        root = self._require_open()
        path = (root / self._relative_path(key)).resolve()
        if not path.is_relative_to(root):
            raise StoreUnavailable("thumbnail path escapes namespace root")
        return path

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def _append_journal(self, payload: Mapping[str, Any]) -> None:
        path = self._journal_path
        if path is None:
            raise StoreUnavailable("thumbnail store is not open")
        line = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)

    def _ensure_index(self) -> dict[str, tuple[ThumbnailKey, str, int]]:
        if self._index is not None:
            return self._index
        index: dict[str, tuple[ThumbnailKey, str, int]] = {}
        path = self._journal_path
        if path is not None and path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            event = json.loads(line)
                            digest = str(event["digest"])
                            if event.get("operation") == "delete":
                                index.pop(digest, None)
                            elif event.get("operation") == "put":
                                key = _key_from_payload(event["key"])
                                index[digest] = (key, str(event["path"]), int(event.get("bytes", 0)))
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                            continue
            except OSError:
                pass
        self._index = index
        return index

    def get(self, key: ThumbnailKey) -> ThumbnailRecord | None:
        path = self._path(key)
        try:
            stored_payload = path.read_bytes()
        except FileNotFoundError:
            self._misses += 1
            return None
        except OSError as exc:
            self._errors += 1
            raise StoreUnavailable(str(exc)) from exc
        payload = stored_payload
        if stored_payload.endswith(_CHECKSUM_TRAILER_MAGIC) and len(stored_payload) >= 40:
            payload = stored_payload[:-40]
            expected_checksum = stored_payload[-40:-8].hex()
        else:
            expected_checksum = ""
        try:
            record = ThumbnailRecord(key=key, payload=payload, codec=key.codec)
        except CorruptEntry:
            self._errors += 1
            raise
        if expected_checksum and record.checksum != expected_checksum:
            self._errors += 1
            raise CorruptEntry("filesystem thumbnail checksum does not match payload")
        self._hits += 1
        return record

    def get_many(self, keys: Iterable[ThumbnailKey]) -> Mapping[ThumbnailKey, ThumbnailRecord]:
        return {key: record for key in keys if (record := self.get(key)) is not None}

    def put(self, record: ThumbnailRecord) -> None:
        self.put_many((record,))

    def put_many(self, records: Iterable[ThumbnailRecord]) -> None:
        with self._lock:
            for record in records:
                if len(record.payload) > self._policy.max_entry_bytes:
                    self._errors += 1
                    raise StoreFull("thumbnail exceeds max_entry_bytes")
                path = self._path(record.key)
                try:
                    stored_payload = record.payload + bytes.fromhex(record.checksum) + _CHECKSUM_TRAILER_MAGIC
                    self._atomic_write(path, stored_payload)
                    event = {
                        "bytes": len(record.payload),
                        "checksum": record.checksum,
                        "digest": record.key.hex_digest(),
                        "key": _key_payload(record.key),
                        "operation": "put",
                        "path": self._relative_path(record.key),
                    }
                    self._append_journal(event)
                except OSError as exc:
                    self._errors += 1
                    if getattr(exc, "winerror", None) == 112 or getattr(exc, "errno", None) == 28:
                        raise StoreFull(str(exc)) from exc
                    raise StoreUnavailable(str(exc)) from exc
                if self._index is not None:
                    self._index[record.key.hex_digest()] = (
                        record.key,
                        self._relative_path(record.key),
                        len(record.payload),
                    )
                self._writes += 1

    def delete(self, key: ThumbnailKey) -> bool:
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            self._errors += 1
            raise StoreUnavailable(str(exc)) from exc
        with self._lock:
            self._append_journal({"digest": key.hex_digest(), "operation": "delete"})
            if self._index is not None:
                self._index.pop(key.hex_digest(), None)
        for parent in (path.parent, path.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break
        return True

    def invalidate(self, selector: InvalidationSelector) -> int:
        keys = [key for key, _path, _size in self._ensure_index().values() if selector.matches(key)]
        return sum(1 for key in keys if self.delete(key))

    def clear_namespace(self) -> None:
        root = self._require_open()
        if root == self.root or not root.is_relative_to(self.root):
            raise StoreUnavailable("refusing to clear an unsafe thumbnail namespace")
        try:
            shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._errors += 1
            raise StoreUnavailable(str(exc)) from exc
        self._journal_path = root / "namespace.journal"
        if self._namespace is not None:
            self._atomic_write(root / "namespace.json", self._namespace.canonical().encode("utf-8"))
        self._index = {}

    def stats(self) -> StoreStats:
        index = self._ensure_index()
        return StoreStats(
            entries=len(index),
            bytes=sum(size for _key, _path, size in index.values()),
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            errors=self._errors,
        )

    def compact(self) -> None:
        with self._lock:
            index = self._ensure_index()
            path = self._journal_path
            if path is None:
                raise StoreUnavailable("thumbnail store is not open")
            lines = [
                json.dumps(
                    {
                        "bytes": size,
                        "digest": digest,
                        "key": _key_payload(key),
                        "operation": "put",
                        "path": relative_path,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for digest, (key, relative_path, size) in sorted(index.items())
            ]
            self._atomic_write(path, (("\n".join(lines) + "\n") if lines else "").encode("utf-8"))

    def close(self) -> None:
        self._index = None
        self._namespace = None
        self._namespace_root = None
        self._journal_path = None


__all__ = ["FilesystemThumbnailStore"]
