from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from kraken_manager.infrastructure.blob import BlobRef, FilesystemBlobStore
from kraken_manager.infrastructure.filesystem._atomic import atomic_write_json, fsync_directory
from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore

from .safe_paths import UnsafeBundlePath, iter_regular_files, portable_relative, safe_join, validate_bundle_path


BUNDLE_SCHEMA = "kraken-migration-bundle/v1"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_NAME = ".checkpoint.json"
EVENTS_PATH = "events/events.jsonl"


class MigrationBundleError(RuntimeError):
    pass


class MigrationVerificationError(MigrationBundleError):
    pass


@dataclass(frozen=True, slots=True)
class BundleEntry:
    path: str
    sha256: str
    size: int
    kind: str

    def __post_init__(self) -> None:
        validate_bundle_path(self.path)
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError(f"invalid SHA-256 for bundle entry {self.path!r}")
        if self.size < 0:
            raise ValueError("bundle entry size may not be negative")
        if self.kind not in {"events", "blob", "snapshot"}:
            raise ValueError(f"unsupported bundle entry kind {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size, "kind": self.kind}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BundleEntry:
        return cls(
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            size=int(value["size"]),
            kind=str(value["kind"]),
        )


@dataclass(frozen=True, slots=True)
class KrakenMigrationBundleV1:
    bundle_id: str
    project_id: str
    created_at: str
    source_profile: str
    event_count: int
    last_global_position: int
    entries: tuple[BundleEntry, ...]
    project_descriptor: Mapping[str, Any]
    external_references: tuple[Mapping[str, Any], ...] = ()
    schema: str = BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported migration bundle schema {self.schema!r}")
        try:
            uuid.UUID(self.bundle_id)
        except ValueError as exc:
            raise ValueError("bundle_id must be a UUID") from exc
        if self.event_count < 0 or self.last_global_position < 0:
            raise ValueError("event counters may not be negative")
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("bundle manifest contains duplicate paths")
        event_entries = [entry for entry in self.entries if entry.kind == "events"]
        if len(event_entries) != 1 or event_entries[0].path != EVENTS_PATH:
            raise ValueError(f"bundle must contain exactly one {EVENTS_PATH!r} entry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bundle_id": self.bundle_id,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "source_profile": self.source_profile,
            "event_count": self.event_count,
            "last_global_position": self.last_global_position,
            "project_descriptor": dict(self.project_descriptor),
            "external_references": [dict(reference) for reference in self.external_references],
            "entries": [entry.to_dict() for entry in sorted(self.entries, key=lambda item: item.path)],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KrakenMigrationBundleV1:
        entries = value.get("entries")
        descriptor = value.get("project_descriptor")
        references = value.get("external_references", [])
        if not isinstance(entries, list) or not isinstance(descriptor, dict) or not isinstance(references, list):
            raise ValueError("invalid migration bundle manifest")
        return cls(
            schema=str(value.get("schema", "")),
            bundle_id=str(value["bundle_id"]),
            project_id=str(value["project_id"]),
            created_at=str(value["created_at"]),
            source_profile=str(value["source_profile"]),
            event_count=int(value["event_count"]),
            last_global_position=int(value["last_global_position"]),
            project_descriptor=descriptor,
            external_references=tuple(dict(reference) for reference in references),
            entries=tuple(BundleEntry.from_dict(entry) for entry in entries),
        )


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    project_id: str
    destination: Path
    event_count: int
    blob_count: int
    snapshot_count: int
    total_bytes: int
    external_reference_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationReport:
    valid: bool
    checked_files: int
    checked_bytes: int
    event_count: int
    errors: tuple[str, ...] = ()

    def raise_for_errors(self) -> None:
        if not self.valid:
            raise MigrationVerificationError("; ".join(self.errors))


@dataclass(frozen=True, slots=True)
class ImportResult:
    bundle_id: str
    project_id: str
    blobs_imported: int
    events_imported: int
    snapshots_imported: int
    resumed: bool


def _hash_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _canonical_record(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _write_stream_atomic(
    target: Path,
    chunks: Iterable[bytes],
    *,
    expected_digest: str | None = None,
    expected_size: int | None = None,
) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    hasher = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as stream:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("migration streams must yield bytes")
                stream.write(chunk)
                hasher.update(chunk)
                size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hasher.hexdigest()
        if expected_digest is not None and digest != expected_digest:
            raise MigrationVerificationError(
                f"copied file {target.name!r} digest mismatch: expected {expected_digest}, found {digest}"
            )
        if expected_size is not None and size != expected_size:
            raise MigrationVerificationError(
                f"copied file {target.name!r} size mismatch: expected {expected_size}, found {size}"
            )
        os.replace(temporary, target)
        fsync_directory(target.parent)
        return digest, size
    finally:
        temporary.unlink(missing_ok=True)


def _file_chunks(stream: BinaryIO) -> Iterator[bytes]:
    while chunk := stream.read(1024 * 1024):
        yield chunk


def load_bundle_manifest(bundle_root: str | Path) -> KrakenMigrationBundleV1:
    root = Path(bundle_root).resolve()
    path = safe_join(root, MANIFEST_NAME)
    if path.is_symlink():
        raise MigrationBundleError("bundle manifest may not be a symbolic link")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationBundleError(f"cannot read migration manifest at {path}") from exc
    if not isinstance(value, dict):
        raise MigrationBundleError("migration manifest must be a JSON object")
    try:
        return KrakenMigrationBundleV1.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationBundleError("invalid migration manifest") from exc


class CanonicalBundleExporter:
    def __init__(
        self,
        event_store: FilesystemEventStore,
        blob_store: FilesystemBlobStore,
        *,
        source_profile: str = "local-filesystem",
    ) -> None:
        self.event_store = event_store
        self.blob_store = blob_store
        self.source_profile = source_profile

    def plan(
        self,
        destination: str | Path,
        *,
        external_references: Iterable[Mapping[str, Any]] = (),
    ) -> MigrationPlan:
        events = list(self.event_store.iter_project())
        blobs = list(self.blob_store.iter_refs())
        snapshots = list(iter_regular_files(self.event_store.layout.snapshots_dir))
        references = tuple(external_references)
        warnings = (
            ("External references are recorded but their target bytes are not copied.",) if references else ()
        )
        event_size = sum(len(_canonical_record(event.to_dict(include_storage_metadata=True))) for event in events)
        return MigrationPlan(
            project_id=self.event_store.project_id,
            destination=Path(destination).resolve(),
            event_count=len(events),
            blob_count=len(blobs),
            snapshot_count=len(snapshots),
            total_bytes=event_size + sum(blob.size_bytes for blob in blobs) + sum(path.stat().st_size for path in snapshots),
            external_reference_count=len(references),
            warnings=warnings,
        )

    def export(
        self,
        destination: str | Path,
        *,
        external_references: Iterable[Mapping[str, Any]] = (),
        resume: bool = True,
    ) -> KrakenMigrationBundleV1:
        root = Path(destination).resolve()
        manifest_path = root / MANIFEST_NAME
        if manifest_path.exists():
            manifest = load_bundle_manifest(root)
            if manifest.project_id != self.event_store.project_id:
                raise MigrationBundleError("existing bundle belongs to another project")
            BundleVerifier().verify(root).raise_for_errors()
            return manifest

        root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = root / CHECKPOINT_NAME
        checkpoint = self._load_or_create_checkpoint(checkpoint_path, resume=resume)
        references = tuple(dict(reference) for reference in external_references)

        with self.event_store.lock.hold(self.event_store.lock_timeout):
            entries: list[BundleEntry] = []
            events = list(self.event_store.iter_project())
            events_target = safe_join(root, EVENTS_PATH)
            event_digest, event_size = _write_stream_atomic(
                events_target,
                (
                    _canonical_record(event.to_dict(include_storage_metadata=True))
                    for event in events
                ),
            )
            entries.append(BundleEntry(EVENTS_PATH, event_digest, event_size, "events"))
            self._record_checkpoint(checkpoint_path, checkpoint, EVENTS_PATH, event_digest, event_size)

            for reference in self.blob_store.iter_refs():
                relative = (
                    f"objects/sha256/{reference.sha256[:2]}/{reference.sha256[2:4]}/{reference.sha256}"
                )
                target = safe_join(root, relative)
                if self._matches(target, reference.sha256, reference.size_bytes):
                    digest, size = reference.sha256, reference.size_bytes
                else:
                    with self.blob_store.open(reference) as stream:
                        digest, size = _write_stream_atomic(
                            target,
                            _file_chunks(stream),
                            expected_digest=reference.sha256,
                            expected_size=reference.size_bytes,
                        )
                entries.append(BundleEntry(relative, digest, size, "blob"))
                self._record_checkpoint(checkpoint_path, checkpoint, relative, digest, size)

            snapshots_root = self.event_store.layout.snapshots_dir
            for source in iter_regular_files(snapshots_root):
                relative_snapshot = portable_relative(source, snapshots_root)
                relative = f"snapshots/{relative_snapshot}"
                target = safe_join(root, relative)
                source_digest, source_size = _hash_file(source)
                if not self._matches(target, source_digest, source_size):
                    with source.open("rb") as stream:
                        _write_stream_atomic(
                            target,
                            _file_chunks(stream),
                            expected_digest=source_digest,
                            expected_size=source_size,
                        )
                entries.append(BundleEntry(relative, source_digest, source_size, "snapshot"))
                self._record_checkpoint(checkpoint_path, checkpoint, relative, source_digest, source_size)

            descriptor = (
                self.event_store.layout.read_descriptor()
                if self.event_store.layout.descriptor_path.exists()
                else {"schema_version": 1, "project_id": self.event_store.project_id}
            )
            manifest = KrakenMigrationBundleV1(
                bundle_id=str(checkpoint["bundle_id"]),
                project_id=self.event_store.project_id,
                created_at=str(checkpoint["created_at"]),
                source_profile=self.source_profile,
                event_count=len(events),
                last_global_position=events[-1].global_position if events else 0,
                entries=tuple(entries),
                project_descriptor=descriptor,
                external_references=references,
            )
            atomic_write_json(manifest_path, manifest.to_dict(), overwrite=False)
            checkpoint_path.unlink(missing_ok=True)
            fsync_directory(root)
            return manifest

    def _load_or_create_checkpoint(self, path: Path, *, resume: bool) -> dict[str, Any]:
        if path.exists():
            if not resume:
                raise MigrationBundleError(f"incomplete bundle already exists at {path.parent}")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MigrationBundleError("invalid migration export checkpoint") from exc
            if value.get("project_id") != self.event_store.project_id:
                raise MigrationBundleError("migration checkpoint belongs to another project")
            return value

        unexpected = [item for item in path.parent.iterdir() if item.name not in {CHECKPOINT_NAME}]
        if unexpected:
            raise MigrationBundleError(f"destination is not empty: {path.parent}")
        value = {
            "schema": "kraken-migration-export-checkpoint/v1",
            "bundle_id": str(uuid.uuid4()),
            "project_id": self.event_store.project_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed": {},
        }
        atomic_write_json(path, value)
        return value

    @staticmethod
    def _record_checkpoint(
        path: Path,
        checkpoint: dict[str, Any],
        relative: str,
        digest: str,
        size: int,
    ) -> None:
        checkpoint.setdefault("completed", {})[relative] = {"sha256": digest, "size": size}
        atomic_write_json(path, checkpoint)

    @staticmethod
    def _matches(path: Path, digest: str, size: int) -> bool:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            return False
        actual_digest, actual_size = _hash_file(path)
        return actual_digest == digest and actual_size == size


class BundleVerifier:
    def verify(self, bundle_root: str | Path) -> VerificationReport:
        root = Path(bundle_root).resolve()
        errors: list[str] = []
        checked_files = 0
        checked_bytes = 0
        event_count = 0
        try:
            manifest = load_bundle_manifest(root)
        except (MigrationBundleError, UnsafeBundlePath) as exc:
            return VerificationReport(False, 0, 0, 0, (str(exc),))

        entries_by_path = {entry.path: entry for entry in manifest.entries}
        for entry in manifest.entries:
            try:
                path = safe_join(root, entry.path)
            except UnsafeBundlePath as exc:
                errors.append(str(exc))
                continue
            if path.is_symlink() or not path.is_file():
                errors.append(f"missing regular file: {entry.path}")
                continue
            try:
                digest, size = _hash_file(path)
            except OSError as exc:
                errors.append(f"cannot read {entry.path}: {exc}")
                continue
            checked_files += 1
            checked_bytes += size
            if size != entry.size:
                errors.append(f"size mismatch for {entry.path}: expected {entry.size}, found {size}")
            if digest != entry.sha256:
                errors.append(f"digest mismatch for {entry.path}")
            if entry.kind == "blob":
                name = Path(entry.path).name
                if name != entry.sha256:
                    errors.append(f"blob path does not match digest: {entry.path}")

        events_entry = entries_by_path.get(EVENTS_PATH)
        if events_entry is not None:
            try:
                event_count, last_position = self._verify_events(
                    safe_join(root, EVENTS_PATH), manifest.project_id
                )
                if event_count != manifest.event_count:
                    errors.append(
                        f"event count mismatch: manifest {manifest.event_count}, file {event_count}"
                    )
                if last_position != manifest.last_global_position:
                    errors.append(
                        "last global event position does not match manifest: "
                        f"{manifest.last_global_position} != {last_position}"
                    )
            except MigrationVerificationError as exc:
                errors.append(str(exc))

        expected_files = set(entries_by_path) | {MANIFEST_NAME}
        for path in iter_regular_files(root):
            relative = portable_relative(path, root)
            if relative not in expected_files and relative != CHECKPOINT_NAME:
                errors.append(f"unexpected file in bundle: {relative}")

        return VerificationReport(
            valid=not errors,
            checked_files=checked_files,
            checked_bytes=checked_bytes,
            event_count=event_count,
            errors=tuple(errors),
        )

    @staticmethod
    def _verify_events(path: Path, project_id: str) -> tuple[int, int]:
        count = 0
        expected_position = 1
        stream_revisions: dict[str, int] = {}
        event_ids: set[str] = set()
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith("\n"):
                        raise MigrationVerificationError(f"truncated event bundle record at line {line_number}")
                    try:
                        record = json.loads(line)
                        storage = record["_storage"]
                        position = int(storage["global_position"])
                        record_project = str(record["project_id"])
                        stream_id = str(record["stream_id"])
                        revision = int(record["revision"])
                        event_id = str(record["event_id"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        raise MigrationVerificationError(f"invalid event envelope at line {line_number}") from exc
                    if position != expected_position:
                        raise MigrationVerificationError(
                            f"event position gap at line {line_number}: expected {expected_position}, found {position}"
                        )
                    if record_project != project_id:
                        raise MigrationVerificationError(f"event at line {line_number} belongs to another project")
                    expected_revision = stream_revisions.get(stream_id, 0) + 1
                    if revision != expected_revision:
                        raise MigrationVerificationError(
                            f"stream revision gap for {stream_id!r} at line {line_number}"
                        )
                    if event_id in event_ids:
                        raise MigrationVerificationError(f"duplicate event_id at line {line_number}: {event_id}")
                    event_ids.add(event_id)
                    stream_revisions[stream_id] = revision
                    count += 1
                    expected_position += 1
        except (OSError, UnicodeDecodeError) as exc:
            raise MigrationVerificationError(f"cannot read bundled events: {exc}") from exc
        return count, expected_position - 1


class CanonicalBundleImporter:
    def __init__(self, event_store: FilesystemEventStore, blob_store: FilesystemBlobStore) -> None:
        self.event_store = event_store
        self.blob_store = blob_store

    def import_bundle(
        self,
        bundle_root: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        verify_first: bool = True,
    ) -> ImportResult:
        root = Path(bundle_root).resolve()
        manifest = load_bundle_manifest(root)
        if manifest.project_id != self.event_store.project_id:
            raise MigrationBundleError(
                f"target project {self.event_store.project_id!r} does not match bundle {manifest.project_id!r}"
            )
        if verify_first:
            BundleVerifier().verify(root).raise_for_errors()

        checkpoint_file = (
            Path(checkpoint_path).resolve()
            if checkpoint_path is not None
            else self.event_store.layout.staging_dir / f"import-{manifest.bundle_id}.json"
        )
        checkpoint, resumed = self._load_checkpoint(checkpoint_file, manifest)
        completed_blobs: set[str] = set(checkpoint.get("blobs", []))
        blobs_imported = 0
        events_imported = 0
        snapshots_imported = 0

        descriptor_path = self.event_store.layout.descriptor_path
        if descriptor_path.exists():
            existing = self.event_store.layout.read_descriptor()
            if dict(existing) != dict(manifest.project_descriptor):
                raise MigrationBundleError("target project descriptor differs from migration source")
        else:
            self.event_store.layout.initialize(manifest.project_descriptor)

        for entry in manifest.entries:
            if entry.kind != "blob":
                continue
            source = safe_join(root, entry.path)
            if entry.sha256 in completed_blobs and self.blob_store.exists(entry.sha256):
                self.blob_store.verify(BlobRef(entry.sha256, entry.size))
                continue
            with source.open("rb") as stream:
                stored = self.blob_store.put(stream, expected_sha256=entry.sha256)
            if stored.blob.size_bytes != entry.size:
                raise MigrationVerificationError(f"imported blob size mismatch: {entry.sha256}")
            if not stored.already_existed:
                blobs_imported += 1
            completed_blobs.add(entry.sha256)
            checkpoint["blobs"] = sorted(completed_blobs)
            self._save_checkpoint(checkpoint_file, checkpoint)

        source_events = self._iter_source_events(safe_join(root, EVENTS_PATH))
        target_events = iter(self.event_store.iter_project())
        existing_count = 0
        for target in target_events:
            try:
                source = next(source_events)
            except StopIteration as exc:
                raise MigrationBundleError("target event log is longer than migration source") from exc
            if target.to_dict(include_storage_metadata=True) != source:
                raise MigrationBundleError(
                    f"target event log diverges from migration source at position {target.global_position}"
                )
            existing_count += 1

        while chunk := list(itertools.islice(source_events, 1000)):
            self.event_store.append_preserved(chunk)
            events_imported += len(chunk)
            checkpoint["events_imported"] = existing_count + events_imported
            self._save_checkpoint(checkpoint_file, checkpoint)
        self._save_checkpoint(checkpoint_file, checkpoint)

        for entry in manifest.entries:
            if entry.kind != "snapshot":
                continue
            relative = entry.path.removeprefix("snapshots/")
            if relative == entry.path:
                raise MigrationBundleError(f"invalid snapshot bundle path: {entry.path}")
            target = safe_join(self.event_store.layout.snapshots_dir, relative)
            source = safe_join(root, entry.path)
            if target.exists():
                digest, size = _hash_file(target)
                if (digest, size) != (entry.sha256, entry.size):
                    raise MigrationBundleError(f"target snapshot differs: {relative}")
                continue
            with source.open("rb") as stream:
                _write_stream_atomic(
                    target,
                    _file_chunks(stream),
                    expected_digest=entry.sha256,
                    expected_size=entry.size,
                )
            snapshots_imported += 1

        self.verify_import(root).raise_for_errors()
        checkpoint_file.unlink(missing_ok=True)
        return ImportResult(
            bundle_id=manifest.bundle_id,
            project_id=manifest.project_id,
            blobs_imported=blobs_imported,
            events_imported=events_imported,
            snapshots_imported=snapshots_imported,
            resumed=resumed,
        )

    def verify_import(self, bundle_root: str | Path) -> VerificationReport:
        root = Path(bundle_root).resolve()
        manifest = load_bundle_manifest(root)
        errors: list[str] = []
        checked_files = 0
        checked_bytes = 0

        source_events = self._iter_source_events(safe_join(root, EVENTS_PATH))
        target_events = self.event_store.iter_project()
        event_count = 0
        while True:
            try:
                source = next(source_events)
            except StopIteration:
                source = None
            try:
                target = next(target_events)
            except StopIteration:
                target = None
            if source is None and target is None:
                break
            if source is None or target is None:
                errors.append("source and target event counts differ")
                break
            event_count += 1
            if source != target.to_dict(include_storage_metadata=True):
                errors.append(f"event mismatch at global position {event_count}")
                break

        for entry in manifest.entries:
            if entry.kind != "blob":
                continue
            try:
                reference = self.blob_store.verify(BlobRef(entry.sha256, entry.size))
            except Exception as exc:
                errors.append(f"blob {entry.sha256} failed target verification: {exc}")
                continue
            checked_files += 1
            checked_bytes += reference.size_bytes

        return VerificationReport(
            valid=not errors,
            checked_files=checked_files,
            checked_bytes=checked_bytes,
            event_count=event_count,
            errors=tuple(errors),
        )

    @staticmethod
    def _iter_source_events(path: Path) -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line in stream:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MigrationVerificationError("bundled event is not a JSON object")
                yield value

    @staticmethod
    def _load_checkpoint(
        path: Path,
        manifest: KrakenMigrationBundleV1,
    ) -> tuple[dict[str, Any], bool]:
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MigrationBundleError("invalid migration import checkpoint") from exc
            if value.get("bundle_id") != manifest.bundle_id or value.get("project_id") != manifest.project_id:
                raise MigrationBundleError("import checkpoint belongs to another migration")
            return value, True
        return (
            {
                "schema": "kraken-migration-import-checkpoint/v1",
                "bundle_id": manifest.bundle_id,
                "project_id": manifest.project_id,
                "blobs": [],
                "events_imported": 0,
            },
            False,
        )

    @staticmethod
    def _save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
        atomic_write_json(path, checkpoint)


FilesystemSnapshotExporter = CanonicalBundleExporter
FilesystemSnapshotImporter = CanonicalBundleImporter
