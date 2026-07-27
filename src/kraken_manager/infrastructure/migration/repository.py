"""Durable JSON repository for migration workflow records.

The repository stores no project payload; it only persists orchestration state
and checkpoints.  Each revision is replaced atomically, making a workstation
restart between chunks resumable without trusting transient UI state.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from kraken_manager.domain.common import ProjectId

from .workflow import (
    ExternalReferenceRecord,
    IdentityMappingReport,
    MigrationCheckpoint,
    MigrationOutcome,
    MigrationPreflight,
    MigrationRecord,
    MigrationState,
    MigrationVerificationReport,
    MigrationWorkflowError,
    PreflightIssue,
    SnapshotFingerprint,
    SnapshotInventory,
)


RECORD_SCHEMA = "kraken-migration-workflow/v1"


class MigrationRecordConflict(MigrationWorkflowError):
    """Optimistic workflow record revision did not match durable state."""


class JsonMigrationRecordRepository:
    """Atomic local repository suitable for the desktop agent/workstation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()

    def add(self, record: MigrationRecord) -> None:
        with self._mutex:
            path = self._path(record.migration_id)
            if path.exists():
                raise MigrationRecordConflict(f"migration {record.migration_id} already exists")
            self._write(path, _record_to_dict(record))

    def get(self, migration_id: str) -> MigrationRecord | None:
        path = self._path(migration_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MigrationWorkflowError(f"cannot read migration record {migration_id}") from exc
        if not isinstance(value, dict):
            raise MigrationWorkflowError(f"migration record {migration_id} is not an object")
        try:
            return _record_from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise MigrationWorkflowError(f"invalid migration record {migration_id}") from exc

    def save(self, record: MigrationRecord, *, expected_revision: int) -> None:
        with self._mutex:
            current = self.get(record.migration_id)
            if current is None:
                raise MigrationRecordConflict(f"migration {record.migration_id} does not exist")
            if current.record_revision != expected_revision:
                raise MigrationRecordConflict(
                    f"migration revision conflict: expected {expected_revision}, "
                    f"found {current.record_revision}"
                )
            if record.record_revision != expected_revision + 1:
                raise MigrationRecordConflict("new migration record must advance exactly one revision")
            self._write(self._path(record.migration_id), _record_to_dict(record))

    def _path(self, migration_id: str) -> Path:
        try:
            canonical = str(UUID(migration_id))
        except (TypeError, ValueError) as exc:
            raise ValueError("migration ID must be a UUID") from exc
        return self.root / f"{canonical}.json"

    @staticmethod
    def _write(path: Path, value: Mapping[str, Any]) -> None:
        descriptor = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(descriptor)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            # Persist the directory entry where platforms expose directory fsync.
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    pass
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


def _record_to_dict(record: MigrationRecord) -> dict[str, Any]:
    inventory = record.inventory
    return {
        "schema": RECORD_SCHEMA,
        "migration_id": record.migration_id,
        "project_id": str(record.project_id),
        "source_profile_id": record.source_profile_id,
        "destination_profile_id": record.destination_profile_id,
        "source_locator": record.source_locator,
        "destination_locator": record.destination_locator,
        "state": record.state.value,
        "outcome": record.outcome.value,
        "inventory": {
            "schema_version": inventory.schema_version,
            "frame_count": inventory.frame_count,
            "event_count": inventory.event_count,
            "entries": [
                {
                    "sequence": item.sequence,
                    "kind": item.kind,
                    "key": item.key,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in inventory.entries
            ],
            "stream_revisions": dict(inventory.stream_revisions),
            "external_references": [
                {"uri": item.uri, "fingerprint_sha256": item.fingerprint_sha256}
                for item in inventory.external_references
            ],
        },
        "identity_mapping": {
            "mapped_principals": record.identity_mapping.mapped_principals,
            "unmapped_principals": list(record.identity_mapping.unmapped_principals),
            "destination_owner_gitlab_subject": (
                record.identity_mapping.destination_owner_gitlab_subject
            ),
        },
        "preflight": {
            "required_bytes": record.preflight.required_bytes,
            "available_bytes": record.preflight.available_bytes,
            "issues": [
                {"code": item.code, "message": item.message, "blocking": item.blocking}
                for item in record.preflight.issues
            ],
        },
        "checkpoint": {
            "last_sequence": record.checkpoint.last_sequence,
            "chunks_copied": record.checkpoint.chunks_copied,
            "bytes_copied": record.checkpoint.bytes_copied,
            "destination_token": record.checkpoint.destination_token,
            "complete": record.checkpoint.complete,
        },
        "source_guard_token": record.source_guard_token,
        "verification": (
            None
            if record.verification is None
            else {
                "valid": record.verification.valid,
                "source_unchanged": record.verification.source_unchanged,
                "checked_entries": record.verification.checked_entries,
                "checked_bytes": record.verification.checked_bytes,
                "errors": list(record.verification.errors),
            }
        ),
        "last_error": record.last_error,
        "record_revision": record.record_revision,
    }


def _record_from_dict(value: Mapping[str, Any]) -> MigrationRecord:
    if value.get("schema") != RECORD_SCHEMA:
        raise ValueError("unsupported migration workflow record schema")
    inventory_value = value["inventory"]
    identity_value = value["identity_mapping"]
    preflight_value = value["preflight"]
    checkpoint_value = value["checkpoint"]
    verification_value = value.get("verification")
    if not all(
        isinstance(item, dict)
        for item in (inventory_value, identity_value, preflight_value, checkpoint_value)
    ):
        raise ValueError("invalid nested migration workflow record")
    inventory = SnapshotInventory(
        schema_version=int(inventory_value["schema_version"]),
        frame_count=int(inventory_value["frame_count"]),
        event_count=int(inventory_value["event_count"]),
        entries=tuple(
            SnapshotFingerprint(
                sequence=int(item["sequence"]),
                kind=str(item["kind"]),
                key=str(item["key"]),
                sha256=str(item["sha256"]),
                size_bytes=int(item["size_bytes"]),
            )
            for item in inventory_value["entries"]
        ),
        stream_revisions={
            str(stream): int(revision)
            for stream, revision in inventory_value["stream_revisions"].items()
        },
        external_references=tuple(
            ExternalReferenceRecord(
                uri=str(item["uri"]),
                fingerprint_sha256=item.get("fingerprint_sha256"),
            )
            for item in inventory_value.get("external_references", [])
        ),
    )
    verification = None
    if verification_value is not None:
        if not isinstance(verification_value, dict):
            raise ValueError("invalid verification report")
        verification = MigrationVerificationReport(
            valid=bool(verification_value["valid"]),
            source_unchanged=bool(verification_value["source_unchanged"]),
            checked_entries=int(verification_value["checked_entries"]),
            checked_bytes=int(verification_value["checked_bytes"]),
            errors=tuple(str(item) for item in verification_value.get("errors", [])),
        )
    return MigrationRecord(
        migration_id=str(value["migration_id"]),
        project_id=ProjectId(str(value["project_id"])),
        source_profile_id=str(value["source_profile_id"]),
        destination_profile_id=str(value["destination_profile_id"]),
        source_locator=str(value["source_locator"]),
        destination_locator=str(value["destination_locator"]),
        state=MigrationState(str(value["state"])),
        outcome=MigrationOutcome(str(value["outcome"])),
        inventory=inventory,
        identity_mapping=IdentityMappingReport(
            mapped_principals=int(identity_value["mapped_principals"]),
            unmapped_principals=tuple(
                str(item) for item in identity_value.get("unmapped_principals", [])
            ),
            destination_owner_gitlab_subject=identity_value.get(
                "destination_owner_gitlab_subject"
            ),
        ),
        preflight=MigrationPreflight(
            required_bytes=int(preflight_value["required_bytes"]),
            available_bytes=(
                None
                if preflight_value.get("available_bytes") is None
                else int(preflight_value["available_bytes"])
            ),
            issues=tuple(
                PreflightIssue(
                    code=str(item["code"]),
                    message=str(item["message"]),
                    blocking=bool(item["blocking"]),
                )
                for item in preflight_value.get("issues", [])
            ),
        ),
        checkpoint=MigrationCheckpoint(
            last_sequence=int(checkpoint_value["last_sequence"]),
            chunks_copied=int(checkpoint_value["chunks_copied"]),
            bytes_copied=int(checkpoint_value["bytes_copied"]),
            destination_token=checkpoint_value.get("destination_token"),
            complete=bool(checkpoint_value["complete"]),
        ),
        source_guard_token=value.get("source_guard_token"),
        verification=verification,
        last_error=value.get("last_error"),
        record_revision=int(value["record_revision"]),
    )


__all__ = [
    "JsonMigrationRecordRepository",
    "MigrationRecordConflict",
    "RECORD_SCHEMA",
]
