"""Crash-resumable orchestration for storage-to-storage project migrations.

The canonical bundle and storage snapshot adapters move bytes.  This module
owns the safety protocol around that transfer: preflight, a durable checkpoint,
source write protection, exhaustive verification, and an atomic locator
cutover.  It intentionally depends only on application contracts and the
standard library so the same workflow can coordinate filesystem, PostgreSQL,
and future storage adapters.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from kraken_manager.application.dto import StorageScope
from kraken_manager.application.ports import (
    SnapshotCheckpoint,
    SnapshotChunk,
    StorageProfile,
    StorageSnapshotExporter,
    StorageSnapshotImporter,
)
from kraken_manager.domain.common import ProjectId


class MigrationWorkflowError(RuntimeError):
    """Base class for workflow failures that are safe to show to an operator."""


class MigrationNotReady(MigrationWorkflowError):
    """The requested operation is not valid in the migration's current state."""


class MigrationPreflightError(MigrationWorkflowError):
    def __init__(self, issues: tuple["PreflightIssue", ...]) -> None:
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues if issue.blocking))


class MigrationTransferInterrupted(MigrationWorkflowError):
    """Transfer stopped after its last durable checkpoint and can be resumed."""

    def __init__(self, migration_id: str, last_sequence: int, cause: BaseException) -> None:
        self.migration_id = migration_id
        self.last_sequence = last_sequence
        self.cause = cause
        super().__init__(
            f"migration {migration_id} interrupted after sequence {last_sequence}: {cause}"
        )


class MigrationVerificationFailed(MigrationWorkflowError):
    def __init__(self, report: "MigrationVerificationReport") -> None:
        self.report = report
        super().__init__("; ".join(report.errors) or "migration verification failed")


class MigrationState(StrEnum):
    PLANNED = "planned"
    COPYING = "copying"
    INTERRUPTED = "interrupted"
    COPIED = "copied"
    VERIFICATION_FAILED = "verification_failed"
    VERIFIED = "verified"
    CUTOVER = "cutover"
    ROLLED_BACK = "rolled_back"
    FINALIZED = "finalized"


class MigrationOutcome(StrEnum):
    PENDING = "pending"
    CUTOVER = "cutover"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class SnapshotFingerprint:
    """Immutable manifest entry for one canonical snapshot chunk."""

    sequence: int
    kind: str
    key: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or self.sequence < 1:
            raise ValueError("snapshot sequence must be positive")
        if not self.kind.strip() or not self.key.strip():
            raise ValueError("snapshot kind and key must not be empty")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("snapshot fingerprint requires a lowercase SHA-256")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("snapshot size must be non-negative")
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True)
class ExternalReferenceRecord:
    uri: str
    fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("external reference URI must not be empty")
        object.__setattr__(self, "uri", self.uri.strip())
        if self.fingerprint_sha256 is not None:
            digest = self.fingerprint_sha256.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("external reference fingerprint must be SHA-256")
            object.__setattr__(self, "fingerprint_sha256", digest)


@dataclass(frozen=True, slots=True)
class SnapshotInventory:
    """Canonical, exhaustive inventory used as the migration acceptance oracle."""

    schema_version: int
    frame_count: int
    event_count: int
    entries: tuple[SnapshotFingerprint, ...]
    stream_revisions: Mapping[str, int]
    external_references: tuple[ExternalReferenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema version must be positive")
        if self.frame_count < 0 or self.event_count < 0:
            raise ValueError("inventory counts must be non-negative")
        ordered = tuple(sorted(self.entries, key=lambda item: item.sequence))
        sequences = tuple(entry.sequence for entry in ordered)
        if sequences != tuple(range(1, len(ordered) + 1)):
            raise ValueError("snapshot sequences must be contiguous and start at one")
        identities = tuple((entry.kind, entry.key) for entry in ordered)
        if len(identities) != len(set(identities)):
            raise ValueError("snapshot inventory contains duplicate kind/key entries")
        revisions = dict(self.stream_revisions)
        if any(not stream.strip() or revision < 0 for stream, revision in revisions.items()):
            raise ValueError("stream revisions require a non-empty stream and non-negative value")
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "stream_revisions", revisions)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def blob_count(self) -> int:
        return sum(entry.kind == "blob" for entry in self.entries)

    @property
    def snapshot_count(self) -> int:
        return sum(entry.kind == "snapshot" for entry in self.entries)


@dataclass(frozen=True, slots=True)
class IdentityMappingReport:
    mapped_principals: int
    unmapped_principals: tuple[str, ...] = ()
    destination_owner_gitlab_subject: str | None = None

    def __post_init__(self) -> None:
        if self.mapped_principals < 0:
            raise ValueError("mapped principal count must be non-negative")
        if self.destination_owner_gitlab_subject is not None:
            subject = self.destination_owner_gitlab_subject.strip()
            object.__setattr__(self, "destination_owner_gitlab_subject", subject or None)

    @property
    def complete(self) -> bool:
        return not self.unmapped_principals


@dataclass(frozen=True, slots=True)
class CapacityReport:
    available_bytes: int | None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.available_bytes is not None and self.available_bytes < 0:
            raise ValueError("available capacity must be non-negative or unknown")


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    code: str
    message: str
    blocking: bool


@dataclass(frozen=True, slots=True)
class MigrationPreflight:
    required_bytes: int
    available_bytes: int | None
    issues: tuple[PreflightIssue, ...]

    @property
    def ready(self) -> bool:
        return not any(issue.blocking for issue in self.issues)

    @property
    def errors(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.blocking)

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if not issue.blocking)


@dataclass(frozen=True, slots=True)
class MigrationCheckpoint:
    last_sequence: int = 0
    chunks_copied: int = 0
    bytes_copied: int = 0
    destination_token: str | None = None
    complete: bool = False

    def __post_init__(self) -> None:
        if min(self.last_sequence, self.chunks_copied, self.bytes_copied) < 0:
            raise ValueError("migration checkpoint counters must be non-negative")
        if self.chunks_copied != self.last_sequence:
            raise ValueError("one durable checkpoint is required for every copied sequence")


@dataclass(frozen=True, slots=True)
class MigrationVerificationReport:
    valid: bool
    source_unchanged: bool
    checked_entries: int
    checked_bytes: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    migration_id: str
    project_id: ProjectId
    source_profile_id: str
    destination_profile_id: str
    source_locator: str
    destination_locator: str
    state: MigrationState
    outcome: MigrationOutcome
    inventory: SnapshotInventory
    identity_mapping: IdentityMappingReport
    preflight: MigrationPreflight
    checkpoint: MigrationCheckpoint = field(default_factory=MigrationCheckpoint)
    source_guard_token: str | None = None
    verification: MigrationVerificationReport | None = None
    last_error: str | None = None
    record_revision: int = 0

    def __post_init__(self) -> None:
        try:
            UUID(self.migration_id)
            UUID(str(self.project_id))
        except (TypeError, ValueError) as exc:
            raise ValueError("migration and project IDs must be UUIDs") from exc
        if self.record_revision < 0:
            raise ValueError("migration record revision must be non-negative")


class MigrationSource(StorageSnapshotExporter, Protocol):
    profile: StorageProfile
    locator: str

    def inspect_snapshot(self, project_id: ProjectId) -> SnapshotInventory: ...


class MigrationDestination(StorageSnapshotImporter, Protocol):
    profile: StorageProfile
    locator: str

    def capacity(self, project_id: ProjectId) -> CapacityReport: ...

    def supports_schema(self, schema_version: int) -> bool: ...

    def inspect_snapshot(self, project_id: ProjectId) -> SnapshotInventory: ...


class MigrationRecordRepository(Protocol):
    def add(self, record: MigrationRecord) -> None: ...

    def get(self, migration_id: str) -> MigrationRecord | None: ...

    def save(self, record: MigrationRecord, *, expected_revision: int) -> None: ...


class SourceReadOnlyGuard(Protocol):
    def engage(self, project_id: ProjectId, migration_id: str) -> str: ...

    def finalize(self, token: str, *, retain_read_only: bool) -> None: ...


class ProjectLocator(Protocol):
    def current(self, project_id: ProjectId) -> str: ...

    def compare_and_swap(self, project_id: ProjectId, *, expected: str, replacement: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class PlanMigrationRequest:
    project_id: ProjectId
    identity_mapping: IdentityMappingReport
    migration_id: str = field(default_factory=lambda: str(uuid4()))
    reserve_bytes: int = 0
    allow_unknown_capacity: bool = False

    def __post_init__(self) -> None:
        if self.reserve_bytes < 0:
            raise ValueError("capacity reserve must be non-negative")


class MigrationWorkflowService:
    """Coordinates one source/destination pair using a durable record store."""

    def __init__(
        self,
        *,
        source: MigrationSource,
        destination: MigrationDestination,
        records: MigrationRecordRepository,
        read_only_guard: SourceReadOnlyGuard,
        locator: ProjectLocator,
    ) -> None:
        self.source = source
        self.destination = destination
        self.records = records
        self.read_only_guard = read_only_guard
        self.locator = locator

    def plan(self, request: PlanMigrationRequest) -> MigrationRecord:
        if self.records.get(request.migration_id) is not None:
            return self._require(request.migration_id)
        inventory = self.source.inspect_snapshot(request.project_id)
        capacity = self.destination.capacity(request.project_id)
        required = inventory.total_bytes + request.reserve_bytes
        issues = self._preflight_issues(request, inventory, capacity, required)
        preflight = MigrationPreflight(required, capacity.available_bytes, tuple(issues))
        record = MigrationRecord(
            migration_id=request.migration_id,
            project_id=request.project_id,
            source_profile_id=self.source.profile.id,
            destination_profile_id=self.destination.profile.id,
            source_locator=self.source.locator,
            destination_locator=self.destination.locator,
            state=MigrationState.PLANNED,
            outcome=MigrationOutcome.PENDING,
            inventory=inventory,
            identity_mapping=request.identity_mapping,
            preflight=preflight,
        )
        self.records.add(record)
        return record

    def start(self, migration_id: str) -> MigrationRecord:
        record = self._require(migration_id)
        if not record.preflight.ready:
            raise MigrationPreflightError(record.preflight.errors)
        if record.state in {MigrationState.COPIED, MigrationState.VERIFIED, MigrationState.CUTOVER}:
            return record
        if record.state is not MigrationState.PLANNED:
            raise MigrationNotReady(f"start requires planned state, found {record.state.value}")
        if self.locator.current(record.project_id) != record.source_locator:
            raise MigrationNotReady("project locator no longer points at the planned source")
        token = self.read_only_guard.engage(record.project_id, record.migration_id)
        record = self._update(
            record,
            state=MigrationState.COPYING,
            source_guard_token=token,
            last_error=None,
        )
        return self._transfer(record)

    def resume(self, migration_id: str) -> MigrationRecord:
        record = self._require(migration_id)
        if record.state in {MigrationState.COPIED, MigrationState.VERIFIED, MigrationState.CUTOVER}:
            return record
        if record.state is not MigrationState.INTERRUPTED:
            raise MigrationNotReady(f"resume requires interrupted state, found {record.state.value}")
        if record.source_guard_token is None:
            raise MigrationNotReady("interrupted migration has no source read-only guard")
        record = self._update(record, state=MigrationState.COPYING, last_error=None)
        return self._transfer(record)

    def verify(self, migration_id: str) -> MigrationRecord:
        record = self._require(migration_id)
        if record.state not in {
            MigrationState.COPIED,
            MigrationState.VERIFICATION_FAILED,
            MigrationState.VERIFIED,
        }:
            raise MigrationNotReady(f"verify requires copied state, found {record.state.value}")
        report = self._build_verification_report(record)
        state = MigrationState.VERIFIED if report.valid else MigrationState.VERIFICATION_FAILED
        record = self._update(record, state=state, verification=report, last_error=None)
        if not report.valid:
            raise MigrationVerificationFailed(report)
        return record

    def cutover(self, migration_id: str) -> MigrationRecord:
        record = self._require(migration_id)
        if record.state is MigrationState.CUTOVER:
            return record
        if record.state is not MigrationState.VERIFIED:
            raise MigrationNotReady(f"cutover requires verified state, found {record.state.value}")

        # Verification is deliberately repeated to close the verify/cutover race.
        report = self._build_verification_report(record)
        if not report.valid:
            record = self._update(
                record,
                state=MigrationState.VERIFICATION_FAILED,
                verification=report,
            )
            raise MigrationVerificationFailed(report)
        if not self.locator.compare_and_swap(
            record.project_id,
            expected=record.source_locator,
            replacement=record.destination_locator,
        ):
            raise MigrationNotReady("project locator changed before cutover")
        return self._update(
            record,
            state=MigrationState.CUTOVER,
            outcome=MigrationOutcome.CUTOVER,
            verification=report,
        )

    def rollback(self, migration_id: str) -> MigrationRecord:
        record = self._require(migration_id)
        if record.state is MigrationState.ROLLED_BACK:
            return record
        if record.state is MigrationState.FINALIZED:
            raise MigrationNotReady("a finalized migration cannot be rolled back")
        allowed = {
            MigrationState.COPYING,
            MigrationState.INTERRUPTED,
            MigrationState.COPIED,
            MigrationState.VERIFICATION_FAILED,
            MigrationState.VERIFIED,
            MigrationState.CUTOVER,
        }
        if record.state not in allowed:
            raise MigrationNotReady(f"rollback is not available from {record.state.value}")
        if record.state is MigrationState.CUTOVER and not self.locator.compare_and_swap(
            record.project_id,
            expected=record.destination_locator,
            replacement=record.source_locator,
        ):
            raise MigrationNotReady("project locator changed after cutover; rollback refused")
        return self._update(
            record,
            state=MigrationState.ROLLED_BACK,
            outcome=MigrationOutcome.ROLLED_BACK,
        )

    def finalize(self, migration_id: str) -> MigrationRecord:
        """Complete cutover/rollback and settle the source write guard.

        A successful cutover permanently retains the old source as a sealed,
        read-only recovery copy.  A rollback releases the guard because the
        source becomes authoritative again.
        """

        record = self._require(migration_id)
        if record.state is MigrationState.FINALIZED:
            return record
        if record.state not in {MigrationState.CUTOVER, MigrationState.ROLLED_BACK}:
            raise MigrationNotReady("only cutover or rolled-back migrations can be finalized")
        if record.source_guard_token is None:
            raise MigrationNotReady("migration has no source read-only guard")
        self.read_only_guard.finalize(
            record.source_guard_token,
            retain_read_only=record.outcome is MigrationOutcome.CUTOVER,
        )
        return self._update(record, state=MigrationState.FINALIZED)

    def _preflight_issues(
        self,
        request: PlanMigrationRequest,
        inventory: SnapshotInventory,
        capacity: CapacityReport,
        required_bytes: int,
    ) -> list[PreflightIssue]:
        issues: list[PreflightIssue] = []
        source = self.source.profile
        destination = self.destination.profile
        if self.source.locator == self.destination.locator:
            issues.append(PreflightIssue("same_location", "source and destination are identical", True))
        for side, profile in (("source", source), ("destination", destination)):
            if not profile.capabilities.snapshots:
                issues.append(
                    PreflightIssue(
                        f"{side}_snapshots_unsupported",
                        f"{side} profile does not support snapshots",
                        True,
                    )
                )
            if not profile.capabilities.streaming:
                issues.append(
                    PreflightIssue(
                        f"{side}_streaming_unsupported",
                        f"{side} profile does not support streaming",
                        True,
                    )
                )
        if destination.capabilities.max_frames is not None and (
            inventory.frame_count > destination.capabilities.max_frames
        ):
            issues.append(
                PreflightIssue(
                    "frame_limit_exceeded",
                    f"project has {inventory.frame_count} frames; destination supports "
                    f"{destination.capabilities.max_frames}",
                    True,
                )
            )
        if not self.destination.supports_schema(inventory.schema_version):
            issues.append(
                PreflightIssue(
                    "schema_unsupported",
                    f"destination does not support schema version {inventory.schema_version}",
                    True,
                )
            )
        if capacity.available_bytes is None:
            issues.append(
                PreflightIssue(
                    "capacity_unknown",
                    capacity.detail or "destination free space could not be determined",
                    not request.allow_unknown_capacity,
                )
            )
        elif capacity.available_bytes < required_bytes:
            issues.append(
                PreflightIssue(
                    "insufficient_space",
                    f"destination has {capacity.available_bytes} bytes, {required_bytes} required",
                    True,
                )
            )
        if inventory.external_references:
            issues.append(
                PreflightIssue(
                    "external_references_not_copied",
                    f"{len(inventory.external_references)} external references will be recorded but not copied",
                    False,
                )
            )
        if not request.identity_mapping.complete:
            issues.append(
                PreflightIssue(
                    "identity_mapping_incomplete",
                    "unmapped principals: " + ", ".join(request.identity_mapping.unmapped_principals),
                    destination.scope is StorageScope.SHARED,
                )
            )
        if source.scope is StorageScope.LOCAL and destination.scope is StorageScope.SHARED:
            if request.identity_mapping.destination_owner_gitlab_subject is None:
                issues.append(
                    PreflightIssue(
                        "gitlab_owner_required",
                        "filesystem-to-shared migration requires a mapped GitLab Project Owner",
                        True,
                    )
                )
        if source.scope is StorageScope.SHARED and destination.scope is StorageScope.LOCAL:
            issues.append(
                PreflightIssue(
                    "shared_catalog_removal",
                    "after cutover the project will disappear from the shared server catalog",
                    False,
                )
            )
        return issues

    def _transfer(self, record: MigrationRecord) -> MigrationRecord:
        checkpoint = record.checkpoint
        source_resume = SnapshotCheckpoint(
            project_id=record.project_id,
            last_sequence=checkpoint.last_sequence,
            token=record.migration_id,
            complete=False,
        )
        destination_resume = (
            SnapshotCheckpoint(
                project_id=record.project_id,
                last_sequence=checkpoint.last_sequence,
                token=checkpoint.destination_token,
                complete=False,
            )
            if checkpoint.destination_token is not None
            else None
        )
        expected_entries = {entry.sequence: entry for entry in record.inventory.entries}
        current = record
        try:
            for chunk in self.source.export_snapshot(record.project_id, resume_from=source_resume):
                expected_sequence = current.checkpoint.last_sequence + 1
                if chunk.sequence != expected_sequence:
                    raise MigrationWorkflowError(
                        f"source yielded sequence {chunk.sequence}, expected {expected_sequence}"
                    )
                expected = expected_entries.get(chunk.sequence)
                if expected is None:
                    raise MigrationWorkflowError(
                        f"source yielded unplanned sequence {chunk.sequence}"
                    )
                self._validate_chunk(chunk, expected)
                imported = self.destination.import_snapshot(
                    (chunk,),
                    resume_from=destination_resume,
                )
                if str(imported.project_id) != str(record.project_id):
                    raise MigrationWorkflowError("destination checkpoint belongs to another project")
                if imported.last_sequence < chunk.sequence:
                    raise MigrationWorkflowError("destination did not durably accept the snapshot chunk")
                destination_resume = imported
                next_checkpoint = MigrationCheckpoint(
                    last_sequence=chunk.sequence,
                    chunks_copied=current.checkpoint.chunks_copied + 1,
                    bytes_copied=current.checkpoint.bytes_copied + len(chunk.payload),
                    destination_token=imported.token,
                    complete=False,
                )
                current = self._update(current, checkpoint=next_checkpoint)

            if current.checkpoint.last_sequence != len(record.inventory.entries):
                raise MigrationWorkflowError(
                    f"source ended at sequence {current.checkpoint.last_sequence}; "
                    f"{len(record.inventory.entries)} planned"
                )
            # Give adapters an explicit end-of-stream notification.  A target
            # may use it to atomically publish staged projections.
            completed = self.destination.import_snapshot((), resume_from=destination_resume)
            if str(completed.project_id) != str(record.project_id):
                raise MigrationWorkflowError("final destination checkpoint belongs to another project")
            checkpoint = replace(
                current.checkpoint,
                destination_token=completed.token,
                complete=True,
            )
            return self._update(
                current,
                state=MigrationState.COPIED,
                checkpoint=checkpoint,
                last_error=None,
            )
        except Exception as exc:
            interrupted = self._update(
                current,
                state=MigrationState.INTERRUPTED,
                last_error=str(exc),
            )
            raise MigrationTransferInterrupted(
                record.migration_id,
                interrupted.checkpoint.last_sequence,
                exc,
            ) from exc

    @staticmethod
    def _validate_chunk(chunk: SnapshotChunk, expected: SnapshotFingerprint) -> None:
        digest = hashlib.sha256(chunk.payload).hexdigest()
        errors: list[str] = []
        if (chunk.kind, chunk.key) != (expected.kind, expected.key):
            errors.append("kind/key differs from planned inventory")
        if chunk.sha256.lower() != expected.sha256:
            errors.append("declared digest differs from planned inventory")
        if digest != expected.sha256:
            errors.append("payload digest differs from planned inventory")
        if len(chunk.payload) != expected.size_bytes:
            errors.append("payload size differs from planned inventory")
        if errors:
            raise MigrationWorkflowError(
                f"invalid snapshot chunk {chunk.sequence}: " + "; ".join(errors)
            )

    def _build_verification_report(self, record: MigrationRecord) -> MigrationVerificationReport:
        errors: list[str] = []
        source_inventory = self.source.inspect_snapshot(record.project_id)
        source_unchanged = source_inventory == record.inventory
        if not source_unchanged:
            errors.extend(self._inventory_errors("source", record.inventory, source_inventory))
        target_inventory = self.destination.inspect_snapshot(record.project_id)
        errors.extend(self._inventory_errors("destination", record.inventory, target_inventory))
        return MigrationVerificationReport(
            valid=not errors,
            source_unchanged=source_unchanged,
            checked_entries=len(target_inventory.entries),
            checked_bytes=target_inventory.total_bytes,
            errors=tuple(errors),
        )

    @staticmethod
    def _inventory_errors(
        label: str,
        expected: SnapshotInventory,
        actual: SnapshotInventory,
    ) -> list[str]:
        errors: list[str] = []
        if expected.schema_version != actual.schema_version:
            errors.append(f"{label} schema version mismatch")
        if expected.frame_count != actual.frame_count:
            errors.append(f"{label} frame count mismatch")
        if expected.event_count != actual.event_count:
            errors.append(f"{label} event count mismatch")
        if dict(expected.stream_revisions) != dict(actual.stream_revisions):
            errors.append(f"{label} stream revisions mismatch")
        expected_entries = {(item.sequence, item.kind, item.key): item for item in expected.entries}
        actual_entries = {(item.sequence, item.kind, item.key): item for item in actual.entries}
        missing = sorted(set(expected_entries) - set(actual_entries))
        extra = sorted(set(actual_entries) - set(expected_entries))
        if missing:
            errors.append(f"{label} missing entries: {missing!r}")
        if extra:
            errors.append(f"{label} extra entries: {extra!r}")
        for identity in sorted(set(expected_entries) & set(actual_entries)):
            wanted = expected_entries[identity]
            found = actual_entries[identity]
            if (wanted.sha256, wanted.size_bytes) != (found.sha256, found.size_bytes):
                errors.append(f"{label} hash/size mismatch for {identity!r}")
        if expected.external_references != actual.external_references:
            errors.append(f"{label} external reference inventory mismatch")
        return errors

    def _require(self, migration_id: str) -> MigrationRecord:
        record = self.records.get(migration_id)
        if record is None:
            raise MigrationWorkflowError(f"unknown migration {migration_id}")
        if (
            record.source_profile_id != self.source.profile.id
            or record.destination_profile_id != self.destination.profile.id
            or record.source_locator != self.source.locator
            or record.destination_locator != self.destination.locator
        ):
            raise MigrationWorkflowError("migration endpoints do not match its durable plan")
        return record

    def _update(self, record: MigrationRecord, **changes: object) -> MigrationRecord:
        updated = replace(record, record_revision=record.record_revision + 1, **changes)
        self.records.save(updated, expected_revision=record.record_revision)
        return updated


class PlanMigration:
    def __init__(self, workflow: MigrationWorkflowService) -> None:
        self.workflow = workflow

    def __call__(self, request: PlanMigrationRequest) -> MigrationRecord:
        return self.workflow.plan(request)


class StartMigration:
    def __init__(self, workflow: MigrationWorkflowService) -> None:
        self.workflow = workflow

    def __call__(self, migration_id: str) -> MigrationRecord:
        return self.workflow.start(migration_id)


class ResumeMigration:
    def __init__(self, workflow: MigrationWorkflowService) -> None:
        self.workflow = workflow

    def __call__(self, migration_id: str) -> MigrationRecord:
        return self.workflow.resume(migration_id)


class VerifyMigration:
    def __init__(self, workflow: MigrationWorkflowService) -> None:
        self.workflow = workflow

    def __call__(self, migration_id: str) -> MigrationRecord:
        return self.workflow.verify(migration_id)


class CutoverMigration:
    def __init__(self, workflow: MigrationWorkflowService) -> None:
        self.workflow = workflow

    def __call__(self, migration_id: str) -> MigrationRecord:
        return self.workflow.cutover(migration_id)


class RollbackMigration:
    def __init__(self, workflow: MigrationWorkflowService) -> None:
        self.workflow = workflow

    def __call__(self, migration_id: str) -> MigrationRecord:
        return self.workflow.rollback(migration_id)


class FinalizeMigration:
    def __init__(self, workflow: MigrationWorkflowService) -> None:
        self.workflow = workflow

    def __call__(self, migration_id: str) -> MigrationRecord:
        return self.workflow.finalize(migration_id)


__all__ = [
    "CapacityReport",
    "CutoverMigration",
    "ExternalReferenceRecord",
    "FinalizeMigration",
    "IdentityMappingReport",
    "MigrationCheckpoint",
    "MigrationDestination",
    "MigrationNotReady",
    "MigrationOutcome",
    "MigrationPreflight",
    "MigrationPreflightError",
    "MigrationRecord",
    "MigrationRecordRepository",
    "MigrationSource",
    "MigrationState",
    "MigrationTransferInterrupted",
    "MigrationVerificationFailed",
    "MigrationVerificationReport",
    "MigrationWorkflowError",
    "MigrationWorkflowService",
    "PlanMigration",
    "PlanMigrationRequest",
    "PreflightIssue",
    "ProjectLocator",
    "ResumeMigration",
    "RollbackMigration",
    "SnapshotFingerprint",
    "SnapshotInventory",
    "SourceReadOnlyGuard",
    "StartMigration",
    "VerifyMigration",
]
