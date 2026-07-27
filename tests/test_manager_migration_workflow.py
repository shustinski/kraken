from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from kraken_manager.application.dto import StorageBackendKind, StorageScope
from kraken_manager.application.ports import (
    SnapshotCheckpoint,
    SnapshotChunk,
    StorageCapabilities,
    StorageProfile,
)
from kraken_manager.domain.common import ProjectId
from kraken_manager.infrastructure.migration import (
    CapacityReport,
    ExternalReferenceRecord,
    IdentityMappingReport,
    JsonMigrationRecordRepository,
    MigrationNotReady,
    MigrationPreflightError,
    MigrationState,
    MigrationTransferInterrupted,
    MigrationVerificationFailed,
    MigrationWorkflowService,
    PlanMigrationRequest,
    SnapshotFingerprint,
    SnapshotInventory,
)


def _profile(profile_id: str, scope: StorageScope, *, max_frames: int = 1_000_000) -> StorageProfile:
    return StorageProfile(
        id=profile_id,
        name=profile_id,
        metadata_backend=(
            StorageBackendKind.FILESYSTEM
            if scope is StorageScope.LOCAL
            else StorageBackendKind.POSTGRESQL
        ),
        blob_backend="filesystem",
        scope=scope,
        capabilities=StorageCapabilities(
            multi_writer=scope is StorageScope.SHARED,
            transactions=True,
            snapshots=True,
            streaming=True,
            external_references=True,
            max_frames=max_frames,
        ),
    )


def _fingerprint(sequence: int, kind: str, key: str, payload: bytes) -> SnapshotFingerprint:
    return SnapshotFingerprint(
        sequence=sequence,
        kind=kind,
        key=key,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


class FakeSource:
    def __init__(self, profile: StorageProfile, locator: str) -> None:
        self.profile = profile
        self.locator = locator
        self.payloads = [b'{"event":1}\n', b"blob-one", b"snapshot-state"]
        self.kinds = ["events", "blob", "snapshot"]
        self.keys = ["events/1", "sha256/blob-one", "projection/main"]
        self.fail_before_sequence: int | None = None
        self.failed = False
        self.resume_positions: list[int] = []
        self.event_count = 1
        self.stream_revisions = {"project:one": 1}

    def inspect_snapshot(self, project_id: ProjectId) -> SnapshotInventory:
        return SnapshotInventory(
            schema_version=1,
            frame_count=100,
            event_count=self.event_count,
            entries=tuple(
                _fingerprint(index, kind, key, payload)
                for index, (kind, key, payload) in enumerate(
                    zip(self.kinds, self.keys, self.payloads, strict=True), start=1
                )
            ),
            stream_revisions=self.stream_revisions,
            external_references=(
                ExternalReferenceRecord("file:///offline/stitch.bdt"),
            ),
        )

    def export_snapshot(
        self,
        project_id: ProjectId,
        *,
        resume_from: SnapshotCheckpoint | None = None,
    ):
        last_sequence = 0 if resume_from is None else resume_from.last_sequence
        self.resume_positions.append(last_sequence)
        for sequence, (kind, key, payload) in enumerate(
            zip(self.kinds, self.keys, self.payloads, strict=True), start=1
        ):
            if sequence <= last_sequence:
                continue
            if self.fail_before_sequence == sequence and not self.failed:
                self.failed = True
                raise OSError("simulated removable-media disconnect")
            yield SnapshotChunk(
                kind=kind,
                key=key,
                sequence=sequence,
                payload=payload,
                sha256=hashlib.sha256(payload).hexdigest(),
            )


class FakeDestination:
    def __init__(
        self,
        profile: StorageProfile,
        locator: str,
        source: FakeSource,
        *,
        available_bytes: int | None = 10_000,
    ) -> None:
        self.profile = profile
        self.locator = locator
        self.source = source
        self.available_bytes = available_bytes
        self.payloads: dict[int, tuple[str, str, bytes]] = {}

    def capacity(self, project_id: ProjectId) -> CapacityReport:
        return CapacityReport(self.available_bytes)

    def supports_schema(self, schema_version: int) -> bool:
        return schema_version == 1

    def import_snapshot(self, chunks, *, resume_from: SnapshotCheckpoint | None = None):
        last = 0 if resume_from is None else resume_from.last_sequence
        consumed = False
        for chunk in chunks:
            consumed = True
            if chunk.sequence != last + 1:
                raise RuntimeError("non-contiguous destination import")
            self.payloads[chunk.sequence] = (chunk.kind, chunk.key, bytes(chunk.payload))
            last = chunk.sequence
        return SnapshotCheckpoint(
            project_id=ProjectId(str(PROJECT_ID)),
            last_sequence=last,
            token=f"target-{last}",
            complete=not consumed,
        )

    def inspect_snapshot(self, project_id: ProjectId) -> SnapshotInventory:
        entries = tuple(
            _fingerprint(sequence, kind, key, payload)
            for sequence, (kind, key, payload) in sorted(self.payloads.items())
        )
        return SnapshotInventory(
            schema_version=1,
            frame_count=100,
            event_count=self.source.event_count if len(entries) == len(self.source.payloads) else 0,
            entries=entries,
            stream_revisions=(
                self.source.stream_revisions if len(entries) == len(self.source.payloads) else {}
            ),
            external_references=(
                (ExternalReferenceRecord("file:///offline/stitch.bdt"),)
                if len(entries) == len(self.source.payloads)
                else ()
            ),
        )


class FakeReadOnlyGuard:
    def __init__(self) -> None:
        self.tokens: dict[str, bool] = {}

    def engage(self, project_id: ProjectId, migration_id: str) -> str:
        token = f"guard:{migration_id}"
        self.tokens[token] = True
        return token

    def finalize(self, token: str, *, retain_read_only: bool) -> None:
        if token not in self.tokens:
            raise RuntimeError("unknown guard")
        self.tokens[token] = retain_read_only

    def is_read_only(self, token: str) -> bool:
        return self.tokens[token]


class FakeLocator:
    def __init__(self, project_id: ProjectId, locator: str) -> None:
        self.values = {str(project_id): locator}

    def current(self, project_id: ProjectId) -> str:
        return self.values[str(project_id)]

    def compare_and_swap(self, project_id: ProjectId, *, expected: str, replacement: str) -> bool:
        key = str(project_id)
        if self.values[key] != expected:
            return False
        self.values[key] = replacement
        return True


PROJECT_ID = ProjectId(str(uuid4()))


class MigrationWorkflowTests(unittest.TestCase):
    def _workflow(
        self,
        root: Path,
        *,
        source_scope: StorageScope = StorageScope.LOCAL,
        destination_scope: StorageScope = StorageScope.SHARED,
        available_bytes: int | None = 10_000,
    ):
        source = FakeSource(_profile("source", source_scope), "storage://source")
        destination = FakeDestination(
            _profile("destination", destination_scope),
            "storage://destination",
            source,
            available_bytes=available_bytes,
        )
        repository = JsonMigrationRecordRepository(root / "migration-state")
        guard = FakeReadOnlyGuard()
        locator = FakeLocator(PROJECT_ID, source.locator)
        service = MigrationWorkflowService(
            source=source,
            destination=destination,
            records=repository,
            read_only_guard=guard,
            locator=locator,
        )
        return service, source, destination, repository, guard, locator

    @staticmethod
    def _request(*, gitlab_owner: bool = True, unmapped: tuple[str, ...] = ()):
        return PlanMigrationRequest(
            project_id=PROJECT_ID,
            identity_mapping=IdentityMappingReport(
                mapped_principals=2,
                unmapped_principals=unmapped,
                destination_owner_gitlab_subject=("gitlab-sub-1" if gitlab_owner else None),
            ),
            reserve_bytes=100,
        )

    def test_interruption_is_durably_resumed_then_verified_and_cut_over(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service, source, destination, repository, guard, locator = self._workflow(root)
            source.fail_before_sequence = 2
            planned = service.plan(self._request())
            self.assertTrue(planned.preflight.ready)
            self.assertIn("external_references_not_copied", {w.code for w in planned.preflight.warnings})

            with self.assertRaises(MigrationTransferInterrupted) as raised:
                service.start(planned.migration_id)
            self.assertEqual(raised.exception.last_sequence, 1)
            durable = repository.get(planned.migration_id)
            self.assertIsNotNone(durable)
            assert durable is not None
            self.assertEqual(durable.state, MigrationState.INTERRUPTED)
            self.assertEqual(durable.checkpoint.last_sequence, 1)
            self.assertTrue(guard.is_read_only(durable.source_guard_token or ""))

            # Reconstructing the service demonstrates that resume does not rely
            # on an in-memory checkpoint owned by the UI process.
            restarted = MigrationWorkflowService(
                source=source,
                destination=destination,
                records=JsonMigrationRecordRepository(root / "migration-state"),
                read_only_guard=guard,
                locator=locator,
            )
            copied = restarted.resume(planned.migration_id)
            self.assertEqual(copied.state, MigrationState.COPIED)
            self.assertEqual(source.resume_positions, [0, 1])
            self.assertEqual(copied.checkpoint.chunks_copied, 3)
            self.assertTrue(copied.checkpoint.complete)

            verified = restarted.verify(planned.migration_id)
            self.assertTrue(verified.verification and verified.verification.valid)
            cut_over = restarted.cutover(planned.migration_id)
            self.assertEqual(locator.current(PROJECT_ID), destination.locator)
            self.assertTrue(guard.is_read_only(cut_over.source_guard_token or ""))
            finalized = restarted.finalize(planned.migration_id)
            self.assertEqual(finalized.state, MigrationState.FINALIZED)
            # A successful migration keeps the old source sealed as a recovery copy.
            self.assertTrue(guard.is_read_only(finalized.source_guard_token or ""))

    def test_tamper_or_revision_mismatch_blocks_cutover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _, destination, repository, _, locator = self._workflow(Path(directory))
            planned = service.plan(self._request())
            service.start(planned.migration_id)
            kind, key, _ = destination.payloads[2]
            destination.payloads[2] = (kind, key, b"tampered")

            with self.assertRaises(MigrationVerificationFailed) as raised:
                service.verify(planned.migration_id)
            self.assertTrue(any("hash/size mismatch" in item for item in raised.exception.report.errors))
            failed = repository.get(planned.migration_id)
            assert failed is not None
            self.assertEqual(failed.state, MigrationState.VERIFICATION_FAILED)
            with self.assertRaises(MigrationNotReady):
                service.cutover(planned.migration_id)
            self.assertEqual(locator.current(PROJECT_ID), "storage://source")

            # Even if bytes are repaired, a revision mismatch is a full failure.
            destination.payloads[2] = (kind, key, b"blob-one")
            destination.source.stream_revisions = {"project:one": 2}
            with self.assertRaises(MigrationVerificationFailed) as revision_failure:
                service.verify(planned.migration_id)
            self.assertTrue(any("stream revisions mismatch" in e for e in revision_failure.exception.report.errors))

    def test_cutover_rollback_restores_locator_and_releases_guard_only_on_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _, destination, _, guard, locator = self._workflow(Path(directory))
            planned = service.plan(self._request())
            service.start(planned.migration_id)
            service.verify(planned.migration_id)
            cut_over = service.cutover(planned.migration_id)
            rolled_back = service.rollback(planned.migration_id)
            self.assertEqual(rolled_back.state, MigrationState.ROLLED_BACK)
            self.assertEqual(locator.current(PROJECT_ID), "storage://source")
            self.assertTrue(guard.is_read_only(cut_over.source_guard_token or ""))
            finalized = service.finalize(planned.migration_id)
            self.assertFalse(guard.is_read_only(finalized.source_guard_token or ""))
            self.assertNotEqual(locator.current(PROJECT_ID), destination.locator)

    def test_preflight_enforces_identity_capacity_and_direction_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, _, _, _, _, _ = self._workflow(Path(directory), available_bytes=1)
            planned = service.plan(self._request(gitlab_owner=False, unmapped=("local:alice",)))
            codes = {issue.code for issue in planned.preflight.errors}
            self.assertEqual(
                codes,
                {"insufficient_space", "identity_mapping_incomplete", "gitlab_owner_required"},
            )
            with self.assertRaises(MigrationPreflightError):
                service.start(planned.migration_id)

        with tempfile.TemporaryDirectory() as directory:
            service, _, _, _, _, _ = self._workflow(
                Path(directory),
                source_scope=StorageScope.SHARED,
                destination_scope=StorageScope.LOCAL,
            )
            planned = service.plan(self._request(gitlab_owner=False))
            self.assertTrue(planned.preflight.ready)
            self.assertIn("shared_catalog_removal", {w.code for w in planned.preflight.warnings})


if __name__ == "__main__":
    unittest.main()
