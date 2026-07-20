from __future__ import annotations

import hashlib
import unittest
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from kraken_manager.application import (
    CommandContext,
    ConflictError,
    ImportPluginResultCommand,
    ImportPluginResultHandler,
    StorageBackendKind,
    StorageCapabilities,
    StorageProfile,
    StorageScope,
    StoredContent,
    SubmitPluginJobCommand,
    SubmitPluginJobHandler,
)
from kraken_manager.domain import (
    ArtifactSeries,
    ArtifactVersion,
    BlobRef,
    FrameSelectionV1,
    GridOrientation,
    Layer,
    LayerType,
    PluginFrameOutcome,
    PluginFrameResultV1,
    PluginInputV1,
    PluginJobState,
    PluginResultManifestV1,
    PluginResultOutcome,
    Principal,
    Project,
    ProjectRole,
    Representation,
    RepresentationKind,
)
from kraken_manager.domain.common import ArtifactVersionId, FrameId, ProjectId
from kraken_manager.domain.identity import ProjectRoleAssignment
from kraken_manager.infrastructure.projections.rebuild import ProjectionRebuilder


NOW = datetime(2026, 7, 17, 9, tzinfo=timezone.utc)


class Clock:
    def now(self):
        return NOW


class Events:
    def __init__(self):
        self.streams = defaultdict(list)

    def load_stream(self, stream_id, *, after_revision=0, as_of=None):
        return tuple(event for event in self.streams[stream_id] if event.revision > after_revision)

    def current_revision(self, stream_id):
        return len(self.streams[stream_id])

    def append(self, stream_id, *, expected_revision, events: Sequence[object]):
        if len(self.streams[stream_id]) != expected_revision:
            raise RuntimeError("revision conflict")
        self.streams[stream_id].extend(events)
        return len(self.streams[stream_id])

    def find_by_idempotency_key(self, project_id, key):
        return tuple(
            event
            for stream in self.streams.values()
            for event in stream
            if event.project_id == project_id and event.idempotency_key == key
        )


class Projections:
    def __init__(self):
        self.projects = {}
        self.layers = {}
        self.representations = {}
        self.series = {}
        self.versions = {}
        self.active_versions = {}
        self.jobs = {}

    def get_project(self, identifier, *, as_of=None):
        return self.projects.get(identifier)

    def get_layer(self, identifier, *, as_of=None):
        return self.layers.get(identifier)

    def get_representation(self, identifier, *, as_of=None):
        return self.representations.get(identifier)

    def get_artifact_series(self, identifier, *, as_of=None):
        return self.series.get(identifier)

    def save_artifact_series(self, series):
        self.series[series.id] = series

    def get_artifact_version(self, identifier, *, as_of=None):
        return self.versions.get(identifier)

    def get_active_artifact_version(self, series_id, *, as_of=None):
        identifier = self.active_versions.get(series_id)
        return None if identifier is None else self.versions[identifier]

    def save_artifact_version(self, version, *, activate):
        self.versions[version.id] = version
        if activate:
            self.active_versions[version.series_id] = version.id

    def save_plugin_job(self, job):
        self.jobs[job.id] = job

    def get_plugin_job(self, identifier, *, as_of=None):
        return self.jobs.get(identifier)


class Blobs:
    def __init__(self):
        self.values = {}

    def put(self, chunks: Iterable[bytes], *, expected_sha256=None):
        content = b"".join(chunks)
        digest = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ValueError("hash mismatch")
        existed = digest in self.values
        self.values[digest] = content
        return StoredContent(BlobRef(digest, len(content)), existed)

    def iter_bytes(self, reference, *, chunk_size=1024 * 1024) -> Iterator[bytes]:
        yield self.values[reference.sha256]

    def exists(self, reference):
        return reference.sha256 in self.values


class Acl:
    def __init__(self):
        self.assignments = []

    def roles_for(self, project_id, principal_id):
        return frozenset(
            item.role
            for item in self.assignments
            if item.project_id == project_id and item.principal_id == principal_id and item.active
        )


class Uow:
    def __init__(self):
        self.event_store = Events()
        self.projections = Projections()
        self.blobs = Blobs()
        self.acl = Acl()
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class Factory:
    def __init__(self):
        self.uow = Uow()

    def __call__(self):
        return self.uow


class Profiles:
    def __init__(self, profile):
        self.profile = profile

    def get(self, identifier):
        return self.profile if identifier == self.profile.id else None

    def list(self):
        return (self.profile,)


class Gateway:
    def __init__(self):
        self.submitted = []

    def is_available(self, capability, protocol_version):
        return capability == "frames.vectorize.v1" and protocol_version == "1.0"

    def submit(self, manifest):
        self.submitted.append(manifest)

    def cancel(self, job_id):
        pass


class ResultReader:
    def __init__(self):
        self.files = {}
        self.reads = 0

    def iter_output(self, manifest, relative_path):
        self.reads += 1
        yield self.files[relative_path]


class Fixture:
    def __init__(self, *, shared=False, frame_count=1):
        self.actor = (
            Principal.gitlab(issuer="https://gitlab.local", subject="42", display_name="Alice")
            if shared
            else Principal.local(subject="alice", display_name="Alice")
        )
        self.profile = StorageProfile(
            id="shared" if shared else "local",
            name="Profile",
            metadata_backend=StorageBackendKind.POSTGRESQL if shared else StorageBackendKind.FILESYSTEM,
            blob_backend="filesystem",
            scope=StorageScope.SHARED if shared else StorageScope.LOCAL,
            capabilities=StorageCapabilities(True, True, True, True, True, 1_000_000),
        )
        self.factory = Factory()
        self.gateway = Gateway()
        self.reader = ResultReader()
        self.project = replace(
            Project.create(
                project_id=ProjectId(str(uuid4())),
                name="Chip",
                width=10,
                height=10,
                orientation=GridOrientation.Y_DOWN,
                storage_profile=self.profile.id,
                created_at=NOW,
            ),
            revision=1,
        )
        self.layer = Layer.create(
            project_id=self.project.id,
            name="Metal",
            type=LayerType.METAL,
            order=0,
            created_at=NOW,
        )
        self.source = Representation.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            name="Images",
            kind=RepresentationKind.IMAGE,
            created_at=NOW,
        )
        self.target = Representation.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            name="Vectors",
            kind=RepresentationKind.VECTOR,
            created_at=NOW,
        )
        projections = self.factory.uow.projections
        projections.projects[self.project.id] = self.project
        projections.layers[self.layer.id] = self.layer
        projections.representations[self.source.id] = self.source
        projections.representations[self.target.id] = self.target
        self.factory.uow.acl.assignments.append(
            ProjectRoleAssignment.create(
                project_id=self.project.id,
                principal_id=self.actor.id,
                role=ProjectRole.OWNER,
                assigned_by=self.actor.id,
                assigned_at=NOW,
            )
        )
        inputs = []
        for x in range(1, frame_count + 1):
            frame_id = self.project.frame_id_at(x, 1)
            content = f"input-{x}".encode()
            blob = BlobRef(hashlib.sha256(content).hexdigest(), len(content))
            series = ArtifactSeries.for_frame(
                project_id=self.project.id,
                layer_id=self.layer.id,
                representation_id=self.source.id,
                frame_id=frame_id,
                name=f"{x}_1.png",
            )
            version = ArtifactVersion.managed(
                series_id=series.id,
                blob=blob,
                media_type="image/png",
                filename=f"{x}_1.png",
                author_principal_id=self.actor.id,
                created_at=NOW,
            )
            projections.save_artifact_series(series)
            projections.save_artifact_version(version, activate=True)
            inputs.append(PluginInputV1(frame_id, version.id, version.sha256, f"inputs/{x}_1.png"))
        self.inputs = tuple(inputs)
        self.selection = FrameSelectionV1.rectangle(1, 1, frame_count, 1)
        self.profiles = Profiles(self.profile)

    def submit(self):
        return SubmitPluginJobHandler(
            self.factory, self.profiles, Clock(), self.gateway
        )(
            SubmitPluginJobCommand(
                context=CommandContext(
                    actor=self.actor,
                    idempotency_key="submit-job",
                    gitlab_identity_verified=self.actor.issuer is not None,
                ),
                project_id=self.project.id,
                layer_id=self.layer.id,
                target_representation_id=self.target.id,
                selection=self.selection,
                capability="frames.vectorize.v1",
                inputs=self.inputs,
                parameters={"threshold": 0.5},
            )
        )

    def importer(self):
        return ImportPluginResultHandler(
            self.factory, self.profiles, Clock(), self.reader
        )

    def result(self, job, *, count=None, bad_frame=None):
        count = len(self.inputs) if count is None else count
        results = []
        for index, item in enumerate(self.inputs[:count], start=1):
            frame_id = bad_frame if index == 1 and bad_frame is not None else item.frame_id
            path = f"outputs/{index}_1.cif"
            content = f"vector-{index}".encode()
            self.reader.files[path] = content
            results.append(
                PluginFrameResultV1(
                    output_id=str(uuid4()),
                    frame_id=frame_id,
                    outcome=PluginFrameOutcome.SUCCEEDED,
                    relative_path=path,
                    sha256=hashlib.sha256(content).hexdigest(),
                    media_type="application/x-cif",
                    role="vector",
                )
            )
        return PluginResultManifestV1(
            job_id=job.id,
            plugin_name="Contour",
            plugin_version="2.0",
            results=tuple(results),
            parameters_applied={"threshold": 0.5},
        )


class PluginJobApplicationTests(unittest.TestCase):
    def test_submission_is_persisted_and_agent_handoff_is_retryable(self):
        fixture = Fixture()
        job = fixture.submit()

        command_manifest = fixture.gateway.submitted[0]
        retry = SubmitPluginJobHandler(
            fixture.factory, fixture.profiles, Clock(), fixture.gateway
        )(
            SubmitPluginJobCommand(
                context=CommandContext(actor=fixture.actor, idempotency_key="submit-job"),
                project_id=fixture.project.id,
                layer_id=fixture.layer.id,
                target_representation_id=fixture.target.id,
                selection=fixture.selection,
                capability="frames.vectorize.v1",
                inputs=fixture.inputs,
                parameters={"threshold": 0.5},
            )
        )

        self.assertEqual(retry, job)
        self.assertEqual(len(fixture.factory.uow.event_store.streams[f"plugin-job:{job.id}"]), 1)
        self.assertEqual(fixture.gateway.submitted, [command_manifest, command_manifest])
        rebuilt = Projections()
        created = fixture.factory.uow.event_store.streams[f"plugin-job:{job.id}"][0]
        self.assertTrue(ProjectionRebuilder(rebuilt).apply(created))
        self.assertEqual(rebuilt.get_plugin_job(job.id), job)

    def test_successful_result_creates_immutable_provenance_and_duplicate_is_noop(self):
        fixture = Fixture()
        job = fixture.submit()
        manifest = fixture.result(job)
        command = ImportPluginResultCommand(
            context=CommandContext(actor=fixture.actor, idempotency_key="result-callback"),
            manifest=manifest,
        )

        imported = fixture.importer()(command)
        repeated = fixture.importer()(command)

        self.assertEqual(imported.job.state, PluginJobState.SUCCEEDED)
        self.assertEqual(len(imported.versions), 1)
        version = imported.versions[0]
        self.assertEqual(str(version.id), manifest.results[0].output_id)
        self.assertEqual(version.input_version_ids, (fixture.inputs[0].artifact_version_id,))
        self.assertEqual(version.tool_name, "Contour")
        self.assertEqual(fixture.reader.reads, 1)
        self.assertTrue(repeated.already_imported)
        self.assertEqual(repeated.versions, imported.versions)

    def test_gitlab_outage_defers_bytes_and_same_callback_can_resume_after_live_check(self):
        fixture = Fixture(shared=True)
        job = fixture.submit()
        manifest = fixture.result(job)
        offline = ImportPluginResultCommand(
            context=CommandContext(actor=fixture.actor, idempotency_key="callback"),
            manifest=manifest,
        )

        deferred = fixture.importer()(offline)
        self.assertTrue(deferred.awaiting_authorization)
        self.assertEqual(deferred.job.state, PluginJobState.AWAITING_AUTHORIZATION)
        self.assertEqual(fixture.reader.reads, 0)

        resumed = fixture.importer()(
            ImportPluginResultCommand(
                context=CommandContext(
                    actor=fixture.actor,
                    idempotency_key="callback",
                    gitlab_identity_verified=True,
                ),
                manifest=manifest,
            )
        )
        self.assertEqual(resumed.job.state, PluginJobState.SUCCEEDED)
        self.assertEqual(len(resumed.versions), 1)

    def test_partial_result_needs_explicit_confirmation(self):
        fixture = Fixture(frame_count=2)
        job = fixture.submit()
        manifest = fixture.result(job, count=1)
        preview_command = ImportPluginResultCommand(
            context=CommandContext(actor=fixture.actor, idempotency_key="partial"),
            manifest=manifest,
        )

        preview = fixture.importer()(preview_command)
        self.assertTrue(preview.requires_partial_confirmation)
        self.assertEqual(preview.job.state, PluginJobState.PARTIAL)
        self.assertEqual(fixture.reader.reads, 0)

        committed = fixture.importer()(
            ImportPluginResultCommand(
                context=CommandContext(actor=fixture.actor, idempotency_key="partial"),
                manifest=manifest,
                confirm_partial=True,
            )
        )
        self.assertEqual(committed.job.state, PluginJobState.PARTIAL)
        self.assertEqual(len(committed.versions), 1)

    def test_failed_and_cancelled_results_finish_without_publishing_artifacts(self):
        for outcome, state in (
            (PluginResultOutcome.FAILED, PluginJobState.FAILED),
            (PluginResultOutcome.CANCELLED, PluginJobState.CANCELLED),
        ):
            with self.subTest(outcome=outcome):
                fixture = Fixture()
                job = fixture.submit()
                manifest = PluginResultManifestV1(
                    job_id=job.id,
                    plugin_name="Contour",
                    plugin_version="2.0",
                    results=(),
                    parameters_applied={},
                    outcome=outcome,
                )
                command = ImportPluginResultCommand(
                    context=CommandContext(actor=fixture.actor, idempotency_key=f"terminal-{outcome.value}"),
                    manifest=manifest,
                )

                terminal = fixture.importer()(command)
                repeated = fixture.importer()(command)

                self.assertEqual(terminal.job.state, state)
                self.assertEqual(terminal.versions, ())
                self.assertEqual(fixture.reader.reads, 0)
                self.assertTrue(repeated.already_imported)
                self.assertFalse(
                    any(version.tool_name == "Contour" for version in fixture.factory.uow.projections.versions.values())
                )

    def test_unknown_frame_and_content_hash_mismatch_are_rejected(self):
        fixture = Fixture()
        job = fixture.submit()
        unknown = fixture.result(job, bad_frame=FrameId(str(uuid4())))
        with self.assertRaisesRegex(ConflictError, "unknown frame"):
            fixture.importer()(
                ImportPluginResultCommand(
                    context=CommandContext(actor=fixture.actor, idempotency_key="unknown"),
                    manifest=unknown,
                )
            )

        bad_hash = fixture.result(job)
        fixture.reader.files[bad_hash.results[0].relative_path] = b"tampered"  # type: ignore[index]
        with self.assertRaisesRegex(ConflictError, "bad SHA-256"):
            fixture.importer()(
                ImportPluginResultCommand(
                    context=CommandContext(actor=fixture.actor, idempotency_key="bad-hash"),
                    manifest=bad_hash,
                )
            )
        self.assertFalse(any(version.tool_name == "Contour" for version in fixture.factory.uow.projections.versions.values()))


if __name__ == "__main__":
    unittest.main()
