from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import unittest
from uuid import uuid4

from kraken_manager.application import (
    AcceptReviewCommand,
    AcceptReviewHandler,
    AuthorizationError,
    CommandContext,
    CommitReviewReturnCommand,
    CommitReviewReturnHandler,
    ConflictError,
    CreateReviewBatchCommand,
    CreateReviewBatchHandler,
    DryRunReviewReturnCommand,
    DryRunReviewReturnHandler,
    ExportReviewPackageCommand,
    ExportReviewPackageHandler,
    PlanReviewPackageCommand,
    PlanReviewPackageHandler,
    RequestReviewChangesCommand,
    RequestReviewChangesHandler,
    StorageBackendKind,
    StorageCapabilities,
    StorageProfile,
    StorageScope,
    StoredContent,
)
from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion, BlobRef
from kraken_manager.domain.common import (
    PerformerId,
    ProjectId,
)
from kraken_manager.domain.identity import Performer, Principal, ProjectRole
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
)
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import ReviewBatchState, ReviewFileStatus, ReviewItem
from kraken_manager.infrastructure.projections import ProjectionRebuilder


NOW = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)


class Clock:
    def now(self):
        return NOW


class Profiles:
    def __init__(self):
        self.profile = StorageProfile(
            id="local",
            name="Local",
            metadata_backend=StorageBackendKind.FILESYSTEM,
            blob_backend="filesystem",
            scope=StorageScope.LOCAL,
            capabilities=StorageCapabilities(False, True, True, True, True, 100_000),
        )

    def get(self, profile_id):
        return self.profile if profile_id == self.profile.id else None

    def list(self):
        return (self.profile,)


class Events:
    def __init__(self):
        self.streams = {}
        self.all = []

    def current_revision(self, stream_id):
        return len(self.streams.get(stream_id, ()))

    def append(self, stream_id, *, expected_revision, events):
        actual = self.current_revision(stream_id)
        if actual != expected_revision:
            raise RuntimeError("revision conflict")
        assert all(event.revision == expected_revision + index for index, event in enumerate(events, 1))
        self.streams.setdefault(stream_id, []).extend(events)
        self.all.extend(events)
        return self.current_revision(stream_id)

    def load_stream(self, stream_id, *, after_revision=0, as_of=None):
        return tuple(
            event
            for event in self.streams.get(stream_id, ())
            if event.revision > after_revision and (as_of is None or event.recorded_at <= as_of)
        )

    def find_by_idempotency_key(self, project_id, idempotency_key):
        return tuple(
            event
            for stream in self.streams.values()
            for event in stream
            if event.project_id == project_id and event.idempotency_key == idempotency_key
        )


class Projections:
    def __init__(self):
        self.projects = {}
        self.layers = {}
        self.representations = {}
        self.series = {}
        self.versions = {}
        self.active_versions = {}
        self.batches = {}

    def get_project(self, identifier, *, as_of=None):
        return self.projects.get(identifier)

    def save_project(self, value):
        self.projects[value.id] = value

    def get_layer(self, identifier, *, as_of=None):
        return self.layers.get(identifier)

    def save_layer(self, value):
        self.layers[value.id] = value

    def list_layers(self, project_id, *, include_archived=False, as_of=None):
        return tuple(value for value in self.layers.values() if value.project_id == project_id)

    def get_representation(self, identifier, *, as_of=None):
        return self.representations.get(identifier)

    def save_representation(self, value):
        self.representations[value.id] = value

    def list_representations(self, layer_id, *, include_archived=False, as_of=None):
        return tuple(value for value in self.representations.values() if value.layer_id == layer_id)

    def get_artifact_series(self, identifier, *, as_of=None):
        return self.series.get(identifier)

    def save_artifact_series(self, value):
        self.series[value.id] = value

    def get_artifact_version(self, identifier, *, as_of=None):
        return self.versions.get(identifier)

    def get_active_artifact_version(self, series_id, *, as_of=None):
        identifier = self.active_versions.get(series_id)
        return None if identifier is None else self.versions[identifier]

    def save_artifact_version(self, value, *, activate):
        self.versions[value.id] = value
        if activate:
            self.active_versions[value.series_id] = value.id

    def save_plugin_job(self, value):
        raise NotImplementedError

    def get_plugin_job(self, identifier, *, as_of=None):
        return None

    def get_review_batch(self, identifier, *, as_of=None):
        return self.batches.get(identifier)

    def save_review_batch(self, value):
        self.batches[value.id] = value

    def list_active_review_batches(self, project_id, layer_id, *, as_of=None):
        return tuple(
            value
            for value in self.batches.values()
            if value.project_id == project_id
            and value.layer_id == layer_id
            and value.state not in {ReviewBatchState.COMPLETED, ReviewBatchState.CANCELLED}
        )


class Blobs:
    def __init__(self):
        self.data = {}

    def put(self, chunks, *, expected_sha256=None):
        content = b"".join(chunks)
        digest = sha256(content).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ValueError("hash mismatch")
        existed = digest in self.data
        self.data[digest] = content
        return StoredContent(BlobRef(digest, len(content)), existed)

    def iter_bytes(self, reference, *, chunk_size=1024 * 1024):
        yield self.data[reference.sha256]

    def exists(self, reference):
        return reference.sha256 in self.data


class Acl:
    def __init__(self):
        self.values = {}

    def grant(self, project_id, principal_id, role):
        self.values.setdefault((project_id, principal_id), set()).add(role)

    def roles_for(self, project_id, principal_id):
        return frozenset(self.values.get((project_id, principal_id), ()))

    def assign(self, assignment):
        self.grant(assignment.project_id, assignment.principal_id, assignment.role)

    def revoke(self, project_id, principal_id, role):
        self.values.get((project_id, principal_id), set()).discard(role)


class Identities:
    def get(self, identifier):
        return None

    def get_by_external_key(self, external_key):
        return None

    def save(self, value):
        pass


class Performers:
    def __init__(self, performer):
        self.performer = performer

    def get(self, identifier):
        return self.performer if self.performer.id == identifier else None

    def get_by_principal(self, principal_id):
        return None

    def list(self, *, include_archived=False):
        return (self.performer,) if include_archived or self.performer.active else ()

    def create(self, performer):
        self.performer = performer
        return performer

    def update(self, performer):
        self.performer = performer
        return performer

    def archive(self, performer_id):
        self.performer = self.performer.archive()
        return self.performer


class Uow:
    def __init__(self, fixture):
        self.event_store = fixture.events
        self.projections = fixture.projections
        self.blobs = fixture.blobs
        self.identities = Identities()
        self.acl = fixture.acl

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def commit(self):
        pass

    def rollback(self):
        pass


class Writer:
    def __init__(self, fail=False):
        self.fail = fail
        self.manifest = None
        self.files = {}

    def write(self, destination, manifest, files):
        if self.fail:
            raise OSError("USB write failed")
        self.manifest = manifest
        self.files = {path: b"".join(open_file()) for path, open_file in files.items()}


class Reader:
    def __init__(self, manifest, files):
        self.manifest = manifest
        self.files = dict(files)

    def read_manifest(self, source):
        return self.manifest

    def list_relative_paths(self, source):
        return tuple(self.files)

    def iter_file(self, source, relative_path):
        yield self.files[relative_path]


class Fixture:
    def __init__(self, *, gitlab=False):
        self.events = Events()
        self.projections = Projections()
        self.blobs = Blobs()
        self.acl = Acl()
        self.profiles = Profiles()
        self.clock = Clock()
        self.manager = (
            Principal.gitlab(
                issuer="https://gitlab.local",
                subject="manager",
                display_name="Manager",
            )
            if gitlab
            else Principal.local(subject="manager", display_name="Manager")
        )
        self.reviewer = Principal.local(subject="reviewer", display_name="Reviewer")
        self.project = replace(
            Project.create(
                name="Chip",
                width=2,
                height=1,
                orientation=GridOrientation.Y_DOWN,
                storage_profile="local",
                project_id=ProjectId(str(uuid4())),
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
        self.vector_representation = Representation.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            name="Vectors",
            kind=RepresentationKind.VECTOR,
            created_at=NOW,
        )
        self.projections.save_project(self.project)
        self.projections.save_layer(self.layer)
        self.projections.save_representation(self.vector_representation)
        self.acl.grant(self.project.id, self.manager.id, ProjectRole.MANAGER)
        self.acl.grant(self.project.id, self.reviewer.id, ProjectRole.REVIEWER)
        self.performer_id = PerformerId(str(uuid4()))
        self.performers = Performers(
            Performer.create(name="Reviewer", color="#336699", performer_id=self.performer_id)
        )

        self.items = []
        for x, content in ((1, b"base-one"), (2, b"base-two")):
            frame_id = self.project.frame_id_at(x, 1)
            series = ArtifactSeries.for_frame(
                project_id=self.project.id,
                layer_id=self.layer.id,
                representation_id=self.vector_representation.id,
                frame_id=frame_id,
                name=f"{x}_1.cif",
            )
            stored = self.blobs.put((content,))
            version = ArtifactVersion.managed(
                series_id=series.id,
                blob=stored.blob,
                media_type="application/x-cif",
                filename=f"{x}_1.cif",
                author_principal_id=self.manager.id,
                created_at=NOW,
            )
            self.projections.save_artifact_series(series)
            self.projections.save_artifact_version(version, activate=True)
            self.items.append(ReviewItem(frame_id, version.id, version.sha256))

    def factory(self):
        return Uow(self)

    def context(self, actor, key, *, verified=False):
        return CommandContext(
            actor=actor,
            idempotency_key=key,
            gitlab_identity_verified=verified,
        )

    def create_batch(self, *, count=1, key="create", verified=False):
        selection = FrameSelectionV1.rectangle(1, 1, count, 1)
        command = CreateReviewBatchCommand(
            context=self.context(self.manager, key, verified=verified),
            project_id=self.project.id,
            layer_id=self.layer.id,
            selection=selection,
            items=tuple(self.items[:count]),
            assignee_id=self.performer_id,
            expected_layer_revision=self.projections.get_layer(self.layer.id).revision,
        )
        return CreateReviewBatchHandler(
            self.factory, self.profiles, self.clock, self.performers
        )(command)


class ReviewWorkflowTests(unittest.TestCase):
    def test_create_rejects_overlapping_active_vector_version(self):
        fixture = Fixture()
        fixture.create_batch()
        with self.assertRaisesRegex(ConflictError, "already in an active"):
            fixture.create_batch(key="second")

    def test_failed_export_does_not_issue_and_successful_export_does(self):
        fixture = Fixture()
        batch = fixture.create_batch()
        plan = PlanReviewPackageHandler(fixture.factory, fixture.profiles, fixture.clock)(
            PlanReviewPackageCommand(
                fixture.context(fixture.manager, "plan"),
                fixture.project.id,
                batch.id,
                0,
            )
        )
        self.assertTrue(plan.can_export)
        command = ExportReviewPackageCommand(
            fixture.context(fixture.manager, "issue"),
            fixture.project.id,
            batch.id,
            0,
            "usb",
            package_id=plan.manifest.package_id,
        )
        with self.assertRaises(OSError):
            ExportReviewPackageHandler(
                fixture.factory, fixture.profiles, fixture.clock, Writer(fail=True)
            )(command)
        self.assertIs(fixture.projections.get_review_batch(batch.id).state, ReviewBatchState.DRAFT)

        writer = Writer()
        issued = ExportReviewPackageHandler(
            fixture.factory, fixture.profiles, fixture.clock, writer
        )(command)
        self.assertIs(issued.state, ReviewBatchState.ISSUED)
        self.assertEqual(issued.revision, 1)
        self.assertEqual(next(iter(writer.files.values())), b"base-one")

    def test_export_rejects_reused_key_before_writing(self):
        fixture = Fixture()
        batch = fixture.create_batch(key="already-used")
        writer = Writer()
        with self.assertRaisesRegex(ConflictError, "Idempotency key"):
            ExportReviewPackageHandler(
                fixture.factory, fixture.profiles, fixture.clock, writer
            )(
                ExportReviewPackageCommand(
                    fixture.context(fixture.manager, "already-used"),
                    fixture.project.id,
                    batch.id,
                    0,
                    "usb",
                )
            )
        self.assertIsNone(writer.manifest)

    def test_changed_return_creates_candidate_then_manager_accepts(self):
        fixture = Fixture()
        batch = fixture.create_batch()
        writer = Writer()
        issued = ExportReviewPackageHandler(
            fixture.factory, fixture.profiles, fixture.clock, writer
        )(
            ExportReviewPackageCommand(
                fixture.context(fixture.manager, "issue"),
                fixture.project.id,
                batch.id,
                0,
                "usb",
            )
        )
        path = next(iter(writer.files))
        reader = Reader(writer.manifest, {path: b"reviewed-change"})
        dry_run = DryRunReviewReturnHandler(
            fixture.factory, fixture.profiles, fixture.clock, reader
        )(
            DryRunReviewReturnCommand(
                fixture.context(fixture.reviewer, "dry-run"),
                fixture.project.id,
                issued.id,
                1,
                "usb",
            )
        )
        self.assertEqual(dry_run.report.count(ReviewFileStatus.CHANGED), 1)

        commit_command = CommitReviewReturnCommand(
            fixture.context(fixture.reviewer, "return"),
            fixture.project.id,
            issued.id,
            1,
            "usb",
        )
        result = CommitReviewReturnHandler(
            fixture.factory, fixture.profiles, fixture.clock, reader
        )(commit_command)
        self.assertIs(result.batch.state, ReviewBatchState.AWAITING_ACCEPTANCE)
        self.assertEqual(len(result.candidate_versions), 1)
        candidate = result.candidate_versions[0]
        base = fixture.projections.get_artifact_version(fixture.items[0].vector_version_id)
        self.assertEqual(candidate.parent_version_id, base.id)
        self.assertEqual(fixture.projections.get_active_artifact_version(base.series_id).id, base.id)

        repeated = CommitReviewReturnHandler(
            fixture.factory, fixture.profiles, fixture.clock, reader
        )(commit_command)
        self.assertEqual(tuple(item.id for item in repeated.candidate_versions), (candidate.id,))

        accepted = AcceptReviewHandler(fixture.factory, fixture.profiles, fixture.clock)(
            AcceptReviewCommand(
                fixture.context(fixture.manager, "accept"),
                fixture.project.id,
                batch.id,
                2,
                (candidate.id,),
            )
        )
        self.assertIs(accepted.state, ReviewBatchState.COMPLETED)
        self.assertEqual(fixture.projections.get_active_artifact_version(base.series_id).id, candidate.id)

        rebuilt = Projections()
        projector = ProjectionRebuilder(rebuilt)
        for event in fixture.events.all:
            projector.apply(event)
        self.assertIs(rebuilt.get_review_batch(batch.id).state, ReviewBatchState.COMPLETED)
        self.assertEqual(rebuilt.get_active_artifact_version(base.series_id).id, candidate.id)

    def test_manager_can_request_changes_without_activating_candidate(self):
        fixture = Fixture()
        batch = fixture.create_batch()
        writer = Writer()
        ExportReviewPackageHandler(fixture.factory, fixture.profiles, fixture.clock, writer)(
            ExportReviewPackageCommand(
                fixture.context(fixture.manager, "issue"), fixture.project.id, batch.id, 0, "usb"
            )
        )
        path = next(iter(writer.files))
        reader = Reader(writer.manifest, {path: b"different"})
        returned = CommitReviewReturnHandler(
            fixture.factory, fixture.profiles, fixture.clock, reader
        )(
            CommitReviewReturnCommand(
                fixture.context(fixture.reviewer, "return"), fixture.project.id, batch.id, 1, "usb"
            )
        )
        changed = RequestReviewChangesHandler(fixture.factory, fixture.profiles, fixture.clock)(
            RequestReviewChangesCommand(
                fixture.context(fixture.manager, "changes"),
                fixture.project.id,
                batch.id,
                returned.batch.revision,
                "Fix the contour",
            )
        )
        self.assertIs(changed.state, ReviewBatchState.CHANGES_REQUESTED)
        base = fixture.projections.get_artifact_version(fixture.items[0].vector_version_id)
        self.assertEqual(fixture.projections.get_active_artifact_version(base.series_id).id, base.id)

    def test_partial_return_resumes_without_duplicate_candidate(self):
        fixture = Fixture()
        batch = fixture.create_batch(count=2)
        writer = Writer()
        ExportReviewPackageHandler(fixture.factory, fixture.profiles, fixture.clock, writer)(
            ExportReviewPackageCommand(
                fixture.context(fixture.manager, "issue"), fixture.project.id, batch.id, 0, "usb"
            )
        )
        paths = tuple(writer.files)
        first_reader = Reader(writer.manifest, {paths[0]: b"changed-first"})
        first = CommitReviewReturnHandler(
            fixture.factory, fixture.profiles, fixture.clock, first_reader
        )(
            CommitReviewReturnCommand(
                fixture.context(fixture.reviewer, "partial"), fixture.project.id, batch.id, 1, "usb"
            )
        )
        self.assertIs(first.batch.state, ReviewBatchState.PARTIALLY_RETURNED)
        self.assertEqual(len(first.candidate_versions), 1)

        complete_reader = Reader(
            writer.manifest,
            {paths[0]: b"changed-first", paths[1]: writer.files[paths[1]]},
        )
        completed_return = CommitReviewReturnHandler(
            fixture.factory, fixture.profiles, fixture.clock, complete_reader
        )(
            CommitReviewReturnCommand(
                fixture.context(fixture.reviewer, "complete-return"),
                fixture.project.id,
                batch.id,
                2,
                "usb",
            )
        )
        self.assertIs(completed_return.batch.state, ReviewBatchState.AWAITING_ACCEPTANCE)
        self.assertEqual(
            tuple(value.id for value in completed_return.candidate_versions),
            tuple(value.id for value in first.candidate_versions),
        )
        candidate_events = [
            event
            for event in fixture.events.all
            if event.event_type == "ArtifactVersionCreated" and event.payload.get("candidate")
        ]
        self.assertEqual(len(candidate_events), 1)

    def test_stale_unchanged_return_is_an_inactive_conflict_candidate(self):
        fixture = Fixture()
        batch = fixture.create_batch()
        writer = Writer()
        ExportReviewPackageHandler(fixture.factory, fixture.profiles, fixture.clock, writer)(
            ExportReviewPackageCommand(
                fixture.context(fixture.manager, "issue"), fixture.project.id, batch.id, 0, "usb"
            )
        )
        base = fixture.projections.get_artifact_version(fixture.items[0].vector_version_id)
        newer_blob = fixture.blobs.put((b"newer-active",)).blob
        newer = ArtifactVersion.managed(
            series_id=base.series_id,
            blob=newer_blob,
            media_type=base.media_type,
            filename=base.filename,
            author_principal_id=fixture.manager.id,
            created_at=NOW,
            parent_version_id=base.id,
        )
        fixture.projections.save_artifact_version(newer, activate=True)
        path = next(iter(writer.files))
        reader = Reader(writer.manifest, {path: writer.files[path]})
        result = CommitReviewReturnHandler(
            fixture.factory, fixture.profiles, fixture.clock, reader
        )(
            CommitReviewReturnCommand(
                fixture.context(fixture.reviewer, "stale-return"),
                fixture.project.id,
                batch.id,
                1,
                "usb",
            )
        )
        self.assertEqual(result.report.count(ReviewFileStatus.STALE_BASE_CONFLICT), 1)
        self.assertEqual(len(result.candidate_versions), 1)
        self.assertEqual(result.candidate_versions[0].sha256, base.sha256)
        self.assertEqual(fixture.projections.get_active_artifact_version(base.series_id).id, newer.id)

    def test_gitlab_mutation_requires_live_verification(self):
        fixture = Fixture(gitlab=True)
        with self.assertRaises(AuthorizationError) as raised:
            fixture.create_batch()
        self.assertEqual(raised.exception.code, "gitlab_live_check_required")
        created = fixture.create_batch(key="verified", verified=True)
        self.assertIs(created.state, ReviewBatchState.DRAFT)

    def test_repeated_partial_return_advances_revision(self):
        fixture = Fixture()
        batch = fixture.create_batch(count=2).issue(at=NOW)
        first = batch.register_return(has_missing=True, has_changed=False, at=NOW)
        second = first.register_return(has_missing=True, has_changed=False, at=NOW)
        self.assertEqual((first.revision, second.revision), (2, 3))


if __name__ == "__main__":
    unittest.main()
