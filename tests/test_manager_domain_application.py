from __future__ import annotations

import hashlib
import unittest
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from uuid import uuid4

from kraken_manager.application import (
    AddArtifactVersionCommand,
    AddArtifactVersionHandler,
    AuthorizationPolicy,
    CommandContext,
    CreateLayerCommand,
    CreateLayerHandler,
    CreateProjectCommand,
    CreateProjectHandler,
    CreateRepresentationCommand,
    CreateRepresentationHandler,
    ReturnedFileDigest,
    ReviewReturnComparator,
    StorageBackendKind,
    StorageCapabilities,
    StorageProfile,
    StorageScope,
    StoredContent,
)
from kraken_manager.domain import (
    ArtifactSeries,
    BlobRef,
    FrameId,
    GridOrientation,
    LayerId,
    LayerType,
    Permission,
    Principal,
    ProjectId,
    ProjectRole,
    RepresentationKind,
    ReviewFileStatus,
    ReviewPackageFileV1,
    ReviewPackageManifestV1,
)
from kraken_manager.domain.common import ArtifactSeriesId, ArtifactVersionId, RepresentationId


NOW = datetime(2026, 4, 5, 8, tzinfo=timezone.utc)


class FakeClock:
    def now(self) -> datetime:
        return NOW


class FakeEventStore:
    def __init__(self) -> None:
        self.streams: dict[str, list[object]] = defaultdict(list)

    def load_stream(self, stream_id: str, *, after_revision: int = 0, as_of=None):
        return tuple(event for event in self.streams[stream_id] if event.revision > after_revision)

    def current_revision(self, stream_id: str) -> int:
        return len(self.streams[stream_id])

    def append(self, stream_id: str, *, expected_revision: int, events: Sequence[object]) -> int:
        if self.current_revision(stream_id) != expected_revision:
            raise RuntimeError("revision conflict")
        self.streams[stream_id].extend(events)
        return self.current_revision(stream_id)

    def find_by_idempotency_key(self, project_id: ProjectId | None, idempotency_key: str):
        events = (event for stream in self.streams.values() for event in stream)
        return tuple(
            event
            for event in events
            if event.idempotency_key == idempotency_key
            and (project_id is None or event.project_id == project_id)
        )


class FakeProjections:
    def __init__(self) -> None:
        self.projects = {}
        self.layers = {}
        self.representations = {}
        self.series = {}
        self.versions = {}
        self.active_versions = {}

    def get_project(self, project_id, *, as_of=None):
        return self.projects.get(project_id)

    def save_project(self, project) -> None:
        self.projects[project.id] = project

    def get_layer(self, layer_id, *, as_of=None):
        return self.layers.get(layer_id)

    def list_layers(self, project_id, *, include_archived=False):
        return tuple(item for item in self.layers.values() if item.project_id == project_id)

    def save_layer(self, layer) -> None:
        self.layers[layer.id] = layer

    def get_representation(self, representation_id, *, as_of=None):
        return self.representations.get(representation_id)

    def list_representations(self, layer_id, *, include_archived=False):
        return tuple(item for item in self.representations.values() if item.layer_id == layer_id)

    def save_representation(self, representation) -> None:
        self.representations[representation.id] = representation

    def get_artifact_series(self, series_id):
        return self.series.get(series_id)

    def save_artifact_series(self, series) -> None:
        self.series[series.id] = series

    def get_artifact_version(self, version_id):
        return self.versions.get(version_id)

    def get_active_artifact_version(self, series_id):
        version_id = self.active_versions.get(series_id)
        return None if version_id is None else self.versions[version_id]

    def save_artifact_version(self, version, *, activate: bool) -> None:
        self.versions[version.id] = version
        if activate:
            self.active_versions[version.series_id] = version.id

    def save_plugin_job(self, job) -> None:
        pass

    def get_review_batch(self, batch_id):
        return None

    def save_review_batch(self, batch) -> None:
        pass

    def list_active_review_batches(self, project_id, layer_id):
        return ()


class FakeBlobStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def put(self, chunks: Iterable[bytes], *, expected_sha256: str | None = None) -> StoredContent:
        value = b"".join(chunks)
        digest = hashlib.sha256(value).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("digest mismatch")
        existed = digest in self.values
        self.values[digest] = value
        return StoredContent(BlobRef(digest, len(value)), existed)

    def iter_bytes(self, reference: BlobRef, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        yield self.values[reference.sha256]

    def exists(self, reference: BlobRef) -> bool:
        return reference.sha256 in self.values


class FakeAcl:
    def __init__(self) -> None:
        self.assignments = []

    def roles_for(self, project_id, principal_id):
        return frozenset(
            item.role
            for item in self.assignments
            if item.project_id == project_id and item.principal_id == principal_id and item.active
        )

    def assign(self, assignment) -> None:
        self.assignments.append(assignment)

    def revoke(self, project_id, principal_id, role) -> None:
        pass


class FakeIdentities:
    def get(self, principal_id):
        return None

    def get_by_external_key(self, external_key):
        return None

    def save(self, principal) -> None:
        pass


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.event_store = FakeEventStore()
        self.projections = FakeProjections()
        self.blobs = FakeBlobStore()
        self.acl = FakeAcl()
        self.identities = FakeIdentities()
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        pass


class FakeUowFactory:
    def __init__(self) -> None:
        self.uow = FakeUnitOfWork()

    def __call__(self) -> FakeUnitOfWork:
        return self.uow


class FakeProfiles:
    def __init__(self, *profiles: StorageProfile) -> None:
        self.profiles = {profile.id: profile for profile in profiles}

    def get(self, profile_id: str):
        return self.profiles.get(profile_id)

    def list(self):
        return tuple(self.profiles.values())


def profile(scope: StorageScope, *, profile_id: str) -> StorageProfile:
    return StorageProfile(
        id=profile_id,
        name=profile_id,
        metadata_backend=(
            StorageBackendKind.FILESYSTEM if scope is StorageScope.LOCAL else StorageBackendKind.POSTGRESQL
        ),
        blob_backend="filesystem",
        scope=scope,
        capabilities=StorageCapabilities(
            multi_writer=scope is StorageScope.SHARED,
            transactions=True,
            snapshots=True,
            streaming=True,
            external_references=True,
            max_frames=100_000 if scope is StorageScope.LOCAL else 1_000_000,
        ),
    )


class AuthorizationPolicyTests(unittest.TestCase):
    def test_local_account_is_hard_denied_shared_mutation_even_as_owner(self) -> None:
        decision = AuthorizationPolicy().decide(
            principal=Principal.local(subject="alice", display_name="Alice"),
            storage=profile(StorageScope.SHARED, profile_id="shared"),
            permission=Permission.MANAGE_STRUCTURE,
            roles={ProjectRole.OWNER},
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "local_shared_mutation_denied")

    def test_authenticated_read_is_global_but_gitlab_shared_write_needs_live_check(self) -> None:
        principal = Principal.gitlab(
            issuer="https://gitlab.local", subject="42", display_name="Alice"
        )
        policy = AuthorizationPolicy()
        shared = profile(StorageScope.SHARED, profile_id="shared")

        self.assertTrue(
            policy.decide(
                principal=principal,
                storage=shared,
                permission=Permission.VIEW_PROJECT,
            ).allowed
        )
        self.assertFalse(
            policy.decide(
                principal=principal,
                storage=shared,
                permission=Permission.MANAGE_STRUCTURE,
                roles={ProjectRole.OWNER},
                gitlab_identity_verified=False,
            ).allowed
        )


class CommandHandlerTests(unittest.TestCase):
    def test_local_vertical_slice_creates_structure_and_immutable_artifact(self) -> None:
        actor = Principal.local(subject="alice", display_name="Alice")
        local = profile(StorageScope.LOCAL, profile_id="local")
        profiles = FakeProfiles(local)
        factory = FakeUowFactory()
        common = (factory, profiles, FakeClock())

        create_project = CreateProjectHandler(*common)
        project_command = CreateProjectCommand(
            context=CommandContext(actor=actor, idempotency_key="create-project"),
            name="Chip",
            width=10,
            height=10,
            orientation=GridOrientation.Y_DOWN,
            storage_profile_id="local",
            project_id=ProjectId(str(uuid4())),
        )
        project = create_project(project_command)
        self.assertEqual(project.revision, 1)
        self.assertEqual(create_project(project_command).id, project.id)

        layer = CreateLayerHandler(*common)(
            CreateLayerCommand(
                context=CommandContext(actor=actor, idempotency_key="create-layer"),
                project_id=project.id,
                name="Metal 1",
                type=LayerType.METAL,
                order=0,
                expected_project_revision=1,
            )
        )
        representation = CreateRepresentationHandler(*common)(
            CreateRepresentationCommand(
                context=CommandContext(actor=actor, idempotency_key="create-representation"),
                project_id=project.id,
                layer_id=layer.id,
                name="Vectors",
                kind=RepresentationKind.VECTOR,
                expected_layer_revision=0,
                active=True,
            )
        )
        series = ArtifactSeries.for_frame(
            project_id=project.id,
            layer_id=layer.id,
            representation_id=representation.id,
            frame_id=project.frame_id_at(1, 1),
            name="1_1.cif",
        )
        factory.uow.projections.save_artifact_series(series)
        content = b"immutable cif"
        version = AddArtifactVersionHandler(*common)(
            AddArtifactVersionCommand(
                context=CommandContext(actor=actor, idempotency_key="add-version"),
                project_id=project.id,
                series_id=series.id,
                filename="1_1.cif",
                media_type="application/x-cif",
                expected_series_revision=0,
            ),
            (content,),
        )

        self.assertEqual(version.sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(factory.uow.projections.get_active_artifact_version(series.id), version)


class ReviewComparatorTests(unittest.TestCase):
    def test_dry_run_classifies_hashes_missing_duplicates_extras_and_stale_base(self) -> None:
        project_id = ProjectId(str(uuid4()))
        layer_id = LayerId(str(uuid4()))
        frames = [FrameId(str(uuid4())) for _ in range(4)]
        versions = [ArtifactVersionId(str(uuid4())) for _ in range(4)]
        files = tuple(
            ReviewPackageFileV1(
                frame_id=frame,
                artifact_version_id=version,
                sha256=character * 64,
                relative_path=f"vectors/{index}.cif",
            )
            for index, (frame, version, character) in enumerate(zip(frames, versions, "abcd", strict=True))
        )
        manifest = ReviewPackageManifestV1(
            package_id=str(uuid4()),  # type: ignore[arg-type]
            project_id=project_id,
            layer_id=layer_id,
            issued_at=NOW,
            files=files,
        )
        returned = (
            ReturnedFileDigest("vectors/0.cif", "a" * 64),
            ReturnedFileDigest("vectors/1.cif", "f" * 64),
            ReturnedFileDigest("vectors/3.cif", "d" * 64),
            ReturnedFileDigest("extra.cif", "e" * 64),
            ReturnedFileDigest("EXTRA.cif", "e" * 64),
            ReturnedFileDigest("../escape.cif", "e" * 64),
        )
        report = ReviewReturnComparator().compare(
            manifest=manifest,
            returned_files=returned,
            active_versions={frames[3]: ArtifactVersionId(str(uuid4()))},
        )

        self.assertEqual(report.count(ReviewFileStatus.UNCHANGED), 1)
        self.assertEqual(report.count(ReviewFileStatus.CHANGED), 1)
        self.assertEqual(report.count(ReviewFileStatus.MISSING), 1)
        self.assertEqual(report.count(ReviewFileStatus.STALE_BASE_CONFLICT), 1)
        self.assertEqual(report.count(ReviewFileStatus.DUPLICATE), 1)
        self.assertEqual(report.count(ReviewFileStatus.INVALID), 1)
        self.assertFalse(report.can_commit)


if __name__ == "__main__":
    unittest.main()
