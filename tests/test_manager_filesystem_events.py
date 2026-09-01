from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from kraken_manager.application.dto import CommandContext, CreateLayerCommand, CreateProjectCommand
from kraken_manager.application.ports import EventStore, ProjectionStore, UnitOfWork
from kraken_manager.application.use_cases import CreateLayerHandler, CreateProjectHandler
from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion, BlobRef
from kraken_manager.domain.common import PerformerId, PrincipalId
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Principal, PrincipalProvider
from kraken_manager.domain.project import GridOrientation, Layer, LayerType, Project, Representation, RepresentationKind
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import PluginJob, ReviewBatch, ReviewItem
from kraken_manager.infrastructure.filesystem import (
    CorruptEventLogError,
    EventStreamConflict,
    FilesystemEventStore,
    ProjectFileLock,
    ProjectLockTimeout,
    ProjectionMutation,
    SQLiteProjectionStore,
    LocalProjectUnitOfWorkFactory,
    filesystem_storage_profile,
)


def _actor() -> ActorSnapshot:
    return ActorSnapshot(
        principal_id=str(uuid4()),
        provider=PrincipalProvider.LOCAL,
        subject="local:test",
        display_name="Test User",
    )


def _event(project_id: str, stream_id: str, revision: int, *, at: datetime, key: str) -> EventEnvelope:
    return EventEnvelope.create(
        project_id=project_id,
        stream_id=stream_id,
        revision=revision,
        event_type="TestEvent",
        payload={"revision": revision},
        actor=_actor(),
        recorded_at=at,
        idempotency_key=key,
    )


class FilesystemEventStoreTests(unittest.TestCase):
    def test_domain_event_round_trip_concurrency_and_temporal_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_id = str(uuid4())
            stream_id = f"project:{project_id}"
            store = FilesystemEventStore(directory, project_id)
            self.assertIsInstance(store, EventStore)
            first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
            second_time = first_time + timedelta(hours=1)
            first = _event(project_id, stream_id, 1, at=first_time, key="create")
            second = _event(project_id, stream_id, 2, at=second_time, key="rename")

            self.assertEqual(store.append(stream_id, expected_revision=0, events=(first,)), 1)
            self.assertEqual(store.append(stream_id, expected_revision=1, events=(second,)), 2)
            loaded = store.load_stream(stream_id)
            self.assertEqual(loaded, (first, second))
            self.assertEqual(
                store.load_stream(stream_id, as_of=first_time + timedelta(minutes=1)),
                (first,),
            )
            self.assertEqual(store.find_by_idempotency_key(None, "rename"), (second,))
            self.assertEqual(store.find_by_idempotency_key(str(uuid4()), "rename"), ())

            with self.assertRaises(EventStreamConflict):
                store.append(stream_id, expected_revision=1, events=(second,))
            segments = list(store.layout.events_dir.glob("*.jsonl"))
            self.assertEqual(len(segments), 2)
            self.assertTrue(all(path.read_bytes().endswith(b"\n") for path in segments))

    def test_corruption_is_detected_and_project_lock_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_id = str(uuid4())
            store = FilesystemEventStore(directory, project_id)
            stream_id = f"project:{project_id}"
            event = _event(
                project_id,
                stream_id,
                1,
                at=datetime.now(timezone.utc),
                key="one",
            )
            store.append(stream_id, expected_revision=0, events=(event,))
            segment = next(store.layout.events_dir.glob("*.jsonl"))
            segment.write_bytes(segment.read_bytes().rstrip(b"\n"))
            with self.assertRaises(CorruptEventLogError):
                list(store.iter_project())

            first = ProjectFileLock(Path(directory) / "separate.lock")
            second = ProjectFileLock(Path(directory) / "separate.lock", poll_interval=0.001)
            with first.hold():
                with self.assertRaises(ProjectLockTimeout):
                    second.acquire(timeout=0.005)

    def test_last_global_position_uses_segment_ranges_without_reading_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_id = str(uuid4())
            store = FilesystemEventStore(directory, project_id)
            stream_id = f"project:{project_id}"
            event = _event(
                project_id,
                stream_id,
                1,
                at=datetime.now(timezone.utc),
                key="one",
            )
            store.append(stream_id, expected_revision=0, events=(event,))

            segment = next(store.layout.events_dir.glob("*.jsonl"))
            segment.write_text("not read by the position lookup", encoding="utf-8")

            self.assertEqual(store.last_global_position(), 1)
            with self.assertRaises(CorruptEventLogError):
                list(store.iter_project())

    def test_last_global_position_rejects_segment_range_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_id = str(uuid4())
            store = FilesystemEventStore(directory, project_id)
            segment = store.layout.events_dir / "00000000000000000002-00000000000000000002.jsonl"
            segment.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(CorruptEventLogError, "expected 1, found 2"):
                store.last_global_position()


class SQLiteProjectionStoreTests(unittest.TestCase):
    def test_typed_crud_temporal_history_and_rebuildable_event_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_id = str(uuid4())
            projections = SQLiteProjectionStore(directory, project_id)
            self.assertIsInstance(projections, ProjectionStore)
            project = Project.create(
                project_id=project_id,
                name="Before",
                width=10,
                height=20,
                storage_profile="local-filesystem",
                created_at=datetime.now(timezone.utc),
            )
            projections.save_project(project)
            midpoint = datetime.now(timezone.utc)
            renamed = replace(project, name="After", revision=1)
            projections.save_project(renamed)
            self.assertEqual(projections.get_project(project.id), renamed)
            self.assertEqual(projections.get_project(project.id, as_of=midpoint), project)

            layer = Layer.create(project_id=project.id, name="Metal", type=LayerType.METAL, order=2)
            projections.save_layer(layer)
            self.assertEqual(projections.list_layers(project.id), (layer,))
            projections.save_layer(layer.archive())
            self.assertEqual(projections.list_layers(project.id), ())
            self.assertEqual(projections.list_layers(project.id, include_archived=True)[0].state.value, "archived")

            representation = Representation.create(
                project_id=project.id,
                layer_id=layer.id,
                name="Raw",
                kind=RepresentationKind.IMAGE,
                active=True,
            )
            projections.save_representation(representation)
            self.assertEqual(projections.get_representation(representation.id), representation)
            self.assertEqual(projections.list_representations(layer.id), (representation,))

            frame_id = project.frame_id_at(1, 1)
            series = ArtifactSeries.for_frame(
                project_id=project.id,
                layer_id=layer.id,
                representation_id=representation.id,
                frame_id=frame_id,
                name="1_1.png",
            )
            projections.save_artifact_series(series)
            self.assertEqual(projections.get_artifact_series(series.id), series)
            digest = "a" * 64
            version = ArtifactVersion.managed(
                series_id=series.id,
                blob=BlobRef(digest, 12),
                media_type="image/png",
                filename="1_1.png",
                author_principal_id=PrincipalId(str(uuid4())),
            )
            projections.save_artifact_version(version, activate=True)
            self.assertEqual(projections.get_artifact_version(version.id), version)
            self.assertEqual(projections.get_active_artifact_version(series.id), version)

            selection = FrameSelectionV1.rectangle(1, 1, 2, 2)
            principal_id = PrincipalId(str(uuid4()))
            job = PluginJob.create(
                project_id=project.id,
                layer_id=layer.id,
                selection=selection,
                actor_principal_id=principal_id,
                target_representation_id=representation.id,
                capability="contour.vectorize",
            )
            projections.save_plugin_job(job)
            self.assertEqual(projections.get_plugin_job(job.id), job)
            batch = ReviewBatch.create(
                project_id=project.id,
                layer_id=layer.id,
                selection=selection,
                items=(ReviewItem(frame_id=frame_id, vector_version_id=version.id, vector_sha256=digest),),
                assignee_id=PerformerId(str(uuid4())),
                created_by=principal_id,
            )
            projections.save_review_batch(batch)
            self.assertEqual(projections.get_review_batch(batch.id), batch)
            self.assertEqual(projections.list_active_review_batches(project.id, layer.id), (batch,))

            stored_event = {
                "global_position": 1,
                "event_id": str(uuid4()),
                "project_id": project_id,
                "stream_id": f"project:{project_id}",
                "revision": 1,
                "event_type": "ProjectCreated",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            projections.apply(
                stored_event,
                (ProjectionMutation("catalog", project_id, {"name": "Before"}),),
            )
            self.assertEqual(projections.get("catalog", project_id), {"name": "Before"})
            projections.destroy_cache()
            self.assertFalse(projections.path.exists())
            count = projections.rebuild(
                [stored_event],
                lambda event: (ProjectionMutation("catalog", project_id, {"name": "Rebuilt"}),),
            )
            self.assertEqual(count, 1)
            self.assertEqual(projections.get("catalog", project_id), {"name": "Rebuilt"})


class _MemoryIdentityStore:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    def get(self, principal_id):
        return self.principal if principal_id == self.principal.id else None

    def get_by_external_key(self, external_key):
        return self.principal if external_key == self.principal.external_key else None

    def save(self, principal):
        self.principal = principal


class _MemoryAclStore:
    def __init__(self) -> None:
        self.assignments = set()

    def roles_for(self, project_id, principal_id):
        return frozenset(
            role
            for stored_project, stored_principal, role in self.assignments
            if stored_project == project_id and stored_principal == principal_id
        )

    def assign(self, assignment):
        self.assignments.add((assignment.project_id, assignment.principal_id, assignment.role))

    def revoke(self, project_id, principal_id, role):
        self.assignments.discard((project_id, principal_id, role))


class _ProfileCatalog:
    def __init__(self, profile) -> None:
        self.profile = profile

    def get(self, profile_id):
        return self.profile if profile_id == self.profile.id else None

    def list(self):
        return (self.profile,)


class _FixedClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value


class LocalProjectUnitOfWorkTests(unittest.TestCase):
    def test_real_create_project_and_layer_vertical_slice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_id = str(uuid4())
            principal = Principal.local(subject="owner", display_name="Owner")
            identities = _MemoryIdentityStore(principal)
            acl = _MemoryAclStore()
            factory = LocalProjectUnitOfWorkFactory(
                directory,
                project_id,
                identities=identities,
                acl=acl,
            )
            with factory() as sample:
                self.assertIsInstance(sample, UnitOfWork)
            profile = filesystem_storage_profile()
            profiles = _ProfileCatalog(profile)
            clock = _FixedClock(datetime(2026, 2, 3, tzinfo=timezone.utc))
            context = CommandContext(actor=principal, idempotency_key="create-project")
            create = CreateProjectHandler(factory, profiles, clock)
            project = create(
                CreateProjectCommand(
                    context=context,
                    name="Vertical",
                    width=5,
                    height=4,
                    orientation=GridOrientation.Y_DOWN,
                    storage_profile_id=profile.id,
                    project_id=project_id,
                )
            )
            self.assertEqual(project.revision, 1)
            self.assertEqual(create(
                CreateProjectCommand(
                    context=context,
                    name="Vertical",
                    width=5,
                    height=4,
                    orientation=GridOrientation.Y_DOWN,
                    storage_profile_id=profile.id,
                    project_id=project_id,
                )
            ), project)

            layer = CreateLayerHandler(factory, profiles, clock)(
                CreateLayerCommand(
                    context=CommandContext(actor=principal, idempotency_key="create-layer"),
                    project_id=project.id,
                    name="Metal",
                    type=LayerType.METAL,
                    order=0,
                    expected_project_revision=1,
                )
            )
            persisted_events = FilesystemEventStore(directory, project_id)
            self.assertEqual(persisted_events.current_revision(f"project:{project_id}"), 2)
            self.assertEqual(len(tuple(persisted_events.iter_project())), 2)
            persisted = SQLiteProjectionStore(directory, project_id)
            self.assertEqual(persisted.get_project(project.id).revision, 2)
            self.assertEqual(persisted.get_layer(layer.id), layer)
            self.assertTrue(persisted_events.layout.descriptor_path.exists())


if __name__ == "__main__":
    unittest.main()
