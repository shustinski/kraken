from __future__ import annotations

import hashlib
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from kraken_manager.domain import (
    ActorSnapshot,
    ArtifactSeries,
    ArtifactVersion,
    BlobRef,
    DomainValidationError,
    EventEnvelope,
    ExternalReference,
    FrameCoordinate,
    FrameRectangle,
    FrameRowRange,
    FrameSelectionV1,
    FrameStatus,
    FrameStatusFacts,
    GridOrientation,
    InvalidStateTransition,
    Layer,
    LayerType,
    PluginJob,
    PluginJobState,
    Principal,
    Project,
    Representation,
    RepresentationKind,
    ReviewBatch,
    ReviewBatchState,
    ReviewItem,
    derive_frame_status,
)
from kraken_manager.domain.common import ArtifactVersionId, PerformerId


NOW = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


class ProjectDomainTests(unittest.TestCase):
    def test_project_grid_is_validated_frozen_and_frame_ids_are_deterministic(self) -> None:
        project = Project.create(
            name="Chip A",
            width=100,
            height=80,
            orientation=GridOrientation.Y_UP,
            storage_profile="local",
            created_at=NOW,
        )

        self.assertEqual(project.frame_count, 8_000)
        self.assertEqual(project.frame_id_at(4, 7), project.frame_id_at(4, 7))
        self.assertNotEqual(project.frame_id_at(4, 7), project.frame_id_at(5, 7))
        with self.assertRaises(DomainValidationError):
            project.coordinate(101, 1)
        with self.assertRaises(FrozenInstanceError):
            project.width = 101  # type: ignore[misc]

    def test_layer_types_are_closed_and_revisions_are_optimistic(self) -> None:
        project = Project.create(name="P", width=2, height=2, storage_profile="local", created_at=NOW)
        layer = Layer.create(project_id=project.id, name="M1", type=LayerType.METAL, order=0, created_at=NOW)

        renamed = layer.rename("M2", expected_revision=0)

        self.assertEqual(renamed.revision, 1)
        with self.assertRaises(DomainValidationError):
            renamed.rename("M3", expected_revision=0)
        with self.assertRaises(ValueError):
            Layer.create(project_id=project.id, name="X", type="custom", order=1)  # type: ignore[arg-type]

    def test_representation_is_sparse_and_archiving_deactivates_it(self) -> None:
        project = Project.create(name="P", width=2, height=2, storage_profile="local", created_at=NOW)
        layer = Layer.create(project_id=project.id, name="D", type=LayerType.DIFFUSION, order=0, created_at=NOW)
        representation = Representation.create(
            project_id=project.id,
            layer_id=layer.id,
            name="Originals",
            kind=RepresentationKind.IMAGE,
            active=True,
            created_at=NOW,
        )

        archived = representation.archive()

        self.assertFalse(archived.active)
        with self.assertRaises(DomainValidationError):
            archived.activate()


class SelectionTests(unittest.TestCase):
    def test_selection_merges_overlaps_and_applies_exclusions(self) -> None:
        selection = FrameSelectionV1(
            rectangles=(FrameRectangle(1, 1, 3, 2),),
            row_ranges=(FrameRowRange(y=2, x_start=3, x_end=5),),
            exclusions=frozenset({FrameCoordinate(2, 1)}),
        )

        self.assertEqual(selection.cardinality(), 7)
        self.assertFalse(selection.contains(FrameCoordinate(2, 1)))
        self.assertEqual(selection.intervals_for_row(2), ((1, 5),))
        self.assertEqual(FrameSelectionV1.from_dict(selection.to_dict()), selection)
        with self.assertRaises(DomainValidationError):
            selection.validate_bounds(width=4, height=2)

    def test_exclusion_must_first_be_included(self) -> None:
        with self.assertRaises(DomainValidationError):
            FrameSelectionV1(
                rectangles=(FrameRectangle(1, 1, 2, 2),),
                exclusions=frozenset({FrameCoordinate(3, 3)}),
            )


class ArtifactAndEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.principal = Principal.local(subject="alice", display_name="Alice")
        self.project = Project.create(name="P", width=2, height=2, storage_profile="local", created_at=NOW)
        self.layer = Layer.create(
            project_id=self.project.id,
            name="M1",
            type=LayerType.METAL,
            order=0,
            created_at=NOW,
        )
        self.representation = Representation.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            name="Vectors",
            kind=RepresentationKind.VECTOR,
            created_at=NOW,
        )
        self.series = ArtifactSeries.for_frame(
            project_id=self.project.id,
            layer_id=self.layer.id,
            representation_id=self.representation.id,
            frame_id=self.project.frame_id_at(1, 1),
            name="1_1.cif",
        )

    def test_managed_version_is_content_addressed_and_metadata_is_frozen(self) -> None:
        content = b"CIF bytes"
        digest = hashlib.sha256(content).hexdigest()
        version = ArtifactVersion.managed(
            series_id=self.series.id,
            blob=BlobRef(digest, len(content)),
            media_type="application/x-cif",
            filename="1_1.cif",
            author_principal_id=self.principal.id,
            created_at=NOW,
            parameters={"threshold": 0.5, "steps": [1, 2]},
        )

        self.assertTrue(version.managed_content)
        self.assertEqual(version.blob.storage_key, f"sha256/{digest[:2]}/{digest[2:4]}/{digest}")
        with self.assertRaises(TypeError):
            version.parameters["threshold"] = 1  # type: ignore[index]

    def test_external_version_explicitly_does_not_guarantee_content_history(self) -> None:
        content = b"stitch"
        reference = ExternalReference(
            uri="file:///media/review/stitch.bdt",
            fingerprint_sha256=hashlib.sha256(content).hexdigest(),
            observed_size_bytes=len(content),
        )
        version = ArtifactVersion.external_link(
            series_id=self.series.id,
            reference=reference,
            media_type="application/octet-stream",
            filename="stitch.bdt",
            author_principal_id=self.principal.id,
            created_at=NOW,
        )

        self.assertFalse(version.managed_content)
        self.assertFalse(reference.history_is_guaranteed)

    def test_event_envelope_preserves_actor_and_freezes_payload(self) -> None:
        event = EventEnvelope.create(
            stream_id=f"project:{self.project.id}",
            project_id=self.project.id,
            revision=1,
            event_type="ProjectCreated",
            payload={"name": "P", "nested": {"value": 1}},
            actor=ActorSnapshot.from_principal(self.principal),
            recorded_at=NOW.astimezone(timezone(timedelta(hours=3))),
            idempotency_key="create-p",
        )

        self.assertEqual(event.recorded_at.tzinfo, timezone.utc)
        self.assertEqual(event.actor.display_name, "Alice")
        with self.assertRaises(TypeError):
            event.payload["name"] = "changed"  # type: ignore[index]


class WorkflowStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.principal = Principal.local(subject="alice", display_name="Alice")
        self.project = Project.create(name="P", width=2, height=2, storage_profile="local", created_at=NOW)
        self.layer = Layer.create(
            project_id=self.project.id,
            name="M1",
            type=LayerType.METAL,
            order=0,
            created_at=NOW,
        )
        self.representation = Representation.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            name="Vectors",
            kind=RepresentationKind.VECTOR,
            created_at=NOW,
        )
        self.selection = FrameSelectionV1.rectangle(1, 1, 2, 1)

    def test_plugin_job_enforces_transition_graph_and_terminal_state(self) -> None:
        job = PluginJob.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            selection=self.selection,
            actor_principal_id=self.principal.id,
            target_representation_id=self.representation.id,
            capability="frames.vectorize",
            created_at=NOW,
        )

        running = job.transition(PluginJobState.STAGING, at=NOW).transition(
            PluginJobState.RUNNING, at=NOW, progress=0.3
        )
        finished = running.transition(PluginJobState.IMPORTING, at=NOW).transition(
            PluginJobState.SUCCEEDED, at=NOW
        )

        self.assertEqual(finished.progress, 1.0)
        self.assertIsNotNone(finished.finished_at)
        with self.assertRaises(InvalidStateTransition):
            finished.transition(PluginJobState.RUNNING)

    def test_frame_status_is_derived_per_layer_frame_with_error_priority(self) -> None:
        self.assertIs(
            derive_frame_status(FrameStatusFacts(has_image=True, has_vector=True)),
            FrameStatus.VECTORIZED,
        )
        self.assertIs(
            derive_frame_status(FrameStatusFacts(has_vector=True, in_review=True, error=True)),
            FrameStatus.ERROR,
        )

    def test_review_batch_requires_acceptance_for_changed_return(self) -> None:
        item = ReviewItem(
            frame_id=self.project.frame_id_at(1, 1),
            vector_version_id=ArtifactVersionId("03ee6e1c-9f00-4b45-b277-94af6c623e48"),
            vector_sha256="a" * 64,
        )
        batch = ReviewBatch.create(
            project_id=self.project.id,
            layer_id=self.layer.id,
            selection=self.selection,
            items=(item,),
            assignee_id=PerformerId("3660cd8b-160f-40bd-b2eb-4db0d9b75d0d"),
            created_by=self.principal.id,
            created_at=NOW,
        )

        returned = batch.issue(at=NOW).register_return(has_missing=False, has_changed=True, at=NOW)

        self.assertIs(returned.state, ReviewBatchState.AWAITING_ACCEPTANCE)
        self.assertIs(returned.transition(ReviewBatchState.COMPLETED, at=NOW).state, ReviewBatchState.COMPLETED)


if __name__ == "__main__":
    unittest.main()
