"""review_workflow: create review batch handler."""
from __future__ import annotations
from dataclasses import replace
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    CreateReviewBatchCommand,
)
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.ports import (
    Clock,
    PerformerStore,
    StorageProfileCatalog,
    UnitOfWork,
    UnitOfWorkFactory,
)
from kraken_manager.application.use_cases import (
    _layer_stream,
    _prior_entity_id,
)
from kraken_manager.domain.common import ReviewBatchId
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import RepresentationKind, StructureState
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewItem,
)

from kraken_manager.application.review_workflow_parts.review_handler import (
    _ReviewHandler,
    _batch_payload,
    _prior_command_event,
)

class CreateReviewBatchHandler(_ReviewHandler):
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        performer_store: PerformerStore,
        *,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        super().__init__(uow_factory, storage_profiles, clock, authorization)
        self._performers = performer_store

    def __call__(self, command: CreateReviewBatchCommand) -> ReviewBatch:
        with self._uow_factory() as uow:
            project, _ = self._load_project_and_authorize(
                uow,
                project_id=command.project_id,
                actor=command.context.actor,
                permission=Permission.MANAGE_REVIEW,
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            )
            previous_id = _prior_entity_id(
                uow,
                project.id,
                command.context.idempotency_key,
                "ReviewBatchCreated",
                "review_batch_id",
            )
            if previous_id is not None:
                previous = uow.projections.get_review_batch(ReviewBatchId(previous_id))
                if previous is not None:
                    return previous
            _prior_command_event(
                uow,
                project_id=project.id,
                idempotency_key=command.context.idempotency_key,
                event_type="ReviewBatchCreated",
            )
            layer = uow.projections.get_layer(command.layer_id)
            if layer is None or layer.project_id != project.id:
                raise NotFoundError("Layer was not found in the project")
            if layer.state is StructureState.ARCHIVED:
                raise ConflictError("Archived layer is read-only")
            if layer.revision != command.expected_layer_revision:
                raise ConcurrencyError(
                    f"Expected layer revision {command.expected_layer_revision}, found {layer.revision}"
                )
            command.selection.validate_bounds(width=project.width, height=project.height)
            now = self._clock.now()
            batch = ReviewBatch.create(
                project_id=project.id,
                layer_id=layer.id,
                selection=command.selection,
                items=tuple(command.items),
                assignee_id=command.assignee_id,
                created_by=command.context.actor.id,
                instructions=command.instructions,
                due_at=command.due_at,
                batch_id=command.batch_id,
                created_at=now,
            )
            performer = self._performers.get(batch.assignee_id)
            if performer is None or not performer.active:
                raise ConflictError("Review assignee is missing or archived")

            if batch.selection.cardinality() != len(batch.items):
                raise ConflictError("Every selected frame must have exactly one vector version")
            selected_ids = {
                coordinate.frame_id(project.id) for coordinate in batch.selection.iter_coordinates()
            }
            if selected_ids != {item.frame_id for item in batch.items}:
                raise ConflictError("Review items do not exactly match the immutable frame selection")

            for item in batch.items:
                self._validate_item(uow, batch, item)
            requested = {(item.frame_id, item.vector_version_id) for item in batch.items}
            for active in uow.projections.list_active_review_batches(project.id, layer.id):
                occupied = {(item.frame_id, item.vector_version_id) for item in active.items}
                if requested & occupied:
                    raise ConflictError("A selected vector version is already in an active review batch")

            next_layer = replace(layer, revision=layer.revision + 1)
            event = EventEnvelope.create(
                stream_id=_layer_stream(str(layer.id)),
                project_id=project.id,
                revision=next_layer.revision,
                event_type="ReviewBatchCreated",
                payload=_batch_payload(batch),
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _layer_stream(str(layer.id)), expected_revision=layer.revision, events=(event,)
            )
            uow.projections.save_review_batch(batch)
            uow.projections.save_layer(next_layer)
            uow.commit()
            return batch

    @staticmethod
    def _validate_item(uow: UnitOfWork, batch: ReviewBatch, item: ReviewItem) -> None:
        vector = uow.projections.get_artifact_version(item.vector_version_id)
        if vector is None or vector.sha256 != item.vector_sha256:
            raise ConflictError("Review vector version is missing or its hash snapshot differs")
        series = uow.projections.get_artifact_series(vector.series_id)
        if (
            series is None
            or series.archived
            or series.project_id != batch.project_id
            or series.layer_id != batch.layer_id
            or series.frame_id != item.frame_id
        ):
            raise ConflictError("Review vector does not belong to the selected layer and frame")
        representation = uow.projections.get_representation(series.representation_id)
        if (
            representation is None
            or representation.kind is not RepresentationKind.VECTOR
            or representation.state is StructureState.ARCHIVED
        ):
            raise ConflictError("Review item must reference a vector representation")
        if item.image_version_id is None:
            return
        image = uow.projections.get_artifact_version(item.image_version_id)
        image_series = None if image is None else uow.projections.get_artifact_series(image.series_id)
        image_representation = (
            None
            if image_series is None
            else uow.projections.get_representation(image_series.representation_id)
        )
        if (
            image is None
            or image_series is None
            or image_series.archived
            or image_series.project_id != batch.project_id
            or image_series.layer_id != batch.layer_id
            or image_series.frame_id != item.frame_id
            or image_representation is None
            or image_representation.kind is not RepresentationKind.IMAGE
            or image_representation.state is StructureState.ARCHIVED
        ):
            raise ConflictError("Review image does not belong to the selected layer and frame")
