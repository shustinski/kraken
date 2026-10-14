"""composition: create review batch."""
from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime
from kraken_manager.application.dto import (
    CommandContext,
    CreateReviewBatchCommand,
)
from kraken_manager.application.review_workflow import (
    CreateReviewBatchHandler,
)
from kraken_manager.domain.artifacts import (
    deterministic_frame_series_id,
)
from kraken_manager.domain.common import (
    LayerId,
    PerformerId,
    ProjectId,
    RepresentationId,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.project import (
    RepresentationKind,
)
from kraken_manager.domain.workflows import (
    ReviewBatch,
    ReviewItem,
)


class CreateReviewBatchMixin:
    def create_review_batch(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        image_representation_id: RepresentationId | str,
        vector_representation_id: RepresentationId | str,
        coordinates: Iterable[tuple[int, int]],
        assignee_id: PerformerId | str,
        instructions: str = "",
        due_at: datetime | None = None,
        idempotency_key: str,
    ) -> ReviewBatch:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project was not found")
        layer = next(
            (
                item
                for item in self.list_layers(project.id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        if layer is None:
            raise ValueError("Layer was not found")
        representations = {
            str(item.id): item
            for item in self.list_representations(project.id, layer.id)
        }
        image_representation = representations.get(str(image_representation_id))
        vector_representation = representations.get(str(vector_representation_id))
        if (
            image_representation is None
            or image_representation.kind is not RepresentationKind.IMAGE
        ):
            raise ValueError("Select an image representation")
        if (
            vector_representation is None
            or vector_representation.kind is not RepresentationKind.VECTOR
            or vector_representation.source_image_representation_id
            != image_representation.id
        ):
            raise ValueError("The CIF representation is not linked to the selected images")

        selection = self._domain_selection(coordinates)
        projection = self._projection(project.id)
        items: list[ReviewItem] = []
        missing: list[str] = []
        for coordinate in selection.iter_coordinates():
            frame_id = coordinate.frame_id(project.id)
            image_series = deterministic_frame_series_id(
                image_representation.id,
                frame_id,
            )
            vector_series = deterministic_frame_series_id(
                vector_representation.id,
                frame_id,
            )
            image_version = projection.get_active_artifact_version(image_series)
            vector_version = projection.get_active_artifact_version(vector_series)
            if image_version is None or vector_version is None:
                absent = []
                if image_version is None:
                    absent.append("изображение")
                if vector_version is None:
                    absent.append("CIF")
                missing.append(
                    f"({coordinate.x}, {coordinate.y}): {', '.join(absent)}"
                )
                continue
            items.append(
                ReviewItem(
                    frame_id=frame_id,
                    vector_version_id=vector_version.id,
                    vector_sha256=vector_version.sha256,
                    image_version_id=image_version.id,
                )
            )
        if missing:
            raise ValueError(
                "Нельзя выдать кадры без обязательных файлов:\n"
                + "\n".join(missing)
            )
        return CreateReviewBatchHandler(
            self._uow(str(project.id)),
            self.profiles,
            self.clock,
            self.performers,
        )(
            CreateReviewBatchCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=project.id,
                layer_id=layer.id,
                selection=selection,
                items=tuple(items),
                assignee_id=PerformerId(str(assignee_id)),
                expected_layer_revision=layer.revision,
                instructions=instructions,
                due_at=due_at,
            )
        )
