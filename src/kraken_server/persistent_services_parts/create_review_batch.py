"""persistent_services: create review batch."""
from __future__ import annotations
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from kraken_manager.application.dto import (
    CreateReviewBatchCommand,
)
from kraken_manager.application.errors import (
    ConflictError as ApplicationConflictError,
)
from kraken_manager.application.errors import (
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.domain.common import (
    PerformerId,
    RepresentationId,
    validate_uuid,
)
from kraken_manager.domain.project import (
    RepresentationKind,
)
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import (
    ReviewItem,
)
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import CommandContext, ConflictError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_id,
    _project_id,
)


class CreateReviewBatchMixin:
    def create_review_batch(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._create_review_batch is None:
            raise ConflictError("Server performer store is not configured")
        try:
            project = self.projections.get_project(_project_id(project_id))
            if project is None:
                raise ApplicationNotFoundError("Project was not found")
            layer_id = _layer_id(str(payload.get("layer_id", "")))
            layer = self.projections.get_layer(layer_id)
            if layer is None or layer.project_id != project.id:
                raise ApplicationNotFoundError("Layer was not found in the project")
            image_id = RepresentationId(
                validate_uuid(
                    str(payload.get("image_representation_id", "")),
                    field="image_representation_id",
                )
            )
            vector_id = RepresentationId(
                validate_uuid(
                    str(payload.get("vector_representation_id", "")),
                    field="vector_representation_id",
                )
            )
            image = self.projections.get_representation(image_id)
            vector = self.projections.get_representation(vector_id)
            if image is None or image.layer_id != layer.id or image.kind is not RepresentationKind.IMAGE:
                raise ApplicationConflictError("Select an image representation from the layer")
            if (
                vector is None
                or vector.layer_id != layer.id
                or vector.kind is not RepresentationKind.VECTOR
                or vector.source_image_representation_id != image.id
            ):
                raise ApplicationConflictError(
                    "Select a vector representation linked to the selected images"
                )
            raw_coordinates = payload.get("coordinates", ())
            if not isinstance(raw_coordinates, list):
                raise TypeError("coordinates must be an array")
            coordinates = sorted(
                {(int(item[0]), int(item[1])) for item in raw_coordinates},
                key=lambda item: (item[1], item[0]),
            )
            selection = FrameSelectionV1(
                row_ranges=tuple(
                    FrameRowRange(y=y, x_start=x, x_end=x) for x, y in coordinates
                )
            )
            items: list[ReviewItem] = []
            missing: list[str] = []
            for coordinate in selection.iter_coordinates():
                frame_id = coordinate.frame_id(project.id)
                image_version = self.projections.get_active_artifact_version(
                    deterministic_frame_series_id(image.id, frame_id)
                )
                vector_version = self.projections.get_active_artifact_version(
                    deterministic_frame_series_id(vector.id, frame_id)
                )
                if image_version is None or vector_version is None:
                    missing.append(f"({coordinate.x}, {coordinate.y})")
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
                raise ApplicationConflictError(
                    "Missing image or vector artifact versions for frames: "
                    + ", ".join(missing)
                )
            due_at = payload.get("due_at")
            value = self._create_review_batch(
                CreateReviewBatchCommand(
                    context=self._application_context(context),
                    project_id=project.id,
                    layer_id=layer.id,
                    selection=selection,
                    items=tuple(items),
                    assignee_id=PerformerId(
                        validate_uuid(str(payload.get("assignee_id", "")), field="assignee_id")
                    ),
                    expected_layer_revision=self._require_revision(context),
                    instructions=str(payload.get("instructions", "")),
                    due_at=(
                        None
                        if due_at in (None, "")
                        else datetime.fromisoformat(str(due_at))
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")
