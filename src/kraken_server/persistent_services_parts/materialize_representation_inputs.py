"""persistent_services: materialize representation inputs."""
from __future__ import annotations
import mimetypes
from dataclasses import replace
from pathlib import Path
from kraken_core.safe_files import contained_path, ensure_regular_directory, open_regular_read
from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    CreateArtifactSeriesCommand,
)
from kraken_manager.application.errors import (
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.common import (
    RepresentationId,
    validate_uuid,
)
from kraken_manager.domain.project import (
    GridOrientation,
    RepresentationKind,
)
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_server.services import CommandContext, ValidationError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_id,
    _project_id,
    resolve_storage_path,
)


class MaterializeRepresentationInputsMixin:
    def materialize_representation_inputs(
        self, project_id: str, layer_id: str, representation_id: str, context: CommandContext
    ) -> dict[str, int]:
        """Snapshot registered workspace images using the normal artifact handlers."""
        try:
            project = self.projections.get_project(_project_id(project_id))
            layer = self.projections.get_layer(_layer_id(layer_id))
            representation = self.projections.get_representation(
                RepresentationId(validate_uuid(representation_id, field="representation_id"))
            )
            if (
                project is None
                or layer is None
                or str(layer.project_id) != project_id
                or representation is None
                or representation.layer_id != layer.id
                or str(representation.project_id) != project_id
            ):
                raise ApplicationNotFoundError("Source representation was not found in the project")
            if representation.kind is not RepresentationKind.IMAGE:
                raise ValueError("The source representation must contain images")
            files = self.workspace_files
            binding = None if files is None else files.registry.get_layer(project_id, layer_id)
            if binding is None or not binding.frame_positions:
                raise ValueError("The source layer has no registered image files")
            root = ensure_regular_directory(binding.image_directory)
            if resolve_storage_path(self, str(representation.source or "")).resolve() != root:
                raise ValueError("The representation does not use the registered layer image directory")
            application_context = self._application_context(context)
            prepared = 0
            for filename, position in sorted(binding.frame_positions.items(), key=lambda item: item[1]):
                if Path(filename).name != filename or not 1 <= position <= project.frame_count:
                    raise ValueError("Invalid registered image position or filename")
                x = (position - 1) % project.width + 1
                row = (position - 1) // project.width
                y = row + 1 if project.orientation is GridOrientation.Y_DOWN else project.height - row
                frame_id = project.frame_id_at(x, y)
                series_id = deterministic_frame_series_id(representation.id, frame_id)
                key = f"workspace-input:{representation.id}:{frame_id}"
                self._create_artifact_series(
                    CreateArtifactSeriesCommand(
                        context=replace(application_context, idempotency_key=f"{key}:series"),
                        project_id=project.id,
                        scope=ArtifactScope.FRAME_REPRESENTATION,
                        name=filename,
                        layer_id=layer.id,
                        representation_id=representation.id,
                        frame_id=frame_id,
                        series_id=series_id,
                        file_history="batch",
                        batch_key=f"workspace-input:{representation.id}",
                    )
                )
                if self.projections.get_active_artifact_version(series_id) is not None:
                    prepared += 1
                    continue
                path = contained_path(root, (filename,))
                with open_regular_read(path, root=root) as stream:
                    self._add_artifact_version(
                        AddArtifactVersionCommand(
                            context=replace(application_context, idempotency_key=f"{key}:version"),
                            project_id=project.id,
                            series_id=series_id,
                            filename=filename,
                            media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
                            expected_series_revision=self.artifact_stream_revision(project_id, str(series_id)),
                            parameters={"x": x, "y": y, "source_path": str(path)},
                            file_history="batch",
                            batch_key=f"workspace-input:{representation.id}",
                        ),
                        iter(lambda: stream.read(1024 * 1024), b""),
                    )
                prepared += 1
            return {"prepared_frames": prepared}
        except OSError as exc:
            raise ValidationError(f"Не удалось подготовить исходные изображения: {exc}") from exc
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")
