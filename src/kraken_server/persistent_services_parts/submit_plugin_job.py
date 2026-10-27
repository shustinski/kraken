"""persistent_services: submit plugin job."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from pathlib import Path
from kraken_manager.application.dto import (
    SubmitPluginJobCommand,
)
from kraken_manager.application.errors import (
    ConflictError as ApplicationConflictError,
)
from kraken_manager.application.errors import (
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.domain.common import (
    ArtifactVersionId,
    RepresentationId,
    validate_uuid,
)
from kraken_manager.domain.artifacts import deterministic_frame_series_id, is_training_dataset_artifact
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import (
    PluginInputV1,
)
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_server.services import CommandContext, ConflictError

from kraken_server.persistent_services_parts.layer_source_scan import (
    _layer_id,
    _project_id,
)


class SubmitPluginJobMixin:
    def submit_plugin_job(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._submit_plugin_job is None:
            raise ConflictError("Server agent queue is not configured")
        try:
            project = self.projections.get_project(_project_id(project_id))
            if project is None:
                raise ApplicationNotFoundError("Project was not found")
            layer_id = _layer_id(str(payload.get("layer_id", "")))
            source_id = RepresentationId(
                validate_uuid(str(payload.get("source_representation_id", "")), field="source_representation_id")
            )
            target_id = RepresentationId(
                validate_uuid(str(payload.get("target_representation_id", "")), field="target_representation_id")
            )
            source = self.projections.get_representation(source_id)
            target = self.projections.get_representation(target_id)
            if source is None or source.layer_id != layer_id or target is None or target.layer_id != layer_id:
                raise ApplicationNotFoundError("Source or target representation was not found in the layer")
            raw_coordinates = payload.get("coordinates", ())
            if not isinstance(raw_coordinates, list):
                raise TypeError("coordinates must be an array")
            coordinates = sorted(
                {(int(item[0]), int(item[1])) for item in raw_coordinates},
                key=lambda item: (item[1], item[0]),
            )
            selection = FrameSelectionV1(
                row_ranges=tuple(FrameRowRange(y=y, x_start=x, x_end=x) for x, y in coordinates)
            )
            inputs: list[PluginInputV1] = []
            input_media_types: dict[str, str] = {}
            raw_parameters = payload.get("parameters", {})
            parameters = dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
            missing: list[str] = []
            for coordinate in selection.iter_coordinates():
                frame_id = coordinate.frame_id(project.id)
                series_id = deterministic_frame_series_id(source.id, frame_id)
                version = self.projections.get_active_artifact_version(series_id)
                if version is None:
                    missing.append(f"({coordinate.x}, {coordinate.y})")
                    continue
                suffix = "." + version.filename.rsplit(".", 1)[-1] if "." in version.filename else ".bin"
                input_media_types[str(version.id)] = version.media_type
                inputs.append(
                    PluginInputV1(
                        frame_id=frame_id,
                        artifact_version_id=version.id,
                        sha256=version.sha256,
                        relative_path=(
                            f"inputs/{coordinate.x}_{coordinate.y}/{Path(version.filename).name}"
                            if parameters.get("input_transport") == "server"
                            else f"inputs/{coordinate.x}_{coordinate.y}{suffix}"
                        ),
                        external_uri=(
                            None if version.external is None else version.external.uri
                        ),
                    )
                )
            if str(payload.get("capability", "")) == "dataset.model.train.v1":
                dataset_id = parameters.get("dataset_version_id")
                version = self.projections.get_artifact_version(ArtifactVersionId(str(dataset_id)))
                series = None if version is None else self.projections.get_artifact_series(version.series_id)
                if (
                    version is None
                    or series is None
                    or series.project_id != project.id
                    or str(series.layer_id) != str(layer_id)
                    or not is_training_dataset_artifact(version)
                ):
                    raise ValueError("Select a published dataset from this layer")
                coordinate = next(selection.iter_coordinates())
                selection = FrameSelectionV1(
                    row_ranges=(FrameRowRange(y=coordinate.y, x_start=coordinate.x, x_end=coordinate.x),)
                )
                inputs = [
                    PluginInputV1(
                        frame_id=coordinate.frame_id(project.id),
                        artifact_version_id=version.id,
                        sha256=version.sha256,
                        relative_path="inputs/dataset.zip",
                    )
                ]
                input_media_types = {str(version.id): version.media_type}
                missing = []
            if missing:
                raise ApplicationConflictError(
                    "Missing input artifact versions for frames: " + ", ".join(missing)
                )
            value = self._submit_plugin_job(
                SubmitPluginJobCommand(
                    context=self._application_context(context),
                    project_id=project.id,
                    layer_id=layer_id,
                    target_representation_id=target_id,
                    selection=selection,
                    capability=str(payload.get("capability", "")),
                    inputs=tuple(inputs),
                    parameters={
                        **parameters,
                        "input_media_types": input_media_types,
                        "project_width": project.width,
                    },
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")
