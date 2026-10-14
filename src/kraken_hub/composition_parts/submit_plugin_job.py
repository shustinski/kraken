"""composition: submit plugin job."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from pathlib import Path
from kraken_manager.application.dto import (
    CommandContext,
    SubmitPluginJobCommand,
)
from kraken_manager.application.plugin_jobs import (
    SubmitPluginJobHandler,
)
from kraken_manager.domain.artifacts import (
    deterministic_frame_series_id,
)
from kraken_manager.domain.common import (
    ArtifactVersionId,
    LayerId,
    ProjectId,
    RepresentationId,
)
from kraken_manager.domain.identity import (
    Principal,
)
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import (
    PluginInputV1,
)


class SubmitPluginJobMixin:
    def submit_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        source_representation_id: RepresentationId | str,
        target_representation_id: RepresentationId | str,
        coordinates: Iterable[tuple[int, int]],
        capability: str,
        parameters: Mapping[str, object],
        idempotency_key: str,
    ):
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project was not found")
        projection = self._projection(project.id)
        source_id = RepresentationId(str(source_representation_id))
        target_id = RepresentationId(str(target_representation_id))
        source = projection.get_representation(source_id)
        target = projection.get_representation(target_id)
        if source is None or str(source.layer_id) != str(layer_id):
            raise ValueError("Source representation was not found in the layer")
        if target is None or str(target.layer_id) != str(layer_id):
            raise ValueError("Target representation was not found in the layer")
        selection = self._domain_selection(coordinates)
        inputs: list[PluginInputV1] = []
        missing: list[str] = []
        for coordinate in selection.iter_coordinates():
            frame_id = coordinate.frame_id(project.id)
            series_id = deterministic_frame_series_id(source.id, frame_id)
            version = projection.get_active_artifact_version(series_id)
            if version is None:
                missing.append(f"({coordinate.x}, {coordinate.y})")
                continue
            suffix = Path(version.filename).suffix or ".bin"
            inputs.append(
                PluginInputV1(
                    frame_id=frame_id,
                    artifact_version_id=version.id,
                    sha256=version.sha256,
                    relative_path=f"inputs/{coordinate.x}_{coordinate.y}{suffix}",
                )
            )
        if capability == "dataset.model.train.v1":
            dataset_id = parameters.get("dataset_version_id")
            version = projection.get_artifact_version(ArtifactVersionId(str(dataset_id)))
            series = None if version is None else projection.get_artifact_series(version.series_id)
            if (
                version is None
                or series is None
                or series.project_id != project.id
                or str(series.layer_id) != str(layer_id)
                or version.media_type != "application/zip"
                or version.parameters.get("plugin_output_role") != "dataset"
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
            missing = []
        if missing:
            raise ValueError(
                "Для запуска отсутствуют входные версии кадров: "
                + ", ".join(missing)
            )
        return SubmitPluginJobHandler(
            self._uow(str(project.id)),
            self.profiles,
            self.clock,
            gateway,
        )(
            SubmitPluginJobCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=project.id,
                layer_id=LayerId(str(layer_id)),
                target_representation_id=target.id,
                selection=selection,
                capability=capability,
                inputs=tuple(inputs),
                parameters=dict(parameters),
            )
        )
