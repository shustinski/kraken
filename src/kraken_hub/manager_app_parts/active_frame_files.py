"""manager_app: active frame files."""
from __future__ import annotations
from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion


class ActiveFrameFilesMixin:
    def _active_frame_files(
        self,
        x: int,
        y: int,
    ) -> tuple[tuple[str, ArtifactSeries, ArtifactVersion], ...]:
        workspace = self._workspace
        project_id = str(self._project_id or "")
        layer_id = str(
            getattr(workspace, "_selected_layer_id", "") if workspace else ""
        )
        project = self.service.get_project(project_id) if project_id else None
        if workspace is None or project is None or not layer_id:
            return ()
        frame_id = str(project.frame_id_at(int(x), int(y)))
        representation_ids = tuple(
            identifier
            for identifier in (
                str(workspace.image_representation_combo.currentData() or ""),
                str(workspace.vector_representation_combo.currentData() or ""),
            )
            if identifier
        )
        representations = {
            str(representation.id): representation.name
            for representation in self.service.list_representations(
                project.id,
                layer_id,
            )
            if str(representation.id) in representation_ids
        }
        files: list[tuple[str, ArtifactSeries, ArtifactVersion]] = []
        for representation_id in representation_ids:
            series = next(
                (
                    candidate
                    for candidate in self.service.list_artifact_series(
                        project.id,
                        layer_id=layer_id,
                        representation_id=representation_id,
                    )
                    if str(candidate.frame_id or "") == frame_id
                    and not candidate.archived
                ),
                None,
            )
            if series is None:
                continue
            version = self.service.active_artifact_version(project.id, series.id)
            if version is not None:
                files.append(
                    (
                        representations.get(representation_id, series.name),
                        series,
                        version,
                    )
                )
        return tuple(files)
