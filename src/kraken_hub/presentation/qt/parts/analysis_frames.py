"""manager_app: analysis frames."""
from __future__ import annotations
from pathlib import Path
from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    AnalysisSourceBinding,
    payload_sha256,
)
from kraken_manager.domain.project import RepresentationKind


class AnalysisFramesMixin:
    def _analysis_frames(self, workspace, project_id, configuration) -> tuple[
        tuple[AnalysisFrameInput, ...], tuple[AnalysisSourceBinding, ...]
    ]:
        layer_id = getattr(workspace, "_selected_layer_id", None)
        if not layer_id:
            raise ValueError("Сначала выберите слой проекта")
        coordinates = workspace.matrix_view.selected_coordinates(maximum=100_000)
        if not coordinates:
            raise ValueError("Выборка не содержит кадров")
        raw_bindings = configuration.get("bindings")
        if not isinstance(raw_bindings, dict):
            raise ValueError("Некорректные привязки моделей A/B/C")
        representations = {
            str(item.id): item
            for item in self.service.list_representations(project_id, layer_id)
        }
        resolved: dict[str, tuple[object, dict[tuple[int, int], object]]] = {}
        source_bindings: list[AnalysisSourceBinding] = []
        for key, identifier_value in raw_bindings.items():
            identifier = str(identifier_value)
            representation = representations.get(identifier)
            if representation is None or representation.kind is not RepresentationKind.IMAGE:
                raise ValueError(f"Модель {key} не является доступным изображением этого слоя")
            cells = {
                (item.x, item.y): item
                for item in self.service.frame_cells(project_id, layer_id, identifier)
            }
            selected_cells = [cells.get(coordinate) for coordinate in coordinates]
            missing = [coordinate for coordinate, cell in zip(coordinates, selected_cells, strict=True) if cell is None]
            if missing:
                preview = ", ".join(f"({x}, {y})" for x, y in missing[:5])
                raise ValueError(f"У модели {key} нет {len(missing)} выбранных кадров: {preview}")
            digests = [str(cell.sha256) for cell in selected_cells if cell is not None]
            source_bindings.append(
                AnalysisSourceBinding(
                    binding_key=str(key),
                    source_id=identifier,
                    source_version=payload_sha256(digests),
                    display_name=representation.name,
                )
            )
            resolved[str(key)] = (representation, cells)

        frames: list[AnalysisFrameInput] = []
        for x, y in coordinates:
            artifacts: list[AnalysisArtifactInput] = []
            frame_id = ""
            for key, (representation, cells) in resolved.items():
                cell = cells[(x, y)]
                if frame_id and frame_id != cell.frame_id:
                    raise ValueError(f"Привязки моделей расходятся по идентификатору кадра ({x}, {y})")
                frame_id = cell.frame_id
                version_item = self.service.get_artifact_version(project_id, cell.artifact_version_id)
                if version_item is None:
                    raise ValueError(f"Версия артефакта для кадра ({x}, {y}) больше недоступна")
                suffix = Path(version_item.filename).suffix.lower()
                artifacts.append(
                    AnalysisArtifactInput(
                        binding_key=key,
                        role=AnalysisSourceRole.MODEL_OUTPUT,
                        artifact_id=str(version_item.series_id),
                        artifact_version_id=str(version_item.id),
                        relative_path=f"inputs/{key}/{frame_id}{suffix}",
                        media_type=version_item.media_type,
                        sha256=version_item.sha256,
                        display_name=representation.name,
                    )
                )
            frames.append(AnalysisFrameInput(frame_id=frame_id, x=x, y=y, artifacts=tuple(artifacts)))
        return tuple(frames), tuple(source_bindings)
