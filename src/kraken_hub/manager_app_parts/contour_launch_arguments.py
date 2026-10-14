"""manager_app: contour launch arguments."""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
from kraken_core.plugin_protocol import (
    WorkspacePluginContextV1,
)
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.presentation.qt import (
    PipelineNode,
)
from kraken_manager.workspace import DerivedRunKind


class ContourLaunchArgumentsMixin:
    def _contour_launch_arguments(
        self,
        *,
        layer_id: str,
        node: PipelineNode,
        action: str,
        source_representation_id: str,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        representations = self.service.list_representations(
            self._project_id, layer_id
        )
        input_representation_id = str(
            node.representation_id or source_representation_id
        )
        representation = next(
            (
                item
                for item in representations
                if str(item.id) == input_representation_id
            ),
            None,
        )
        if representation is None or representation.kind is not RepresentationKind.IMAGE:
            raise ValueError("выбранный базовый слой изображений не найден")

        if not all(
            hasattr(self.service, name)
            for name in (
                "list_layers",
                "project_workspace",
                "layer_file_binding",
                "begin_derived_run",
            )
        ):
            stage = (
                self.service.data_dir
                / "agent-staging"
                / f"{self._project_id}-contour-{action}-{uuid4()}"
            ).resolve()
            stage.mkdir(parents=True, exist_ok=False)
            source_path = Path(str(representation.source or "")).expanduser()
            if source_path.is_absolute() and source_path.is_dir():
                input_directory = source_path.resolve()
            else:
                input_directory = stage / "inputs"
                input_directory.mkdir()
                for cell in self.service.frame_cells(
                    self._project_id,
                    layer_id,
                    representation.id,
                ):
                    if not cell.sha256:
                        continue
                    destination = input_directory / f"{cell.x}_{cell.y}.png"
                    with destination.open("xb") as stream:
                        stream.write(
                            self.service.read_project_blob(
                                self._project_id,
                                cell.sha256,
                            )
                        )
            destination = stage / (
                "dataset" if action == "prepare_dataset" else "vectors"
            )
            destination.mkdir()
            destination_option = (
                "--dataset-dir"
                if action == "prepare_dataset"
                else "--output-dir"
            )
            return (
                (
                    "--input-dir",
                    str(input_directory),
                    destination_option,
                    str(destination),
                ),
                {
                    "input_representation_id": str(representation.id),
                    "input_directory": str(input_directory),
                    "result_directory": str(destination),
                },
            )

        layer = next(
            (
                value
                for value in self.service.list_layers(self._project_id)
                if str(value.id) == str(layer_id)
            ),
            None,
        )
        if layer is None:
            raise ValueError("слой больше не доступен")
        workspace = self.service.project_workspace(self._project_id)
        binding = self.service.layer_file_binding(self._project_id, layer_id)
        if workspace is None or binding is None or not workspace.available:
            raise ValueError("двухдисковое хранилище слоя недоступно")
        source_path = Path(str(representation.source or "")).expanduser()
        if source_path.is_absolute() and source_path.is_dir():
            input_directory = source_path.resolve()
        else:
            input_directory = (
                Path(workspace.derived_project_dir)
                / ".plugin-inputs"
                / f"contour-{action}-{uuid4()}"
            )
            input_directory.mkdir(parents=True)
            cells = self.service.frame_cells(
                self._project_id,
                layer_id,
                representation.id,
            )
            for cell in cells:
                if not cell.sha256:
                    continue
                destination = input_directory / f"{cell.x}_{cell.y}.png"
                with destination.open("xb") as stream:
                    stream.write(
                        self.service.read_project_blob(
                            self._project_id, cell.sha256
                        )
                    )
            if not any(input_directory.iterdir()):
                raise ValueError("в базовом слое нет доступных изображений")

        operation = self._action_requirement(action)[1]
        run = self.service.begin_derived_run(
            project_id=self._project_id,
            layer_id=layer_id,
            layer_name=layer.name,
            kind=(
                DerivedRunKind.DATASET
                if action == "prepare_dataset"
                else DerivedRunKind.VECTOR
            ),
            plugin_id="contour",
            operation=operation,
            principal=self.session.principal,
        )
        destination = Path(run.path)
        if action == "prepare_dataset":
            (destination / "images").mkdir(exist_ok=True)
            (destination / "cif").mkdir(exist_ok=True)
        result_manifest = destination / ".kraken-result.json"
        context_path = destination / ".kraken-context.json"
        input_directories = {"images": str(input_directory)}
        if binding.ssc_directory:
            input_directories["ssc"] = binding.ssc_directory
        if binding.prv_directory:
            input_directories["prv"] = binding.prv_directory
        current_project = self.service.get_project(self._project_id)
        if current_project is None:
            raise ValueError("проект больше не доступен")
        context = WorkspacePluginContextV1(
            project_id=str(self._project_id),
            project_name=current_project.name,
            layer_id=str(layer_id),
            layer_name=layer.name,
            operation=operation,
            plugin_id="contour",
            run_id=run.run_id,
            input_directories=input_directories,
            proposed_output_directory=str(destination),
            result_manifest_path=str(result_manifest),
        )
        context.write(context_path)
        return (
            (
                "--kraken-workspace-context",
                str(context_path),
            ),
            {
                "input_representation_id": str(representation.id),
                "input_directory": str(input_directory),
                "result_directory": str(destination),
                "workspace_run_id": run.run_id,
                "workspace_result_manifest": str(result_manifest),
            },
        )
