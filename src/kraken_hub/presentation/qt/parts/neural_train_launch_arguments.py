"""manager_app: neural train launch arguments."""
from __future__ import annotations
from pathlib import Path
from kraken_core.plugin_protocol import (
    WorkspacePluginContextV1,
)
from kraken_manager.presentation.qt import (
    PipelineNode,
)
from kraken_manager.workspace import DerivedRunKind


class NeuralTrainLaunchArgumentsMixin:
    def _neural_train_launch_arguments(
        self,
        *,
        layer_id: str,
        node: PipelineNode,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        dataset = Path(str(node.details.get("path", ""))).expanduser()
        images = dataset / "images"
        cif = dataset / "cif"
        if not dataset.is_absolute() or not images.is_dir() or not cif.is_dir():
            raise ValueError(
                "выбранная опубликованная выборка не содержит папки images и cif"
            )
        project = self.service.get_project(self._project_id)
        layer = next(
            (
                value
                for value in self.service.list_layers(self._project_id)
                if str(value.id) == str(layer_id)
            ),
            None,
        )
        if project is None or layer is None:
            raise ValueError("проект или слой больше не доступен")
        workspace = self.service.project_workspace(project.id)
        if workspace is None or not workspace.available:
            raise ValueError("двухдисковое хранилище проекта недоступно")
        operation = self._action_requirement("train")[1]
        run = self.service.begin_derived_run(
            project_id=project.id,
            layer_id=layer.id,
            layer_name=layer.name,
            kind=DerivedRunKind.RESULT,
            plugin_id="neuralimage",
            operation=operation,
            principal=self.session.principal,
        )
        destination = Path(run.path)
        result_manifest = destination / ".kraken-result.json"
        context_path = destination / ".kraken-context.json"
        WorkspacePluginContextV1(
            project_id=str(project.id),
            project_name=project.name,
            layer_id=str(layer.id),
            layer_name=layer.name,
            operation=operation,
            plugin_id="neuralimage",
            run_id=run.run_id,
            input_directories={
                "images": str(images.resolve()),
                "cif": str(cif.resolve()),
            },
            proposed_output_directory=str(destination),
            result_manifest_path=str(result_manifest),
        ).write(context_path)
        return (
            ("--kraken-workspace-context", str(context_path)),
            {
                "dataset_run_id": str(node.details.get("run_id", "")),
                "dataset_directory": str(dataset),
                "result_directory": str(destination),
                "workspace_run_id": run.run_id,
                "workspace_result_manifest": str(result_manifest),
            },
        )
