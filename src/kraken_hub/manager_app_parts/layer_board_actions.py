"""Context actions for a multi-selection on the layer board."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from kraken_manager.domain.layer_board import ASSIGNMENT_ACTION, BoardNode
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.presentation.qt import PipelineNode


class LayerBoardActionsMixin:
    def _layer_manager_selection_action(self, layer_id: str, nodes: object, action: str) -> None:
        selected = tuple(nodes) if isinstance(nodes, (list, tuple)) else ()
        if action == "issue_work":
            self._issue_selected_board_work(layer_id, selected)
            return
        if action == "export_selection":
            self._export_board_nodes(layer_id, selected)
            return
        if action == "open_in_contour":
            self._open_board_selection_in_contour(layer_id, selected)

    def _assign_cif_review_layers(self, layer_id: str, node: PipelineNode) -> None:
        from PyQt6.QtWidgets import QDialog

        from kraken_hub.cif_review_layers_dialog import CifReviewLayersDialog
        from kraken_manager.domain.layer_board import review_image_ids

        vector_id = str(node.representation_id or node.node_id)
        images = self._board_image_choices()
        saved = review_image_ids(self.service.history(self._project_id), (vector_id,))
        layer = next(
            (item for item in self.service.list_layers(self._project_id) if str(item.id) == str(layer_id)),
            None,
        )
        dialog = CifReviewLayersDialog(
            source_layer_name="" if layer is None else layer.name,
            images=images,
            selected_ids=saved,
            parent=self.shell,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.service.record_layer_pipeline_action(
                principal=self.session.principal,
                project_id=self._project_id,
                layer_id=layer_id,
                action=ASSIGNMENT_ACTION,
                node_id=vector_id,
                plugin_id="kraken",
                capability="cif.review-layers.v1",
                mode="assignment",
                parameters={
                    "vector_representation_ids": [vector_id],
                    "source_layer_id": str(layer_id),
                    "image_representation_ids": list(dialog.ordered_ids()),
                },
            )
        except Exception as exc:
            self._error(str(exc))
            return
        if self._layer_dialog is not None:
            self._layer_dialog.set_pipeline(self._pipeline_snapshot(self._project_id, layer_id))

    def _board_image_choices(self) -> tuple[tuple[str, str], ...]:
        choices: list[tuple[str, str]] = []
        for layer in self.service.list_layers(self._project_id):
            for representation in self.service.list_representations(self._project_id, layer.id):
                if representation.kind is not RepresentationKind.IMAGE:
                    continue
                choices.append((str(representation.id), f"{layer.name} — {representation.name}"))
        return tuple(choices)

    def _issue_selected_board_work(self, layer_id: str, nodes: tuple[PipelineNode, ...]) -> None:
        workspace = self._workspace
        if workspace is not None:
            image = next((node for node in nodes if node.kind in {"source", "binary"} and node.representation_id), None)
            vector = next((node for node in nodes if node.kind == "vector" and node.representation_id), None)
            if image is not None:
                index = workspace.image_representation_combo.findData(image.representation_id)
                if index >= 0:
                    workspace.image_representation_combo.setCurrentIndex(index)
            if vector is not None:
                index = workspace.vector_representation_combo.findData(vector.representation_id)
                if index >= 0:
                    workspace.vector_representation_combo.setCurrentIndex(index)
        self.issue_work_dialog(
            preset_kind="vector_review" if any(node.kind == "vector" for node in nodes) else "vectorize",
            layer_id=str(layer_id),
        )

    def _open_board_selection_in_contour(self, layer_id: str, nodes: tuple[PipelineNode, ...]) -> None:
        images = [node for node in nodes if node.kind in {"source", "binary"}]
        vector = next((node for node in nodes if node.kind == "vector"), None)
        if not images or vector is None:
            self._error("Для открытия в Contour нужны изображения и один CIF.")
            return
        primary = images[0]
        try:
            arguments, launch = self._contour_launch_arguments(
                layer_id=str(layer_id),
                node=primary,
                action="vectorize",
                source_representation_id=str(primary.representation_id or primary.node_id),
            )
            input_directory = Path(str(launch.get("input_directory", "")))
            if input_directory.is_dir():
                self._copy_board_frames(layer_id, images[1:], input_directory, suffix="png")
                if vector.representation_id:
                    self._copy_board_frames(layer_id, (vector,), input_directory, suffix="cif")
            process = self._launch_managed_plugin("contour", arguments=arguments)
        except Exception as exc:
            self._error(str(exc))
            return
        if process is not None:
            self.shell.statusBar().showMessage("Contour открыт с выбранными изображениями и CIF.", 10000)

    def _copy_board_frames(self, layer_id: str, nodes: tuple[PipelineNode, ...], directory: Path, *, suffix: str) -> None:
        for node in nodes:
            representation_id = str(node.representation_id or "")
            if not representation_id:
                continue
            prefix = representation_id[:8]
            for cell in self.service.frame_cells(self._project_id, layer_id, representation_id):
                if not cell.sha256:
                    continue
                destination = directory / f"{prefix}_{cell.x}_{cell.y}.{suffix}"
                if destination.exists():
                    continue
                with destination.open("xb") as stream:
                    stream.write(self.service.read_project_blob(self._project_id, cell.sha256))

    def _export_board_nodes(self, layer_id: str, nodes: tuple[PipelineNode, ...]) -> None:
        from datetime import UTC, datetime

        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        from kraken_manager.domain.layer_board import companion_export_plan, review_image_ids

        parent = QFileDialog.getExistingDirectory(self.shell, "Каталог для выгрузки")
        if not parent:
            return
        destination = Path(parent) / (
            "Слой_" + datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
        )
        exported = 0
        try:
            destination.mkdir(parents=False, exist_ok=False)
            selected_ids = [
                str(node.representation_id)
                for node in nodes
                if node.representation_id and node.kind in {"source", "binary", "vector"}
            ]
            for representation_id in selected_ids:
                exported += self._export_representation_files(representation_id, destination)
            vector_ids = [str(node.representation_id) for node in nodes if node.kind == "vector" and node.representation_id]
            ordered = review_image_ids(self.service.history(self._project_id), vector_ids)
            names = {
                identifier: tuple(label.split(" — ", 1)) if " — " in label else ("слой", label)
                for identifier, label in self._board_image_choices()
            }
            primary = next((str(node.representation_id) for node in nodes if node.kind in {"source", "binary"} and node.representation_id), "")
            for folder, image_id in companion_export_plan(ordered, primary_image_id=primary, names_by_id=names):
                if image_id in selected_ids:
                    continue
                folder_path = destination / folder
                folder_path.mkdir(exist_ok=True)
                exported += self._export_representation_files(image_id, folder_path)
            if exported == 0:
                destination.rmdir()
                QMessageBox.information(self.shell, "Нет файлов", "У выбранных объектов нет файлов для выгрузки.")
                return
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self.shell, "Не удалось выгрузить файлы", str(exc))
            return
        QMessageBox.information(self.shell, "Файлы выгружены", f"Файлов: {exported}\nКаталог: {destination}")

    def _export_representation_files(self, representation_id: str, destination: Path) -> int:
        layer_id = self._layer_id_for_representation(representation_id)
        if layer_id is None:
            return 0
        count = 0
        for cell in self.service.frame_cells(self._project_id, layer_id, representation_id):
            if not cell.artifact_version_id:
                continue
            version = self.service.artifact_version(self._project_id, cell.artifact_version_id)
            if version is None:
                continue
            filename = version.filename or f"{cell.x}_{cell.y}"
            target = destination / f"{representation_id[:8]}_{filename}"
            self.service.export_artifact_version(self._project_id, version, target)
            count += 1
        return count

    def _layer_id_for_representation(self, representation_id: str) -> str | None:
        for layer in self.service.list_layers(self._project_id):
            for representation in self.service.list_representations(self._project_id, layer.id):
                if str(representation.id) == str(representation_id):
                    return str(layer.id)
        return None


def board_nodes(nodes: tuple[PipelineNode, ...]) -> tuple[BoardNode, ...]:
    return tuple(BoardNode(node.node_id, node.kind, node.title) for node in nodes)
