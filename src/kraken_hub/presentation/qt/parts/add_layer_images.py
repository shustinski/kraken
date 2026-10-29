"""Add another image representation through the same import dialog as a new layer."""

from __future__ import annotations

from uuid import uuid4

from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.domain.layer_board import next_image_representation_name
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose


class AddLayerImagesMixin:
    def _add_images_to_layer(self, workspace, project_id, layer) -> None:
        from PyQt6.QtWidgets import QDialog, QMessageBox

        from kraken_manager.presentation.qt import LayerCreationDialog

        project = self.service.get_project(project_id)
        if project is None or layer is None:
            self._error("Слой или проект больше не доступен")
            return
        is_remote = bool(getattr(self.service, "is_remote_project", lambda _id: False)(project.id))
        dialog = LayerCreationDialog(
            maximum_frames=project.frame_count,
            scanner=lambda directory, *, maximum_frames: self.service.scan_layer_source(
                directory, maximum_frames=maximum_frames, project_id=project.id
            ),
            source_location="server" if is_remote else "local",
            parent=self.shell,
            locked_name=layer.name,
            locked_layer_type=layer.type,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        existing = self.service.list_representations(project.id, layer.id)
        representation_name = next_image_representation_name(
            item.name for item in existing if item.kind is RepresentationKind.IMAGE
        )
        operation_id = str(uuid4())
        if dialog.mode == "manual":
            try:
                representation = self.service.create_representation(
                    principal=self.session.principal,
                    project=project,
                    layer=layer,
                    name=representation_name,
                    kind=RepresentationKind.IMAGE,
                    idempotency_key=operation_id,
                    source=dialog.manual_images.text(),
                    active=True,
                    purpose=RepresentationPurpose.SOURCE,
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Не удалось добавить изображения", str(exc))
                return
            self._load_representations(workspace, project_id, str(layer.id))
            workspace.image_representation_combo.setCurrentIndex(
                workspace.image_representation_combo.findData(str(representation.id))
            )
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(self._pipeline_snapshot(project_id, layer.id))
            return
        scan = dialog.scan_result
        if scan is None or not scan.working_directory:
            self._error("Результат сканирования потерян. Выполните сканирование ещё раз.")
            return
        try:
            import_plan = self.service.plan_import_directory(
                project=project,
                directory=scan.working_directory,
                mode=ImportMappingMode.ROW_MAJOR_SUFFIX,
            )
        except Exception as exc:
            QMessageBox.warning(dialog, "Ошибка проверки файлов", str(exc))
            return
        if not import_plan.ready:
            issue_text = "\n".join(f"• {issue.message}" for issue in import_plan.issues) or "Ошибок не обнаружено"
            QMessageBox.warning(dialog, "Импорт заблокирован", f"Проверка файлов не пройдена:\n{issue_text}")
            return
        try:
            representation = self.service.create_representation(
                principal=self.session.principal,
                project=project,
                layer=layer,
                name=representation_name,
                kind=RepresentationKind.IMAGE,
                idempotency_key=operation_id,
                source="managed-import",
                active=True,
                purpose=RepresentationPurpose.SOURCE,
            )
        except Exception as exc:
            QMessageBox.warning(dialog, "Не удалось добавить изображения", str(exc))
            return
        self._start_representation_import(
            workspace=workspace,
            project_id=project_id,
            layer_id=str(layer.id),
            project=project,
            layer=layer,
            representation=representation,
            import_plan=import_plan,
            kind=RepresentationKind.IMAGE,
        )
