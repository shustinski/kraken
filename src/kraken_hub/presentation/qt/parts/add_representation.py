"""manager_app: add representation."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.domain.project import RepresentationKind


class AddRepresentationMixin:
    def _add_representation(
        self,
        workspace,
        project_id,
        kind: RepresentationKind,
        *,
        source_image_id: str | None = None,
    ) -> None:
        from kraken_hub.presentation.qt.manager_app import RepresentationDialog
        from PyQt6.QtWidgets import QDialog, QMessageBox

        layer_id = getattr(workspace, "_selected_layer_id", None)
        if layer_id is None:
            QMessageBox.information(workspace, "Kraken", "Сначала выберите слой.")
            return
        project = self.service.get_project(project_id)
        layer = next((item for item in self.service.list_layers(project_id) if str(item.id) == layer_id), None)
        if project is None or layer is None:
            self._error("Слой или проект больше не доступен")
            return
        label = "изображение" if kind is RepresentationKind.IMAGE else "вектор"
        if kind is RepresentationKind.VECTOR and not source_image_id:
            source_image_id = str(
                workspace.image_representation_combo.currentData() or ""
            )
        if kind is RepresentationKind.IMAGE:
            source_image_id = None
        if kind is RepresentationKind.VECTOR and not source_image_id:
            QMessageBox.information(
                workspace, "Kraken", "Сначала добавьте и выберите слой изображения."
            )
            return
        form = RepresentationDialog(workspace, kind)
        while form.exec() == QDialog.DialogCode.Accepted:
            if not form.name.text().strip():
                QMessageBox.warning(
                    form.dialog, f"Не удалось добавить {label}", "Введите название."
                )
                form.name.setFocus()
                continue
            import_plan = None
            source_directory = form.source_directory
            if source_directory:
                try:
                    import_plan = self.service.plan_import_directory(
                        project=project,
                        directory=source_directory,
                        mode=ImportMappingMode.ROW_MAJOR_SUFFIX,
                    )
                except Exception as exc:
                    QMessageBox.warning(form.dialog, "Ошибка проверки файлов", str(exc))
                    continue
                issue_text = "\n".join(
                    f"• {issue.message}" for issue in import_plan.issues
                ) or "Ошибок не обнаружено"
                if not import_plan.ready:
                    QMessageBox.warning(
                        form.dialog,
                        "Импорт заблокирован",
                        f"Проверка файлов не пройдена:\n{issue_text}",
                    )
                    continue
                answer = QMessageBox.question(
                    form.dialog,
                    "Подтверждение импорта файлов",
                    (
                        f"Файлов: {len(import_plan.items)}\n"
                        f"Объём: {import_plan.total_bytes:n} байт\n"
                        f"Пустых кадров: {import_plan.missing_coordinates:n}\n\n"
                        f"{issue_text}\n\nЗаписать файлы в базу?"
                    ),
                )
                if answer != QMessageBox.StandardButton.Yes:
                    continue
            try:
                representation = self.service.create_representation(
                    principal=self.session.principal,
                    project=project,
                    layer=layer,
                    name=form.name.text(),
                    kind=kind,
                    idempotency_key=str(uuid4()),
                    note=form.note.toPlainText(),
                    source="managed-import" if import_plan is not None else None,
                    source_image_representation_id=source_image_id,
                    active=form.active.isChecked(),
                )
            except Exception as exc:
                QMessageBox.warning(form.dialog, f"Не удалось добавить {label}", str(exc))
                continue

            if import_plan is not None:
                self._start_representation_import(
                    workspace=workspace,
                    project_id=project_id,
                    layer_id=layer_id,
                    project=project,
                    layer=layer,
                    representation=representation,
                    import_plan=import_plan,
                    kind=kind,
                )
                return

            self._load_representations(workspace, project_id, layer_id)
            combo = (
                workspace.image_representation_combo
                if kind is RepresentationKind.IMAGE
                else workspace.vector_representation_combo
            )
            combo.setCurrentIndex(combo.findData(str(representation.id)))
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(project_id, layer_id)
                )
            return
