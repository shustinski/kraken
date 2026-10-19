"""manager_app: layer manager layer action."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.domain.project import RepresentationKind


class LayerManagerLayerActionMixin:
    def _layer_manager_layer_action(self, _layer_id: str, action: str) -> None:
        if action == "manage_stitching":
            available, reason = self._action_availability(action)
            if not available:
                self._error(reason)
                return
            from kraken_hub.stitching_dialog import StitchingDialog

            try:
                StitchingDialog(self, str(_layer_id)).exec()
            except Exception as exc:
                self._error(str(exc))
            return
        if action in {"rename_layer", "archive_layer"}:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            workspace = self._workspace
            project = self.service.get_project(self._project_id)
            if workspace is None or project is None:
                return
            layer = next(
                (
                    item
                    for item in self.service.list_layers(project.id)
                    if str(item.id) == str(_layer_id)
                ),
                None,
            )
            if layer is None:
                self._error("Слой больше не доступен.")
                return
            try:
                if action == "rename_layer":
                    name, accepted = QInputDialog.getText(
                        self.shell,
                        "Переименовать слой",
                        "Новое название:",
                        text=layer.name,
                    )
                    if not accepted or name.strip() == layer.name:
                        return
                    self.service.rename_layer(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        name=name,
                        idempotency_key=str(uuid4()),
                    )
                else:
                    answer = QMessageBox.question(
                        self.shell,
                        "Архивировать слой",
                        f"Архивировать слой «{layer.name}» без удаления файлов?",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
                    self.service.archive_layer(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        idempotency_key=str(uuid4()),
                    )
            except Exception as exc:
                self._error(str(exc))
                return
            latest = self.service.get_project(project.id)
            if latest is not None:
                self._load_layers(workspace, latest)
            return
        if action == "delete_layer":
            from PyQt6.QtWidgets import (
                QDialog,
                QInputDialog,
                QLineEdit,
                QMessageBox,
            )

            workspace = self._workspace
            project = self.service.get_project(self._project_id)
            if workspace is None or project is None:
                return
            layer = next(
                (
                    item
                    for item in self.service.list_layers(project.id)
                    if str(item.id) == str(_layer_id)
                ),
                None,
            )
            if layer is None:
                self._error("Слой больше не доступен")
                return
            confirmation_dialog = QInputDialog(self.shell)
            confirmation_dialog.setWindowTitle("Удалить слой")
            confirmation_dialog.setLabelText(
                "Управляемые файлы будут перемещены в корзину Kraken, "
                "внешние папки останутся без изменений.\n\n"
                f"Для подтверждения введите: {layer.name}"
            )
            confirmation_dialog.setInputMode(QInputDialog.InputMode.TextInput)
            confirmation_dialog.setTextEchoMode(QLineEdit.EchoMode.Normal)
            confirmation_dialog.setOkButtonText("Удалить")
            confirmation_dialog.setCancelButtonText("Отмена")
            if confirmation_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            confirmation = confirmation_dialog.textValue()
            if confirmation != layer.name:
                QMessageBox.warning(
                    self.shell,
                    "Удаление не подтверждено",
                    "Введённое имя не совпадает с названием слоя.",
                )
                return
            try:
                self.service.delete_layer(
                    principal=self.session.principal,
                    project=project,
                    layer=layer,
                    confirmation_name=confirmation,
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:
                self._error(str(exc))
                return
            latest = self.service.get_project(project.id)
            if latest is not None:
                self._load_layers(workspace, latest)
            QMessageBox.information(
                self.shell,
                "Слой удалён",
                "Метаданные заархивированы, управляемые файлы удаляются из корзины в фоне.",
            )
            return
        if action == "add_image_representation":
            workspace = self._workspace
            if workspace is None:
                return
            item = workspace.layer_model.layer_by_id(_layer_id)
            if item is None:
                self._error("Слой больше не доступен")
                return
            self._select_layer(workspace, self._project_id, item)
            self._add_representation(
                workspace,
                self._project_id,
                RepresentationKind.IMAGE,
            )
            return
        if action == "karakal":
            available, reason = self._action_availability(action)
            if not available:
                self._error(reason)
                return
            _plugin, capability, mode = self._action_requirement(action)
            workspace = self._workspace
            source_id = (
                ""
                if workspace is None
                else str(workspace.image_representation_combo.currentData() or "")
            )
            if source_id and not self._ensure_agent_source_ready(
                layer_id=str(_layer_id),
                source_representation_id=source_id,
                retry=lambda: self._layer_manager_layer_action(_layer_id, action),
            ):
                return
            try:
                job = self._submit_agent_action(
                    layer_id=str(_layer_id),
                    source_representation_id=source_id,
                    capability=capability,
                    parameters={},
                )
                self.service.record_layer_pipeline_action(
                    principal=self.session.principal,
                    project_id=self._project_id,
                    layer_id=_layer_id,
                    action=action,
                    node_id=_layer_id,
                    plugin_id="karakal",
                    capability=capability,
                    mode=mode,
                    parameters={"plugin_job_id": str(job.id)},
                )
            except Exception as exc:
                self._error(f"Не удалось запустить задание: {exc}")
                return
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(self._project_id, _layer_id)
                )
            return
