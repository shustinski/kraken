"""manager_app: delete project."""
from __future__ import annotations
from kraken_manager.presentation.qt.models import ProjectListItem

from kraken_hub.manager_app_parts.deletion_reason_dialog import _deletion_reason_dialog


class DeleteProjectMixin:
    def delete_project(self, item: ProjectListItem | None) -> None:
        if item is None:
            return
        from PyQt6.QtCore import QSettings
        from PyQt6.QtWidgets import QDialog, QInputDialog, QLineEdit, QMessageBox

        project = self.service.get_project(item.project_id)
        if project is None:
            self._error("Проект больше не доступен")
            self.refresh_projects()
            return
        if getattr(self.service, "is_remote_project", lambda _identifier: False)(project.id):
            reason = _deletion_reason_dialog(self.shell, project.name)
            if reason is None:
                return
            try:
                request = self.service.request_project_deletion(project=project, reason=reason)
            except Exception as exc:
                self._error(str(exc))
                return
            QMessageBox.information(self.shell, "Заявка на удаление",
                                    f"Заявка ожидает решения в Kraken Admin.\n{request['request_id']}")
            return
        binding = self.service.project_workspace(project.id)
        preserved_paths = (
            ()
            if binding is None
            else (binding.source_project_dir, binding.derived_project_dir)
        )
        dialog = QInputDialog(self.shell)
        dialog.setObjectName("deleteProjectConfirmationDialog")
        dialog.setWindowTitle("Удалить проект")
        dialog.setLabelText(
            "Проект будет удалён из Kraken вместе с его метаданными, историей "
            "и файлами кэша.\n\n"
            "Папки с изображениями слоёв, результатами нейросети и векторами "
            "останутся без изменений:\n"
            + ("\n".join(preserved_paths) if preserved_paths else "Папки проекта не зарегистрированы.")
            + f"\n\nДля подтверждения введите: {project.name}"
        )
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        dialog.setTextEchoMode(QLineEdit.EchoMode.Normal)
        dialog.setOkButtonText("Удалить из Kraken")
        dialog.setCancelButtonText("Отмена")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.textValue() != project.name:
            QMessageBox.warning(
                self.shell,
                "Удаление не подтверждено",
                "Введённое имя не совпадает с названием проекта.",
            )
            return

        if self._project_id == str(project.id) and self._workspace is not None:
            try:
                self._workspace.matrix_view.clear_thumbnail_cache()
                self._workspace.matrix_view.close()
            finally:
                self._workspace = None
                self._project_id = None
                if self._layer_dialog is not None:
                    self._layer_dialog.hide()
                    self._layer_dialog.deleteLater()
                    self._layer_dialog = None
        try:
            self._clear_configured_thumbnail_cache(str(project.id))
            self.service.delete_project(
                principal=self.session.principal,
                project=project,
                confirmation_name=dialog.textValue(),
            )
        except Exception as exc:
            self._error(str(exc))
            return

        settings = QSettings("Kraken", "KrakenHub")
        settings.remove(f"external-model/{project.id}")
        settings.remove(f"layer-manager/{project.id}")
        self.shell.show_page("projects")
        self.refresh_projects()
        QMessageBox.information(
            self.shell,
            "Проект удалён",
            "Проект и его кэш удалены из Kraken. Файлы в папках проекта сохранены.",
        )
