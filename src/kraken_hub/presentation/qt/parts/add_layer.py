"""manager_app: add layer."""
from __future__ import annotations
from pathlib import Path
from threading import Event
from uuid import uuid4
from kraken_manager.domain.project import LayerType
from kraken_manager.presentation.qt import (
    LayerCreationDialog,
)

from kraken_hub.presentation.qt.parts.layer_create_thread import _LayerCreateThread


class AddLayerMixin:
    def _add_layer(self, workspace, project_id) -> None:
        from PyQt6.QtWidgets import (
            QDialog,
            QInputDialog,
            QMessageBox,
            QProgressDialog,
        )

        project = self.service.get_project(project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return
        project_binding = self.service.project_workspace(project.id)
        is_remote = bool(getattr(self.service, "is_remote_project", lambda _id: False)(project.id))
        if project_binding is None and not is_remote:
            project_binding = self._setup_project_workspace(project)
            if project_binding is None:
                return
        if project_binding is None:
            name, accepted = QInputDialog.getText(
                self.shell,
                "Новый слой",
                "Название:",
            )
            if not accepted or not name.strip():
                return
            layer_labels = {
                "Металл": LayerType.METAL,
                "Контакт": LayerType.CONTACT,
                "Затвор": LayerType.GATE,
                "Диффузия": LayerType.DIFFUSION,
            }
            layer_label, accepted = QInputDialog.getItem(
                self.shell,
                "Новый слой",
                "Тип:",
                tuple(layer_labels),
                editable=False,
            )
            if not accepted:
                return
            try:
                layer = self.service.create_layer(
                    principal=self.session.principal,
                    project=project,
                    name=name,
                    layer_type=layer_labels[layer_label],
                    order=len(self.service.list_layers(project.id)),
                    idempotency_key=str(uuid4()),
                )
            except Exception as exc:  # noqa: BLE001 - Qt boundary reports domain/transport failures
                QMessageBox.warning(
                    self.shell,
                    "Не удалось добавить слой",
                    str(exc),
                )
                return
            self._show_created_layer(workspace, project.id, layer)
            return
        missing = [] if is_remote else [
            value
            for value in (
                project_binding.source_project_dir,
                project_binding.derived_project_dir,
            )
            if not Path(value).is_dir()
        ]
        if missing:
            self._error(
                "Файловые операции заблокированы: недоступно хранилище.\n\n"
                + "\n".join(missing)
            )
            return
        dialog = LayerCreationDialog(
            maximum_frames=project.frame_count,
            scanner=lambda directory, *, maximum_frames: self.service.scan_layer_source(
                directory, maximum_frames=maximum_frames, project_id=project.id
            ),
            source_location="server" if is_remote else "local",
            parent=self.shell,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        order = len(self.service.list_layers(project.id)) + 1
        operation_id = str(uuid4())
        if dialog.mode == "manual":
            try:
                layer, _binding, _representation = self.service.create_external_layer(
                    principal=self.session.principal,
                    project=project,
                    name=dialog.layer_name,
                    layer_type=dialog.layer_type,
                    order=order,
                    image_directory=dialog.manual_images.text(),
                    ssc_directory=dialog.manual_ssc.text() or None,
                    prv_directory=dialog.manual_prv.text() or None,
                    idempotency_key=operation_id,
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Не удалось добавить слой", str(exc))
                return
            self._show_created_layer(workspace, project.id, layer)
            remote = bool(
                getattr(self.service, "is_remote_project", lambda _id: False)(project.id)
            )
            QMessageBox.information(
                self.shell,
                "Слой создан",
                "Файлы размещены в хранилище сервера. Папка, которая уже лежала "
                "внутри этого проекта, оставлена на месте."
                if remote
                else "Внешние каталоги проверены и привязаны. Файлы не копировались.",
            )
            return

        scan = dialog.scan_result
        if scan is None:
            self._error("Результат сканирования потерян. Выполните сканирование ещё раз.")
            return
        conversion = dialog.conversion_settings()
        cancelled = Event()
        progress_dialog = QProgressDialog(
            "Подготовка импорта…",
            "Отменить",
            0,
            max(1, scan.total_files),
            self.shell,
        )
        progress_dialog.setObjectName("layerImportProgress")
        progress_dialog.setWindowTitle("Импорт слоя")
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.canceled.connect(cancelled.set)

        def perform(*, progress, cancelled):
            return self.service.create_layer_from_disk(
                principal=self.session.principal,
                project=project,
                name=dialog.layer_name,
                layer_type=dialog.layer_type,
                order=order,
                scan=scan,
                conversion=conversion,
                idempotency_key=operation_id,
                progress=progress,
                cancelled=cancelled,
            )

        worker = _LayerCreateThread(perform, cancelled=cancelled, parent=self.shell)

        def update_progress(done: int, total: int, label: str) -> None:
            progress_dialog.setMaximum(max(1, total))
            progress_dialog.setValue(done)
            progress_dialog.setLabelText(
                f"Скопировано и обработано: {done:n} из {total:n}\n{label}"
            )

        def succeeded(result: object) -> None:
            progress_dialog.setValue(progress_dialog.maximum())
            progress_dialog.close()
            layer, _binding, _representation = result
            self._show_created_layer(workspace, project.id, layer)
            QMessageBox.information(
                self.shell,
                "Импорт завершён",
                f"Слой «{dialog.layer_name}» создан атомарно. "
                "Исходная папка не изменена.",
            )

        def failed(message: str) -> None:
            progress_dialog.close()
            if cancelled.is_set():
                QMessageBox.information(
                    self.shell,
                    "Импорт отменён",
                    "Слой не создан, частичные файлы удалены.",
                )
            else:
                QMessageBox.warning(self.shell, "Не удалось импортировать слой", message)

        worker.progress.connect(update_progress)
        worker.succeeded.connect(succeeded)
        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        progress_dialog.show()
