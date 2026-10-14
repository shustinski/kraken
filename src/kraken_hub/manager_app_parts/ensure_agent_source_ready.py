"""manager_app: ensure agent source ready."""
from __future__ import annotations
from collections.abc import Callable
from PyQt6.QtCore import QTimer

from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread


class EnsureAgentSourceReadyMixin:
    def _ensure_agent_source_ready(
        self,
        *,
        layer_id: str,
        source_representation_id: str,
        retry: Callable[[], None],
    ) -> bool:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog

        project = self.service.get_project(self._project_id)
        if project is None:
            self._error("Проект больше не доступен")
            return False
        existing = self.service.frame_cells(
            project.id,
            layer_id,
            source_representation_id,
        )
        if existing:
            binding = getattr(self.service, "layer_file_binding", lambda *_args: None)(project.id, layer_id)
            if binding is None or len(existing) >= len(binding.frame_positions):
                return True
        layer = next(
            (
                item
                for item in self.service.list_layers(project.id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        representation = next(
            (
                item
                for item in self.service.list_representations(project.id, layer_id)
                if str(item.id) == str(source_representation_id)
            ),
            None,
        )
        prepare = getattr(self.service, "materialize_representation_inputs", None)
        if layer is None or representation is None or not callable(prepare):
            self._error("Не удалось подготовить изображения для задания")
            return False
        key = (str(project.id), str(representation.id))
        pending = getattr(self, "_source_preparation_keys", None)
        if pending is None:
            pending = set()
            self._source_preparation_keys = pending
        if key in pending:
            self.shell.statusBar().showMessage(
                "Изображения уже подготавливаются для задания…",
                5000,
            )
            return False
        pending.add(key)
        progress_dialog = QProgressDialog(
            "Подготовка изображений для просмотра и обработки…",
            "",
            0,
            0,
            self.shell,
        )
        progress_dialog.setObjectName("agentSourcePreparationProgress")
        progress_dialog.setWindowTitle("Подготовка изображений")
        progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
        progress_dialog.setCancelButton(None)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)

        def perform_preparation():
            return prepare(
                principal=self.session.principal,
                project=project,
                layer=layer,
                representation=representation,
            )

        worker = _BackgroundCallThread(perform_preparation, parent=self.shell)
        workers = getattr(self, "_background_workers", None)
        if workers is None:
            workers = set()
            self._background_workers = workers
        workers.add(worker)

        def preparation_succeeded(cells: object) -> None:
            progress_dialog.close()
            pending.discard(key)
            if not cells:
                QMessageBox.warning(
                    self.shell,
                    "Не удалось подготовить задание",
                    "В каталоге исходных изображений не найдено кадров проекта.",
                )
                return
            self.shell.statusBar().showMessage(
                "Изображения подготовлены.",
                5000,
            )
            QTimer.singleShot(0, retry)

        def preparation_failed(message: str) -> None:
            progress_dialog.close()
            pending.discard(key)
            QMessageBox.warning(
                self.shell,
                "Не удалось подготовить задание",
                message,
            )

        def release_worker() -> None:
            workers.discard(worker)

        worker.succeeded.connect(preparation_succeeded)
        worker.failed.connect(preparation_failed)
        worker.finished.connect(release_worker)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        progress_dialog.show()
        return False
