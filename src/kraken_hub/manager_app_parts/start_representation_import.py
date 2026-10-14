"""manager_app: start representation import."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.domain.project import RepresentationKind

from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread


class StartRepresentationImportMixin:
    def _start_representation_import(
        self,
        *,
        workspace,
        project_id,
        layer_id: str,
        project,
        layer,
        representation,
        import_plan,
        kind: RepresentationKind,
    ) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog

        progress_dialog = QProgressDialog(
            "Файлы сохраняются в проекте в фоновом режиме…",
            "",
            0,
            0,
            self.shell,
        )
        progress_dialog.setObjectName("representationImportProgress")
        progress_dialog.setWindowTitle("Сохранение файлов")
        progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
        progress_dialog.setCancelButton(None)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        operation_id = str(uuid4())

        def perform_import():
            return self.service.commit_managed_import(
                principal=self.session.principal,
                project=project,
                layer=layer,
                representation=representation,
                plan=import_plan,
                idempotency_key=operation_id,
            )

        worker = _BackgroundCallThread(perform_import, parent=self.shell)
        workers = getattr(self, "_background_workers", None)
        if workers is None:
            workers = set()
            self._background_workers = workers
        workers.add(worker)

        def refresh_created_representation() -> None:
            if (
                self._workspace is not workspace
                or self._project_id != str(project_id)
                or getattr(workspace, "_selected_layer_id", None) != layer_id
            ):
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

        def import_succeeded(result: object) -> None:
            progress_dialog.close()
            refresh_created_representation()
            saved_count = len(result.versions)
            message = f"Сохранено файлов: {saved_count:n}."
            if import_plan.missing_coordinates:
                message += (
                    f" Не найдено файлов для кадров: "
                    f"{import_plan.missing_coordinates:n}."
                )
            self.shell.statusBar().showMessage(message, 15000)

        def import_failed(message: str) -> None:
            progress_dialog.close()
            refresh_created_representation()
            QMessageBox.warning(
                self.shell,
                "Не удалось сохранить файлы",
                (
                    "Слой изображений создан, но не все файлы удалось сохранить.\n\n"
                    f"{message}"
                ),
            )

        def release_worker() -> None:
            workers.discard(worker)

        worker.succeeded.connect(import_succeeded)
        worker.failed.connect(import_failed)
        worker.finished.connect(release_worker)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        progress_dialog.show()
