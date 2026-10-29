"""manager_app: run remote interactive job."""
from __future__ import annotations
from kraken_core.plugins import is_plugin_launchable
from kraken_manager.domain.identity import Permission

from kraken_hub.presentation.qt.parts.layer_create_thread import _BackgroundCallThread


class RunRemoteInteractiveJobMixin:
    def _run_remote_interactive_job(self, job, *, registry) -> None:
        project_id = str(job.project_id)
        layer_id = str(job.layer_id)
        worker = _BackgroundCallThread(
            lambda: self.service.run_interactive_job(project_id, job.id, registry=registry),
            parent=self.shell,
        )
        workers = getattr(self, "_background_workers", None)
        if workers is None:
            workers = set()
            self._background_workers = workers
        workers.add(worker)

        def finished(result) -> None:
            imported = result.get("imported", {})
            state = imported.get("job", {}).get("state", "")
            if state == "cancelled":
                self.shell.statusBar().showMessage("Задание отменено.", 10000)
            elif state == "failed":
                self._error(f"Contour завершил задание со статусом: {state}")
            elif imported.get("requires_partial_confirmation"):
                self._error("Contour вернул неполный результат. Проверьте задание в разделе «Моя работа».")
            else:
                self.shell.statusBar().showMessage("Результаты Contour сохранены в проекте.", 10000)
            if (
                self._workspace is not None
                and str(self._project_id) == project_id
                and str(getattr(self._workspace, "_selected_layer_id", "")) == layer_id
            ):
                self._load_representations(self._workspace, project_id, layer_id)
                self._refresh_layer_pipeline(project_id, layer_id)
            if self.my_work_refresh is not None:
                self.my_work_refresh()

        worker.succeeded.connect(finished)
        worker.failed.connect(lambda message: self._error(f"Не удалось завершить работу Contour: {message}"))
        worker.finished.connect(lambda: workers.discard(worker))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    @staticmethod
    def _action_requirement(action: str) -> tuple[str, str, str]:
        return {
            "prepare_dataset": ("contour", "frames.dataset.prepare.v1", "interactive"),
            "vectorize": ("contour", "frames.vectorize.v1", "interactive"),
            "train": ("neuralimage", "dataset.model.train.v1", "interactive"),
            "recognize": ("neuralimage", "frames.binary-segment.v2", "headless"),
            "recognize_external": ("neuralimage", "frames.binary-segment.v2", "headless"),
            "confidence": ("neuralimage", "frames.confidence.v1", "headless"),
            "karakal": ("karakal", "layer.confidence.analyze.v1", "interactive"),
        }.get(action, ("", "", ""))

    def _action_availability(self, action: str) -> tuple[bool, str]:
        if action == "issue_work":
            if not self._has_permission(Permission.ASSIGN_WORK):
                return False, "Недостаточно прав для выдачи работы"
            return True, ""
        if action in {
            "manage_stitching",
            "add_external_vector",
            "add_image_representation",
            "archive_representation",
            "rename_representation",
            "edit_representation_note",
            "activate_representation",
            "deactivate_representation",
            "delete_pipeline_step",
            "delete_layer",
            "rename_layer",
            "archive_layer",
            "assign_review_layers",
        }:
            if not self._has_permission(Permission.MANAGE_STRUCTURE):
                return False, "Недостаточно прав для изменения структуры проекта"
            return True, ""
        if action == "export_selection":
            if not self._has_permission(Permission.VIEW_PROJECT):
                return False, "Недостаточно прав для выгрузки"
            return True, ""
        if action == "open_in_contour":
            action = "vectorize"
        if not self._has_permission(Permission.RUN_PLUGIN):
            return False, "Недостаточно прав для запуска плагинов"
        if action == "continue_pipeline_step":
            return True, ""
        requirement = self._action_requirement(action)
        if not requirement[0]:
            return False, f"Неизвестное действие: {action}"
        plugin_id, operation, mode = requirement
        if plugin_id == "karakal" and getattr(self.service, "is_remote_project", lambda _id: False)(self._project_id):
            return True, ""
        inventory = self.plugin_items.get(plugin_id)
        if inventory is None or not is_plugin_launchable(inventory) or not inventory.metadata.enabled:
            return False, f"Плагин {plugin_id} не установлен или отключён"
        capability = next(
            (
                item
                for item in inventory.metadata.capabilities
                if item.operation == operation and mode in item.modes
            ),
            None,
        )
        if capability is None:
            return False, f"Плагин {plugin_id} не поддерживает {operation} в режиме {mode}"
        return True, ""
