"""manager_app: layer manager action."""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
from kraken_core.external_model import ExternalModelLink
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.presentation.qt import (
    PipelineNode,
)


class LayerManagerActionMixin:
    def _layer_manager_action(self, _layer_id: str, _node: PipelineNode, action: str) -> None:
        if action == "continue_pipeline_step":
            self._continue_pipeline_step(_layer_id, _node)
            return
        if action == "manage_stitching":
            self._layer_manager_layer_action(_layer_id, action)
            return
        if action == "assign_review_layers":
            self._assign_cif_review_layers(_layer_id, _node)
            return
        if action == "issue_work":
            available, reason = self._action_availability(action)
            if not available:
                self._error(reason)
                return
            preset = {
                "source": "vectorize",
                "binary": "vectorize",
                "dataset": "train",
                "vector": "vector_review",
                "stitching": "stitch",
            }.get(_node.kind, "vectorize")
            self.issue_work_dialog(preset_kind=preset, layer_id=str(_layer_id))
            return
        if action == "archive_representation":
            self._archive_layer_representation(_layer_id, _node)
            return
        if action in {
            "rename_representation",
            "edit_representation_note",
            "activate_representation",
            "deactivate_representation",
        }:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox

            workspace = self._workspace
            project = self.service.get_project(self._project_id)
            layer = next(
                (
                    item
                    for item in self.service.list_layers(self._project_id)
                    if str(item.id) == str(_layer_id)
                ),
                None,
            )
            representation_id = str(_node.representation_id or _node.node_id)
            representation = next(
                (
                    item
                    for item in self.service.list_representations(
                        self._project_id,
                        _layer_id,
                    )
                    if str(item.id) == representation_id
                ),
                None,
            )
            if workspace is None or project is None or layer is None or representation is None:
                self._error("Репрезентация больше не доступна.")
                return
            try:
                if action == "rename_representation":
                    value, accepted = QInputDialog.getText(
                        self.shell,
                        "Переименовать репрезентацию",
                        "Новое название:",
                        text=representation.name,
                    )
                    if not accepted or value.strip() == representation.name:
                        return
                    self.service.rename_representation(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        name=value,
                        idempotency_key=str(uuid4()),
                    )
                elif action == "edit_representation_note":
                    value, accepted = QInputDialog.getMultiLineText(
                        self.shell,
                        "Заметка репрезентации",
                        "Заметка:",
                        text=representation.note,
                    )
                    if not accepted or value.strip() == representation.note:
                        return
                    self.service.update_representation_note(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        note=value,
                        idempotency_key=str(uuid4()),
                    )
                elif action == "activate_representation":
                    self.service.activate_representation(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        idempotency_key=str(uuid4()),
                    )
                else:
                    answer = QMessageBox.question(
                        self.shell,
                        "Деактивировать репрезентацию",
                        f"Деактивировать «{representation.name}»?",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
                    self.service.deactivate_representation(
                        principal=self.session.principal,
                        project=project,
                        layer=layer,
                        representation=representation,
                        idempotency_key=str(uuid4()),
                    )
            except Exception as exc:
                self._error(str(exc))
                return
            self._load_representations(workspace, self._project_id, _layer_id)
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(self._project_id, _layer_id)
                )
            return
        if action == "add_external_vector":
            workspace = self._workspace
            if workspace is None:
                return
            snapshot = self._pipeline_snapshot(self._project_id, _layer_id)
            source_representation_id = next(
                (
                    lane.lane_id
                    for lane in snapshot.lanes
                    if any(node.node_id == _node.node_id for node in lane.nodes)
                ),
                "",
            )
            if not source_representation_id:
                self._error("Не удалось определить исходный слой изображений для CIF")
                return
            self._add_representation(
                workspace,
                self._project_id,
                RepresentationKind.VECTOR,
                source_image_id=source_representation_id,
            )
            return
        if action == "delete_pipeline_step":
            action_event_id = str(_node.details.get("pipeline_event_id", ""))
            if not action_event_id:
                self._error("Этот шаг является частью неизменяемой истории задания")
                return
            try:
                job_id = str(_node.details.get("job_id", ""))
                job = next((item for item in self.service.plugin_jobs() if str(item.id) == job_id), None)
                if job is not None and job.state.value not in {"succeeded", "failed", "cancelled"}:
                    self.cancel_agent_job(job)
                self.service.remove_layer_pipeline_action(
                    principal=self.session.principal,
                    project_id=self._project_id,
                    layer_id=_layer_id,
                    action_event_id=action_event_id,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                self._error(str(exc))
                return
            if self._layer_dialog is not None:
                self._layer_dialog.set_pipeline(
                    self._pipeline_snapshot(self._project_id, _layer_id)
                )
            return
        available, reason = self._action_availability(action)
        if not available:
            self._error(reason)
            return
        plugin_id = "contour" if action in {"prepare_dataset", "vectorize"} else "neuralimage"
        _plugin, capability, mode = self._action_requirement(action)
        parameters: dict[str, object] = {}
        if action == "train":
            versions = tuple(_node.details.get("artifact_version_ids", ()))
            if not versions and _node.details.get("path") and not self.service.is_remote_project(self._project_id):
                try:
                    arguments, launch_parameters = self._neural_train_launch_arguments(
                        layer_id=str(_layer_id), node=_node
                    )
                    process = self._launch_managed_plugin("neuralimage", arguments=arguments)
                    if process is not None:
                        self._monitor_workspace_result(
                            process,
                            result_manifest=str(launch_parameters["workspace_result_manifest"]),
                            workspace_run_id=str(launch_parameters["workspace_run_id"]),
                            layer_id=str(_layer_id),
                        )
                except (OSError, ValueError) as exc:
                    self._error(str(exc))
                return
            if not versions:
                self._error("Выборка ещё не опубликована. Верните результат из Contour в Kraken.")
                return
            parameters["dataset_version_id"] = str(versions[0])
        snapshot = self._pipeline_snapshot(self._project_id, _layer_id)
        source_representation_id = next(
            (
                lane.lane_id
                for lane in snapshot.lanes
                if any(node.node_id == _node.node_id for node in lane.nodes)
            ),
            "",
        )
        if action == "vectorize" and _node.kind in {"source", "binary"}:
            source_representation_id = str(_node.representation_id or source_representation_id)
        if source_representation_id:
            parameters["source_representation_id"] = source_representation_id
            if not self._ensure_agent_source_ready(
                layer_id=str(_layer_id),
                source_representation_id=source_representation_id,
                retry=lambda: self._layer_manager_action(_layer_id, _node, action),
            ):
                return
        selected_legacy = False
        if action in {"recognize", "confidence"}:
            from kraken_hub.presentation.qt.published_model_dialog import PublishedModelDialog
            identifiers = tuple(_node.details.get("artifact_version_ids", ()))
            selector = PublishedModelDialog(
                self.service, self._project_id, _layer_id, source_representation_id=source_representation_id,
                preferred_version_id=identifiers[0] if identifiers else None,
                background_workers=getattr(self, "_background_workers", None), parent=self.shell,
            )
            if not selector.exec():
                return
            selected = selector.parameters()
            selected_legacy = bool(selected.get("legacy_file"))
            if not selected.get("legacy_file"):
                parameters.update(selected)
        if action in {"recognize", "recognize_external", "confidence"} and "model_version_id" not in parameters:
            from kraken_core.settings import QSettings
            from PyQt6.QtWidgets import QFileDialog

            settings = QSettings("Kraken", "KrakenHub")
            key_root = f"external-model/{self._project_id}/{_layer_id}"
            key = f"{key_root}/path"
            model_path = ""
            model_filename = ""
            if not selected_legacy and (action == "recognize" or (action == "confidence" and _node.kind == "model")):
                version_ids = tuple(_node.details.get("artifact_version_ids", ()))
                if version_ids:
                    model_version = self.service.artifact_version(self._project_id, version_ids[0])
                    if model_version is not None:
                        model_path = str(self.service.managed_artifact_path(self._project_id, model_version.id))
                        model_filename = model_version.filename
                result_directory = Path(str(_node.details.get("path", "")))
                model_path = model_path or str(
                    next(
                        (
                            path
                            for path in result_directory.rglob("*")
                            if path.is_file() and path.suffix.casefold() in {".onnx", ".pt", ".pth", ".h5", ".keras"}
                        ),
                        "",
                    )
                )
            else:
                model_path = str(settings.value(key, "") or "")
            linked_now = False
            if not model_path or not Path(model_path).is_file():
                model_path, _selected_filter = QFileDialog.getOpenFileName(
                    self.shell,
                    "Выберите готовую модель NeuralImage",
                    "",
                    "Модели (*.onnx *.pt *.pth *.h5 *.keras);;Все файлы (*)",
                )
                linked_now = bool(model_path)
            if not model_path:
                return
            try:
                stored_hash = str(settings.value(f"{key_root}/sha256", "") or "")
                stored_size = int(settings.value(f"{key_root}/size", 0) or 0)
                link = (
                    ExternalModelLink(str(Path(model_path).resolve()), stored_size, stored_hash)
                    if not linked_now and stored_size > 0 and len(stored_hash) == 64
                    else ExternalModelLink.observe(model_path)
                )
            except (OSError, ValueError) as exc:
                self._error(f"Не удалось подготовить внешнюю модель: {exc}")
                return
            if action == "recognize_external" and (linked_now or not stored_hash):
                settings.setValue(key, link.path)
                settings.setValue(f"{key_root}/size", link.size)
                settings.setValue(f"{key_root}/sha256", link.observed_sha256)
            parameters = {
                **parameters,
                "model_source_path": link.path,
                "model_filename": model_filename or Path(link.path).name,
                "model_sha256": link.observed_sha256,
                "observed_size": link.size,
                "changed_since_observation": False,
            }
        try:
            job = self._submit_agent_action(
                layer_id=str(_layer_id),
                source_representation_id=source_representation_id,
                capability=capability,
                parameters=parameters,
            )
        except Exception as exc:
            self._error(f"Не удалось запустить задание: {exc}")
            return
        self.service.record_layer_pipeline_action(
            principal=self.session.principal,
            project_id=self._project_id,
            layer_id=_layer_id,
            action=action,
            node_id=_node.node_id,
            plugin_id=plugin_id,
            capability=capability,
            mode=mode,
            parameters={**parameters, "plugin_job_id": str(job.id)},
        )
        if self._layer_dialog is not None:
            self._layer_dialog.set_pipeline(
                self._pipeline_snapshot(self._project_id, _layer_id)
            )
        from PyQt6.QtWidgets import QMessageBox

        if plugin_id == "contour":
            self.shell.statusBar().showMessage("Contour запускается с изображениями слоя…", 10000)
            return
        QMessageBox.information(
            self.shell,
            "Задание создано",
            f"Задание {job.id} передано Kraken Agent и продолжит работу независимо от окна проекта.",
        )

    def _continue_pipeline_step(self, layer_id: str, node: PipelineNode) -> None:
        available, reason = self._action_availability("continue_pipeline_step")
        if not available:
            self._error(reason)
            return
        try:
            if node.details.get("workflow_operation"):
                state = self.service.get_layer_pipeline(self._project_id, layer_id)
                operation = "resume" if state["status"] == "paused" else None
                if node.details.get("approval_required"):
                    operation = "approve"
                if operation:
                    self.service.command_layer_pipeline(
                        project_id=self._project_id, layer_id=layer_id,
                        operation=operation, node_id=node.node_id,
                        principal=self.session.principal,
                        expected_revision=node.details["workflow_revision"],
                        idempotency_key=str(uuid4()),
                    )
                self._schedule_pipeline_cycle()
                return
            job_id = str(node.details.get("job_id") or "")
            job = next((item for item in self.service.plugin_jobs()
                        if str(item.id) == job_id
                        and str(item.project_id) == str(self._project_id)
                        and str(item.layer_id) == str(layer_id)), None)
            if job is None:
                self._error("Задание больше не доступно")
                return
            if job.state.value in {"succeeded", "failed", "cancelled"}:
                self._error("Задание уже завершено. Обновите граф.")
                return
            if job.state.value == "recovery_required":
                self.retry_agent_job(job)
            elif self.service.is_remote_project(job.project_id):
                if job.capability in {"frames.binary-segment.v2", "frames.confidence.v1"}:
                    registry = self.agent_runtime.execution_registry(job.capability)
                else:
                    registry = self.agent_runtime.interactive_registry(job.capability)
                self._run_remote_interactive_job(job, registry=registry)
            elif job.state.value == "waiting_for_user":
                self.launch_selected_work("job", job)
            else:
                self._gateway_for_job(job)
                self._poll_agent_jobs()
        except Exception as exc:
            self._error(f"Не удалось продолжить выполнение: {exc}")
