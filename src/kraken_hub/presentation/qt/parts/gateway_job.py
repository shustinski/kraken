"""manager_app: gateway job."""
from __future__ import annotations
from uuid import uuid4
from kraken_manager.infrastructure.plugin import AgentPluginGateway


class GatewayJobMixin:
    def _allow_close(self) -> bool:
        from PyQt6.QtWidgets import QMessageBox

        if any(
            worker.isRunning()
            for worker in getattr(self, "_background_workers", ())
        ):
            QMessageBox.information(
                self.shell,
                "Сохранение файлов ещё не завершено",
                "Дождитесь окончания сохранения файлов. Выход отменён, чтобы данные не были повреждены.",
            )
            return False
        # Jobs and results are durable. Reopening Hub reconnects to this Agent.
        self.agent_runtime.detach()
        return True

    def _gateway_for_job(self, job) -> AgentPluginGateway:
        from kraken_hub.presentation.qt.manager_app import AgentPluginGateway
        project = self.service.get_project(job.project_id)
        if project is None:
            raise ValueError("Проект задания больше не доступен")
        capabilities = self.agent_runtime.ensure_started()
        coordinate_by_frame = {
            str(coordinate.frame_id(project.id)): (coordinate.x, coordinate.y)
            for coordinate in job.selection.iter_coordinates()
        }

        def version_path(version_id):
            return self.service.managed_artifact_path(project.id, version_id)

        def frame_coordinate(frame_id):
            coordinate = coordinate_by_frame.get(str(frame_id))
            if coordinate is None:
                raise ValueError(f"Кадр {frame_id} не входит в задание")
            return coordinate

        def media_type(version_id):
            version = self.service.artifact_version(project.id, version_id)
            if version is None:
                raise ValueError(f"Версия {version_id} больше не доступна")
            return version.media_type

        gateway = AgentPluginGateway(
            base_url=self.agent_runtime.base_url,
            token=self.agent_runtime.token,
            staging_root=self.service.data_dir / "agent" / "staging",
            source_for_version=version_path,
            coordinate_for_frame=frame_coordinate,
            media_type_for_version=media_type,
            capabilities=capabilities,
            v2_capabilities=frozenset(
                operation
                for operation, protocol
                in self.agent_runtime.protocol_by_capability.items()
                if protocol == "2.0"
            ),
        )
        self._agent_gateway = gateway
        return gateway

    def cancel_agent_job(self, job) -> None:
        gateway = None if self.service.is_remote_project(job.project_id) else self._gateway_for_job(job)
        self.service.cancel_plugin_job(
            principal=self.session.principal,
            gateway=gateway,
            job=job,
            idempotency_key=str(uuid4()),
        )
        if self.my_work_refresh is not None:
            self.my_work_refresh()

    def delete_agent_job(self, job) -> None:
        if job.state.value not in {"succeeded", "failed", "cancelled"}:
            self.cancel_agent_job(job)
        creation = next(
            event
            for event in self.service.history(job.project_id)
            if event.event_type == "PluginJobCreated" and str(event.payload.get("plugin_job_id", "")) == str(job.id)
        )
        self.service.remove_layer_pipeline_action(
            principal=self.session.principal,
            project_id=job.project_id,
            layer_id=job.layer_id,
            action_event_id=creation.event_id,
            remove_task=True,
        )
        if self.my_work_refresh is not None:
            self.my_work_refresh()

    def retry_agent_job(self, job) -> None:
        gateway = None if self.service.is_remote_project(job.project_id) else self._gateway_for_job(job)
        self.service.retry_plugin_job(
            principal=self.session.principal,
            gateway=gateway,
            job=job,
            idempotency_key=str(uuid4()),
        )
        if self.my_work_refresh is not None:
            self.my_work_refresh()

    def import_partial_agent_job(self, job) -> None:
        gateway = self._gateway_for_job(job)
        gateway.confirm_partial(job.id)
        self._poll_agent_jobs()
