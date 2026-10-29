"""Workflow UI and background continuation for local layer graphs."""

from __future__ import annotations

from types import SimpleNamespace

from kraken_manager.application.errors import NotFoundError
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from .layer_create_thread import _BackgroundCallThread


class PipelineControllerMixin:
    def open_layer_pipeline(self, layer_id):
        from kraken_hub.presentation.qt.layer_pipeline_dialog import LayerPipelineDialog

        if not self._project_id or not self._has_permission(Permission.RUN_PLUGIN):
            self._error("Нет права запуска обработки / Processing permission is required")
            return
        versions = {
            capability.operation: item.metadata.version
            for item in self.plugin_items.values()
            for capability in item.metadata.capabilities
        }
        dialog = LayerPipelineDialog(
            self.service,
            self.session.principal,
            self._project_id,
            layer_id,
            self.shell,
            plugin_versions=versions,
            background_workers=self._background_workers,
            state_changed=lambda _state: self._schedule_pipeline_cycle(),
        )
        self._pipeline_dialog = dialog
        dialog.show()

    def _schedule_pipeline_cycle(self):
        if getattr(self, "_pipeline_worker", None) is not None:
            return
        service, principal = self.service, self.session.principal

        def cycle():
            count = 0
            for project in service.list_projects():
                is_remote = getattr(service, "is_remote_project", lambda _: False)
                if is_remote(project.id):
                    continue
                for layer in service.list_layers(project.id):
                    try:
                        state = service.get_layer_pipeline(project.id, layer.id)
                    except NotFoundError:
                        continue
                    if state["status"] in {"draft", "cancelled"}:
                        continue
                    coordinates = sorted(
                        {tuple(xy) for op in state["definition"]["operations"] for xy in op.get("coordinates", ())}
                    )
                    if not coordinates:
                        from kraken_manager.infrastructure.plugin.stitch_gateway import AgentStitchGateway
                        self.agent_runtime.ensure_started()
                        stitch_gateway = AgentStitchGateway(base_url=self.agent_runtime.base_url, token=self.agent_runtime.token,
                            staging_root=self.agent_runtime.data_dir / "staging")
                        service.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=principal,
                            stitch_gateway=stitch_gateway)
                        count += 1
                        continue
                    context = SimpleNamespace(
                        project_id=project.id,
                        selection=FrameSelectionV1(
                            row_ranges=tuple(FrameRowRange(y=y, x_start=x, x_end=x) for x, y in coordinates)
                        ),
                    )
                    gateway = self._gateway_for_job(context)
                    service.tick_layer_pipeline(
                        project_id=project.id, layer_id=layer.id, principal=principal, gateway=gateway
                    )
                    count += 1
            return count

        worker = _BackgroundCallThread(cycle, parent=self.shell)
        self._pipeline_worker = worker
        self._background_workers.add(worker)
        worker.failed.connect(
            lambda message: self.shell.statusBar().showMessage(f"Технологический процесс: {message}", 10000)
        )

        def finished():
            self._background_workers.discard(worker)
            self._pipeline_worker = None

        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()
