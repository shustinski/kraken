"""Shown Qt workflow editor uses durable services and worker close guards."""

import pytest

from kraken_hub.presentation.qt.layer_pipeline_dialog import LayerPipelineDialog
from tests.test_pipeline_runtime_integration import setup


def wait_done(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)


@pytest.mark.parametrize("language", ["English", "Русский"])
def test_shown_workflow_commands_persist_and_history_is_visible(qtbot, tmp_path, language):
    local, actor, project, layer, definition, state = setup(tmp_path)
    workers = set()
    dialog = LayerPipelineDialog(local, actor, project.id, layer.id, language=language, background_workers=workers)
    qtbot.addWidget(dialog)
    dialog.show()
    wait_done(qtbot, dialog)
    assert dialog.operations.rowCount() == 1
    assert dialog.history.rowCount() >= 2
    assert dialog._state["status"] == "running"
    dialog.command("pause")
    assert workers
    dialog.close()
    assert dialog.isVisible()
    wait_done(qtbot, dialog)
    assert not workers
    assert local.get_layer_pipeline(project.id, layer.id)["status"] == "paused"
    dialog.command("resume")
    wait_done(qtbot, dialog)
    assert local.get_layer_pipeline(project.id, layer.id)["status"] == "running"
    dialog.command("cancel")
    wait_done(qtbot, dialog)
    assert local.get_layer_pipeline(project.id, layer.id)["status"] == "cancelling"
    dialog.close()
    assert not dialog.isVisible()


def test_local_controller_continues_after_workflow_dialog_is_closed(qtbot, tmp_path):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QMainWindow
    from kraken_hub.presentation.qt.parts.pipeline_controller import PipelineControllerMixin
    from tests.test_pipeline_runtime_integration import Gateway

    local, actor, project, layer, definition, state = setup(tmp_path)
    shell = QMainWindow()
    qtbot.addWidget(shell)
    gateway = Gateway()

    class Controller(PipelineControllerMixin):
        def _gateway_for_job(self, context):
            assert context.project_id == project.id
            return gateway

    controller = Controller()
    controller.service, controller.session, controller.shell = local, SimpleNamespace(principal=actor), shell
    controller._background_workers = set()
    controller._schedule_pipeline_cycle()
    qtbot.waitUntil(lambda: controller._pipeline_worker is None, timeout=10000)
    assert not controller._background_workers
    assert len(gateway.manifests) == 1
    assert local.get_layer_pipeline(project.id, layer.id)["nodes"][definition.operations[0].id]["status"] == "running"
    controller._schedule_pipeline_cycle()
    qtbot.waitUntil(lambda: controller._pipeline_worker is None, timeout=10000)
    assert len(gateway.deliveries) == 1
