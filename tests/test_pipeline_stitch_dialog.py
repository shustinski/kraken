"""Real Qt layer editor preserves pinned ports and execution settings."""
from dataclasses import replace

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QLabel
import pytest

from kraken_hub.presentation.qt.layer_pipeline_dialog import LayerPipelineDialog
from kraken_hub.presentation.qt.pipeline_stitch_editor import edit_stitch_step
from kraken_manager.domain.pipeline import PipelineResources
from kraken_manager.domain.pipeline_layer import PipelineLayerContent, PipelineLayerOperation
from kraken_manager.domain.multilayer import MultilayerModel
from tests.test_stitch_plan_workflow import geometry as geometry
from tests.test_stitch_local_workflow import computing as computing


@pytest.mark.parametrize("language", ["English", "Русский"])
def test_edit_and_persist_layer_node_preserves_static_binding_and_budgets(qtbot, geometry, language):
    local, session, project, layer, versions, args, command = geometry
    model = MultilayerModel.from_dict(command("plan_layer_stitch", args))
    stitch = model.stitches[-1]
    representation = local._projection(project.id).get_artifact_series(versions[0].series_id).representation_id
    from uuid import uuid4
    operation = PipelineLayerOperation(str(uuid4()), "Bound mosaic", "layer.stitch.v1", "1.0.0", stitch.id, stitch.revision,
        content=PipelineLayerContent("image", representation_id=str(representation)),
        parameters={"mode": "gray", "block_size": 32, "max_file_bytes": 4000000},
        resources=PipelineResources(cpu=2, gpu=0, memory_mb=768), quality_gate=False, approval_required=True)
    dialog = LayerPipelineDialog(local, session.principal, project.id, layer.id, language=language)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog._worker is None)
    dialog._operations = [operation.to_dict()]
    dialog.capacity_cpu.setValue(2)
    dialog.capacity_memory.setValue(768)
    dialog._dirty = True
    dialog._redraw()
    dialog.operations.selectRow(0)
    observed = []

    def accept_editor():
        editor = QApplication.activeModalWidget()
        observed.append(editor.objectName())
        content = editor.findChild(QComboBox, "pipelineStitchProducer")
        assert content.currentData()[0] == str(representation)
        editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
        if editor.isVisible():
            observed.append(editor.findChild(QLabel, "pipelineStitchValidation").text())
            editor.reject()

    QTimer.singleShot(20, accept_editor)
    edit_stitch_step(dialog, local.get_multilayer_model(project.id), operation.to_dict())
    assert observed == ["pipelineStitchEditor"]
    assert dialog._operations == [operation.to_dict()]
    dialog.save_definition()
    qtbot.waitUntil(lambda: dialog._worker is None)
    persisted = local.get_layer_pipeline(project.id, layer.id)
    assert persisted["definition"]["operations"] == [operation.to_dict()]
    assert dialog.operations.rowCount() == 1
    # Layer rows select without trying to read nonexistent frame coordinates.
    dialog.operations.selectRow(0)
    assert dialog.operation_title.text() == operation.title
    dialog.close()


def test_layer_editor_rejects_invalid_geometry_selection_without_replacing_content(qtbot, geometry):
    local, session, project, layer, _, args, command = geometry
    model = MultilayerModel.from_dict(command("plan_layer_stitch", args))
    stitch = model.stitches[-1]
    from uuid import uuid4
    operation = PipelineLayerOperation(str(uuid4()), "Unavailable pose", "layer.stitch.v1", "1.0.0", stitch.id, stitch.revision)
    dialog = LayerPipelineDialog(local, session.principal, project.id, layer.id)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: dialog._worker is None)
    dialog._operations = [operation.to_dict()]
    missing = replace(operation, stitch_id=str(uuid4())).to_dict()

    def reject_invalid():
        editor = QApplication.activeModalWidget()
        editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
        assert editor.isVisible()
        editor.reject()

    QTimer.singleShot(20, reject_invalid)
    edit_stitch_step(dialog, local.get_multilayer_model(project.id), missing)
    assert dialog._operations == [operation.to_dict()]
    dialog.close()


def test_desktop_background_controller_executes_layer_only_graph(qtbot, computing):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QMainWindow
    from kraken_hub.presentation.qt.parts.pipeline_controller import PipelineControllerMixin
    from tests.test_pipeline_stitch_runtime import define

    local, session, _, _, _, _, gateway, runner, _ = computing
    _, _, args = define(computing)
    shell = QMainWindow()
    qtbot.addWidget(shell)

    class Controller(PipelineControllerMixin):
        def _gateway_for_job(self, context):
            pytest.fail("A layer-only process must not create a frame selection")

    controller = Controller()
    controller.service, controller.session, controller.shell = local, session, shell
    controller._background_workers = set()
    controller.agent_runtime = SimpleNamespace(ensure_started=lambda: None, base_url=gateway.base_url,
        token=gateway.token, data_dir=gateway.staging_root.parent)
    controller._schedule_pipeline_cycle()
    qtbot.waitUntil(lambda: controller._pipeline_worker is None, timeout=10000)
    assert runner.run_once()
    controller._schedule_pipeline_cycle()
    qtbot.waitUntil(lambda: controller._pipeline_worker is None, timeout=10000)
    assert local.get_layer_pipeline(args["project_id"], args["layer_id"])["status"] == "completed"
    assert not controller._background_workers
