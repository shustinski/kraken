"""Shown model/version selection and its actual Hub launch parameter boundary."""
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PyQt6.QtWidgets import QDialogButtonBox, QFileDialog, QMessageBox

from kraken_hub.layer_pipeline_dialog import LayerPipelineDialog
from kraken_hub.published_model_dialog import PublishedModelDialog
from tests.test_pipeline_model_inputs import catalog as catalog
from tests.test_layer_pipeline_dialog import wait_done


@pytest.mark.parametrize("language", ["Русский", "English"])
def test_shown_selector_names_model_and_source_versions_and_retains_file_option(qtbot, catalog, language):
    service, _, project, layer, op, model = catalog
    workers = set()
    dialog = PublishedModelDialog(service, project.id, layer.id, source_representation_id=op.source_representation_id,
        preferred_version_id=model.id, language=language, background_workers=workers)
    qtbot.addWidget(dialog)
    dialog.show()
    wait_done(qtbot, dialog)
    assert not workers
    assert dialog.models.currentData() == model.id
    assert "published.pth" in dialog.models.currentText()
    assert model.id in dialog.details.toPlainText()
    assert model.sha256 in dialog.details.toPlainText()
    assert len(dialog._sources) == 2
    assert all(identifier in dialog.details.toPlainText() for _, identifier, _ in dialog._sources)
    assert dialog.parameters() == {"model_version_id": model.id}
    dialog.models.setCurrentIndex(dialog.models.findData("legacy"))
    assert dialog.parameters() == {"legacy_file": True}
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_workflow_editor_preserves_named_model_binding_and_shows_pinned_attempt_versions(qtbot, catalog):
    service, actor, project, layer, op, model = catalog
    dialog = LayerPipelineDialog(service, actor, project.id, layer.id, language="English")
    qtbot.addWidget(dialog)
    dialog.show()
    wait_done(qtbot, dialog)
    dialog._operations = [op.to_dict()]
    dialog._state = {"status": "draft", "nodes": {op.id: {"status": "running", "attempts": [{
        "request": {"input_heads": {model.series_id: model.id, "image-series": "image-version"},
                    "parameters": {"model_version_id": model.id}},
    }]}}}
    dialog._redraw()
    dialog.operations.selectRow(0)
    assert "published.pth" in dialog.model_input.currentText()
    assert model.sha256 in dialog.model_details.text()
    assert dialog._operation_payload(op.id)["model_input"] == {"version_id": model.id}
    assert model.id in dialog.attempts.item(0, 4).text()
    assert "image-version" in dialog.attempts.item(0, 4).text()
    assert dialog.attempts.item(0, 5).text() == model.id
    from tests.test_pipeline_execution import operation
    training = operation(capability="dataset.model.train.v1", input_kind="dataset", output_kind="model", title="Named training")
    dialog._operations.append(training.to_dict())
    dialog._refresh_dependencies()
    dialog.model_input.setCurrentIndex(dialog.model_input.findData(f"producer:{training.id}"))
    payload = dialog._operation_payload(op.id)
    assert payload["model_input"] == {"producer_id": training.id}
    assert training.id in payload["dependencies"]


def controller(monkeypatch, *, action):
    from kraken_hub.manager_app_parts.layer_manager_action import LayerManagerActionMixin
    recorded, submitted = [], []
    node = SimpleNamespace(node_id="model-node", kind="model", representation_id=None, details={})
    obj = SimpleNamespace(
        _project_id=str(uuid4()), shell=None, _layer_dialog=None, _background_workers=set(),
        session=SimpleNamespace(principal="authorized-owner"),
        service=SimpleNamespace(record_layer_pipeline_action=lambda **kwargs: recorded.append(kwargs)),
        _action_availability=lambda _: (True, ""),
        _action_requirement=lambda _: ("neuralimage", "frames.confidence.v1" if action == "confidence" else "frames.binary-segment.v2", "headless"),
        _pipeline_snapshot=lambda *_: SimpleNamespace(lanes=[SimpleNamespace(lane_id="images", nodes=[node])]),
        _ensure_agent_source_ready=lambda **_: True,
        _submit_agent_action=lambda **kwargs: submitted.append(kwargs) or SimpleNamespace(id="job"),
        _error=lambda message: pytest.fail(message),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    return obj, node, submitted, recorded, LayerManagerActionMixin


@pytest.mark.parametrize("action", ["recognize", "confidence"])
def test_hub_published_selection_submits_reference_without_local_model_path(monkeypatch, action):
    obj, node, submitted, recorded, mixin = controller(monkeypatch, action=action)
    class Selected:
        def __init__(self, *args, **kwargs):
            pass
        def exec(self):
            return True
        def parameters(self):
            return {"model_version_id": "exact-published-version"}
    monkeypatch.setattr("kraken_hub.published_model_dialog.PublishedModelDialog", Selected)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: pytest.fail("Published model opened file picker"))
    mixin._layer_manager_action(obj, "layer", node, action)
    assert submitted[0]["parameters"]["model_version_id"] == "exact-published-version"
    assert "model_source_path" not in submitted[0]["parameters"]
    assert recorded[0]["parameters"]["model_version_id"] == "exact-published-version"


def test_explicit_external_file_action_keeps_its_legacy_hash_pinned_contract(monkeypatch, tmp_path):
    obj, node, submitted, _, mixin = controller(monkeypatch, action="recognize_external")
    path = tmp_path / "legacy.pth"
    path.write_bytes(b"not a trained model: selection contract fixture")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    monkeypatch.setattr("kraken_hub.published_model_dialog.PublishedModelDialog", lambda *a, **k: pytest.fail("Explicit file action changed"))
    mixin._layer_manager_action(obj, "layer", node, "recognize_external")
    parameters = submitted[0]["parameters"]
    assert parameters["model_source_path"] == str(path.resolve())
    assert len(parameters["model_sha256"]) == 64
    assert "model_version_id" not in parameters


def test_selecting_legacy_option_opens_file_workflow_instead_of_exporting_managed_model(monkeypatch, tmp_path):
    obj, node, submitted, _, mixin = controller(monkeypatch, action="recognize")
    path = tmp_path / "chosen.pth"
    path.write_bytes(b"file selection fixture")
    class Selected:
        def __init__(self, *args, **kwargs):
            pass
        def exec(self):
            return True
        def parameters(self):
            return {"legacy_file": True}
    monkeypatch.setattr("kraken_hub.published_model_dialog.PublishedModelDialog", Selected)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    mixin._layer_manager_action(obj, "layer", node, "recognize")
    assert submitted[0]["parameters"]["model_source_path"] == str(path.resolve())


def test_model_version_selected_in_editor_persists_in_real_workflow(qtbot, catalog):
    service, actor, project, layer, op, model = catalog
    dialog = LayerPipelineDialog(service, actor, project.id, layer.id)
    qtbot.addWidget(dialog)
    dialog.show()
    wait_done(qtbot, dialog)
    dialog._operations = [op.to_dict()]
    dialog._dirty = True
    dialog.save_definition()
    wait_done(qtbot, dialog)
    definition = service.get_layer_pipeline(project.id, layer.id)["definition"]
    assert definition["operations"][0]["model_input"] == {"version_id": model.id}


def test_selector_cannot_close_while_catalog_worker_is_active(qtbot, monkeypatch):
    from threading import Event
    started, release = Event(), Event()
    def load(self):
        started.set()
        release.wait(3)
        return (), ()
    monkeypatch.setattr(PublishedModelDialog, "_load", load)
    workers = set()
    dialog = PublishedModelDialog(None, "project", "layer", background_workers=workers)
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        qtbot.waitUntil(started.is_set)
        dialog.close()
        assert dialog.isVisible() and workers
    finally:
        release.set()
    wait_done(qtbot, dialog)
    assert not workers
    dialog.close()
    assert not dialog.isVisible()
