import json
import threading
from types import SimpleNamespace

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow

from kraken_hub.infrastructure.agent.runtime import LocalAgentRuntime
from kraken_hub.presentation.qt.computed_stitch_dialog import ComputedStitchDialog, _STITCH_CALLS
from tests.test_stitch_plan_workflow import geometry as geometry


def test_real_qt_request_builtin_agent_import_and_bounded_view(qtbot, geometry, tmp_path, monkeypatch):
    local, session, project, layer, _, args, command = geometry
    command("plan_layer_stitch", args)
    runtime = LocalAgentRuntime(tmp_path / "real-agent", builtin_stitching=True)
    shell = QMainWindow()
    qtbot.addWidget(shell)
    workers = set()
    controller = SimpleNamespace(shell=shell, service=local, session=session, _project_id=project.id,
        agent_runtime=runtime, _background_workers=workers)
    calls = []
    original = local.get_multilayer_model

    def measured(*args):
        calls.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(local, "get_multilayer_model", measured)
    dialog = ComputedStitchDialog(controller, layer.id)
    qtbot.addWidget(dialog)
    dialog.timer.setInterval(75)
    dialog.show()
    try:
        qtbot.waitUntil(lambda: dialog._worker is None and dialog.snapshot is not None, timeout=10000)
        assert dialog.plans.count() == 1
        assert "current" in dialog.plans.currentText()
        qtbot.mouseClick(dialog.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: dialog.snapshot is not None and bool(dialog.snapshot["model"]["stitch_renders"])
            and dialog.snapshot["model"]["stitch_renders"][-1]["state"] == "imported", timeout=20000)
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=5000)
        dialog.timer.stop()
        qtbot.mouseClick(dialog.view_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: hasattr(dialog, "_viewer"), timeout=10000)
        viewer = dialog._viewer
        qtbot.waitUntil(lambda: viewer._worker is None and viewer._metadata is not None, timeout=10000)
        assert viewer.image.pixmap() is not None and not viewer.image.pixmap().isNull()
        assert viewer._metadata["render"]["width"] >= 16
        viewer.coverage.setChecked(True)
        viewer.refresh()
        qtbot.waitUntil(lambda: viewer._worker is None, timeout=5000)
        assert viewer.image.pixmap() is not None
        assert calls and all(identifier != threading.get_ident() for identifier in calls)
        assert "layer.stitch.v1" in runtime.protocol_by_capability
        viewer.close()
        dialog.close()
        qtbot.waitUntil(lambda: not workers and not _STITCH_CALLS, timeout=10000)
    finally:
        dialog.timer.stop()
        qtbot.waitUntil(lambda: not workers, timeout=10000)
        runtime.shutdown()


def test_observation_document_is_exact_managed_evidence_and_idempotent(geometry, tmp_path):
    local, session, project, layer, _, args, _ = geometry
    document = {k: v for k, v in args.items() if k != "plan_version_id"}
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    payload = local.import_stitch_observations(project_id=project.id, layer_id=layer.id, principal=session.principal,
        source_path=path, idempotency_key="document")
    anchor = payload["model"]["stitches"][-1]["plan"]
    assert local.read_project_blob(project.id, anchor["sha256"]) == path.read_bytes()
    history = local.history(project.id)
    assert local.import_stitch_observations(project_id=project.id, layer_id=layer.id, principal=session.principal,
        source_path=path, idempotency_key="document") == payload
    assert local.history(project.id) == history


def test_lost_mutation_reply_reuses_key_and_does_not_duplicate_request(qtbot, geometry, tmp_path, monkeypatch):
    local, session, project, layer, _, args, command = geometry
    command("plan_layer_stitch", args)
    shell = QMainWindow()
    qtbot.addWidget(shell)
    workers = set()
    controller = SimpleNamespace(shell=shell, service=local, session=session, _project_id=project.id,
        agent_runtime=LocalAgentRuntime(tmp_path / "unused-agent", builtin_stitching=True), _background_workers=workers)
    dialog = ComputedStitchDialog(controller, layer.id)
    qtbot.addWidget(dialog)
    dialog.timer.stop()
    qtbot.waitUntil(lambda: dialog._worker is None and dialog.snapshot is not None, timeout=10000)
    original, keys = local.mutate_multilayer_model, []

    def lost(**kwargs):
        result = original(**kwargs)
        keys.append(kwargs["idempotency_key"])
        if len(keys) == 1:
            raise OSError("lost request reply")
        return result

    monkeypatch.setattr(local, "mutate_multilayer_model", lost)
    dialog.run()
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
    assert dialog._pending is not None and "lost request reply" in dialog.status.text()
    dialog.refresh()
    qtbot.waitUntil(lambda: dialog._worker is None and dialog._pending is None, timeout=10000)
    assert len(keys) == 2 and keys[0] == keys[1]
    assert len(dialog.snapshot["model"]["stitch_renders"]) == 1
    dialog.close()
    qtbot.waitUntil(lambda: not workers and not _STITCH_CALLS, timeout=10000)


def test_deleting_dialog_during_background_read_keeps_worker_alive(qtbot, geometry, tmp_path, monkeypatch):
    from PyQt6 import sip
    local, session, project, layer, _, _, _ = geometry
    entered, release = threading.Event(), threading.Event()
    original = local.get_multilayer_model

    def slow(project_id):
        entered.set()
        assert release.wait(10)
        return original(project_id)

    monkeypatch.setattr(local, "get_multilayer_model", slow)
    shell = QMainWindow()
    qtbot.addWidget(shell)
    workers = set()
    controller = SimpleNamespace(shell=shell, service=local, session=session, _project_id=project.id,
        agent_runtime=LocalAgentRuntime(tmp_path / "unused-agent"), _background_workers=workers)
    dialog = ComputedStitchDialog(controller, layer.id)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    dialog.deleteLater()
    qtbot.waitUntil(lambda: sip.isdeleted(dialog), timeout=5000)
    assert workers and _STITCH_CALLS
    release.set()
    qtbot.waitUntil(lambda: not workers and not _STITCH_CALLS, timeout=10000)


def test_definitive_validation_rejection_releases_mutation_for_correction(qtbot, geometry, tmp_path):
    import copy
    local, session, project, layer, _, args, command = geometry
    bad = copy.deepcopy(args)
    bad["tiles"][1]["control_points"] = bad["tiles"][1]["control_points"][:2]
    command("plan_layer_stitch", bad)
    shell = QMainWindow()
    qtbot.addWidget(shell)
    workers = set()
    controller = SimpleNamespace(shell=shell, service=local, session=session, _project_id=project.id,
        agent_runtime=LocalAgentRuntime(tmp_path / "unused-agent"), _background_workers=workers)
    dialog = ComputedStitchDialog(controller, layer.id)
    qtbot.addWidget(dialog)
    dialog.timer.stop()
    qtbot.waitUntil(lambda: dialog._worker is None and dialog.snapshot is not None, timeout=10000)
    dialog.run()
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
    assert dialog._pending is None and "unresolved" in dialog.status.text()
    assert local.get_multilayer_model(project.id)["model"]["stitch_renders"] == []
    dialog.refresh()
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
    assert "unresolved" in dialog.status.text()
    dialog.close()
