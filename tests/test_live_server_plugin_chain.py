"""Real plugin-chain acceptance, reusing the genuinely trained CPU checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
import pytest
from uuid import uuid4

from kraken_agent.runner import PluginProcessSpec, PluginRegistry
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose
from kraken_server.configuration import unprotect_secret
from tests.test_live_server_worker_acceptance import ObservedWorker, _connect, _open_project, _protected_snapshot


@pytest.fixture
def live_chain(request, tmp_path):
    filename = os.environ.get("KRAKEN_LIVE_WORKER_CONFIG")
    if not filename:
        pytest.skip("Set KRAKEN_LIVE_WORKER_CONFIG for actual plugin chain acceptance")
    config = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
    stage = request.param
    operation, module = {"neuralimage": ("frames.binary-segment.v2", "neuralimage"),
        "karakal": ("karakal.analyze.v1", "karakal.worker"),
        "stitch": ("layer.stitch.v1", "kraken_manager.infrastructure.stitch_worker")}[stage]
    output = Path(config["results_directory"]).parent / "plugin-chain" / stage
    if "reread_only" in request.node.name:
        output = output / "reread-only"
    output.mkdir(parents=True, exist_ok=True)
    remote = _connect(config, tmp_path / "client")
    protected = _protected_snapshot(remote, config["protected_project_id"])
    from kraken_core.settings import configure_settings_directory
    configure_settings_directory(tmp_path / "settings")
    worker = ObservedWorker(server_url=config["server_url"],
        token=unprotect_secret(Path(config[f"{stage}_machine_secret"])), data_dir=output / "worker",
        registry=PluginRegistry({operation: PluginProcessSpec(operation, (sys.executable, "-m", module))}))
    worker.allowed_jobs, worker.transitions = set(), []
    receipt = {"status": "failed", "stage": stage, "operation": operation,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "server_url": config["server_url"], "protected_before": protected}
    state = json.loads((Path(config["results_directory"]) / "receipt.json").read_text(encoding="utf-8"))
    project = remote.get_project(state["project_id"])
    assert state["status"] == "passed" and project.name.startswith("quality-worker-")
    assert str(project.id) != config["protected_project_id"]
    from types import SimpleNamespace
    try:
        yield SimpleNamespace(config=config, worker=worker, remote=remote, output=output,
            receipt=receipt, project=project, state=state, tmp_path=tmp_path)
    finally:
        worker.stop()
        receipt["transitions"] = worker.transitions
        receipt["protected_after"] = _protected_snapshot(remote, config["protected_project_id"])
        receipt["protected_unchanged"] = receipt["protected_after"] == protected
        if not receipt["protected_unchanged"]:
            receipt["status"] = "failed"
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8", newline="\n")
        assert receipt["protected_unchanged"]


def _reopen_chain(env, qtbot, representation_id, outputs):
    for attempt in (1, 2):
        remote = _connect(env.config, env.tmp_path / f"new-session-{attempt}")
        shell, controller, dual = _open_project(qtbot, remote, env.project, env.output, attempt)
        try:
            qtbot.waitUntil(lambda: not controller._background_workers, timeout=15000)
            combo = controller._workspace.image_representation_combo
            index = combo.findData(str(representation_id))
            assert index >= 0
            _select_representation_in_graph(shell, controller, representation_id, qtbot)
            qtbot.waitUntil(lambda: str(combo.currentData()) == str(representation_id))
            qtbot.waitUntil(lambda: not controller._background_workers, timeout=15000)
            matrix = controller._workspace.matrix_view
            expected = {item["id"] for item in outputs}
            qtbot.waitUntil(lambda: {cell.payload.get("artifact_version_id") for cell in matrix._cells.values()} == expected,
                timeout=15000)
            qtbot.waitUntil(lambda: all(cell.thumbnail is not None and not cell.thumbnail.isNull()
                for cell in matrix._cells.values()), timeout=15000)
            for item in outputs:
                path = remote.managed_artifact_path(env.project.id, item["id"])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            assert shell.grab().save(str(env.output / f"hub-{attempt}.png"))
        finally:
            controller._agent_timer.stop()
            controller._workspace._analysis_timer.stop()
            shell.close_guard = None
            shell.close()
            dual.stop_sync()


def _select_representation_in_graph(shell, controller, representation_id, qtbot):
    """Use the visible Layers menu/graph; compatibility combos are hidden."""
    menu_bar = shell.menuBar()
    action = next(a for a in menu_bar.actions() if a.menu() and shell.layers_action in a.menu().actions())
    QTest.mouseClick(menu_bar, Qt.MouseButton.LeftButton, pos=menu_bar.actionGeometry(action).center())
    menu = action.menu()
    qtbot.waitUntil(menu.isVisible)
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(shell.layers_action).center())
    qtbot.waitUntil(lambda: controller._layer_dialog is not None and controller._layer_dialog.isVisible())
    graph = controller._layer_dialog.graph
    for lane in graph._snapshot.lanes:
        graph.expand_lane(lane.lane_id)
    node = next(item for item in graph.graph_scene.items() if getattr(item, "node", None)
                and str(item.node.representation_id) == str(representation_id))
    graph.fitInView(graph.graph_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
    graph.ensureVisible(node)
    QTest.qWait(20)
    point = graph.mapFromScene(node.sceneBoundingRect().center())
    assert graph.isVisible() and graph.viewport().rect().contains(point)
    QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=point)
    qtbot.waitUntil(lambda: str(controller._workspace.image_representation_combo.currentData()) == str(representation_id))
    controller._layer_dialog.close()


@pytest.mark.parametrize("live_chain", ["neuralimage"], indirect=True)
def test_neuralimage_reuses_verified_training_and_publishes_real_inference(live_chain, qtbot):
    env = live_chain
    remote, project, state = env.remote, env.project, env.state
    principal = remote.auth.principal
    layer = next(item for item in remote.list_layers(project.id) if str(item.id) == state["layer_id"])
    reuse = json.loads(Path(env.config["model_reuse_receipt"]).read_text(encoding="utf-8"))
    model = Path(reuse["path"])
    assert hashlib.sha256(model.read_bytes()).hexdigest() == reuse["sha256"]
    assert reuse["original_training"]["execution"]["epoch_upper_bound"] == 1
    series = remote._call("POST", f"/api/v1/projects/{project.id}/artifacts",
        payload={"name": "Verified existing CPU checkpoint", "scope": "layer_attachment", "layer_id": str(layer.id)},
        idempotency_key=str(uuid4()))
    metadata = {"filename": model.name, "media_type": "application/x-pytorch", "sha256": reuse["sha256"],
        "size_bytes": model.stat().st_size, "parameters": {"artifact_role": "model", "reused_training": reuse["original_training"]}}
    key, revision = str(uuid4()), remote.artifact_stream_revision(project.id, series["id"])
    base = f"/api/v1/projects/{project.id}/artifacts/{series['id']}/uploads"
    transfer = remote._call("POST", base, payload=metadata, idempotency_key=key, if_match=revision)
    assert transfer["mode"] == "gateway"
    remote._http.upload_url(transfer["url"], token=transfer["token"], source=model)
    saved = remote._call("POST", base + "/complete", payload={**metadata, "upload_token": transfer["token"]},
        idempotency_key=key, if_match=revision)
    target = remote.create_representation(principal=principal, project=project, layer=layer,
        name=f"Actual NeuralImage inference {uuid4().hex[:6]}", kind=RepresentationKind.IMAGE,
        purpose=RepresentationPurpose.BINARY, idempotency_key=str(uuid4()))
    job = remote.submit_plugin_job(principal=principal, gateway=None, project_id=project.id, layer_id=layer.id,
        source_representation_id=state["image_representation_id"], target_representation_id=target.id,
        coordinates=[(1, 1), (2, 1)], capability="frames.binary-segment.v2",
        parameters={"model_version_id": saved["id"], "patch_size": 128, "execution_mode": "headless",
                    "input_transport": "server", "execution_host": "server"}, idempotency_key=str(uuid4()))
    env.worker.allowed_jobs.add(str(job.id))
    env.receipt.update(project_id=str(project.id), layer_id=str(layer.id), job_id=str(job.id),
        model_id=saved["id"], model_sha256=reuse["sha256"], binary_representation_id=str(target.id))
    assert env.worker.run_once(job_id=str(job.id))
    done = next(item for item in remote.plugin_jobs() if item.id == job.id)
    assert done.state.value == "succeeded", (done.state.value, done.error)
    inputs = {(c.x, c.y): str(c.artifact_version_id)
              for c in remote.frame_cells(project.id, layer.id, state["image_representation_id"])}
    outputs = []
    for cell in remote.frame_cells(project.id, layer.id, target.id):
        version = remote.artifact_version(project.id, cell.artifact_version_id)
        path = remote.managed_artifact_path(project.id, version.id)
        assert version.media_type == "image/png" and version.size_bytes == path.stat().st_size > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == version.sha256
        assert inputs[(cell.x, cell.y)] in set(map(str, version.input_version_ids))
        assert version.parameters["model_sha256"] == reuse["sha256"]
        with Image.open(path) as image:
            assert image.size == (64, 64)
            assert set(image.convert("L").getdata()) <= {0, 255}
        outputs.append({"id": str(version.id), "sha256": version.sha256, "bytes": version.size_bytes,
                        "x": cell.x, "y": cell.y, "inputs": list(map(str, version.input_version_ids))})
    assert len(outputs) == 2
    _reopen_chain(env, qtbot, target.id, outputs)
    env.receipt.update(outputs=outputs, reused_existing_training=True, quality_assessed=False,
        checks=["genuine trained checkpoint SHA verified and published via Gateway", "actual NeuralImage subprocess inference",
                "two binary PNG versions with exact source/model lineage", "two new HTTP+Hub sessions read same bytes"], status="passed")


@pytest.mark.parametrize("live_chain", ["neuralimage"], indirect=True)
def test_neuralimage_hub_reread_only_selects_actual_published_masks(live_chain, qtbot):
    env = live_chain
    saved = json.loads((env.output.parent / "receipt.json").read_text(encoding="utf-8"))
    assert saved["status"] == "passed" and saved["project_id"] == str(env.project.id)
    _reopen_chain(env, qtbot, saved["binary_representation_id"], saved["outputs"])
    env.receipt.update(project_id=str(env.project.id), outputs=saved["outputs"],
        checks=["no inference repeated", "visible Layers menu and graph select published masks in two fresh sessions"],
        status="passed")


def _choose_visible_combo(combo, identifier):
    assert combo.isVisible() and combo.isEnabled()
    index = combo.findData(identifier)
    assert index >= 0
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_Home)
    for _ in range(index):
        QTest.keyClick(combo, Qt.Key.Key_Down)
    assert combo.currentData() == identifier


@pytest.mark.parametrize("live_chain", ["karakal"], indirect=True)
def test_karakal_hub_dialog_runs_machine_analysis_and_reopens_real_metrics(live_chain, qtbot):
    import numpy as np
    from PyQt6.QtWidgets import QDialog, QLabel
    from tests.test_live_ui_workflow import modal

    env = live_chain
    remote, project, state = env.remote, env.project, env.state
    neural = json.loads((env.output.parent / "neuralimage/receipt.json").read_text(encoding="utf-8"))
    assert neural["status"] == "passed" and neural["project_id"] == str(project.id)
    source_id, mask_id = state["image_representation_id"], neural["binary_representation_id"]
    source_cells = remote.frame_cells(project.id, state["layer_id"], source_id)
    mask_cells = {(c.x, c.y): c for c in remote.frame_cells(project.id, state["layer_id"], mask_id)}
    expected = {}
    for cell in source_cells:
        path_a = remote.managed_artifact_path(project.id, cell.artifact_version_id)
        path_b = remote.managed_artifact_path(project.id, mask_cells[(cell.x, cell.y)].artifact_version_id)
        with Image.open(path_a) as a, Image.open(path_b) as b:
            mask_a, mask_b = np.asarray(a.convert("L")) >= 128, np.asarray(b.convert("L")) >= 128
        expected[(cell.x, cell.y)] = float(np.count_nonzero(mask_a & mask_b) / np.count_nonzero(mask_a | mask_b))
    before_runs = {run["run_id"] for run in remote.server_analysis_runs(project.id)}
    shell, controller, dual = _open_project(qtbot, remote, project, env.output, "launch")
    try:
        workspace = controller._workspace
        matrix = workspace.matrix_view
        qtbot.waitUntil(lambda: len(matrix._cells) == 2, timeout=15000)
        for x in (1, 2):
            rect = matrix.scene_rect_for_frame(x, 1)
            matrix.ensureVisible(rect)
            point = matrix.mapFromScene(rect.center())
            assert matrix.viewport().rect().contains(point)
            QTest.mouseClick(matrix.viewport(), Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier if x == 1 else Qt.KeyboardModifier.ControlModifier, pos=point)
        assert set(matrix.selected_coordinates(maximum=10)) == {(1, 1), (2, 1)}

        def configure(dialog):
            assert dialog.objectName() == "analysisSetupDialog" and dialog.frame_count.value() == 2
            _choose_visible_combo(dialog.source_a, source_id)
            _choose_visible_combo(dialog.source_b, mask_id)
            assert dialog.start_button.isVisible() and dialog.start_button.isEnabled()
            QTest.mouseClick(dialog.start_button, Qt.MouseButton.LeftButton)

        assert workspace.evaluate_result_button.isVisible()
        modal(lambda: QTest.mouseClick(workspace.evaluate_result_button, Qt.MouseButton.LeftButton), configure)
        qtbot.waitUntil(lambda: bool({r["run_id"] for r in remote.server_analysis_runs(project.id)} - before_runs), timeout=15000)
        run = next(r for r in remote.server_analysis_runs(project.id) if r["run_id"] not in before_runs)
        run_id = run["run_id"]
        env.receipt.update(project_id=str(project.id), run_id=run_id,
            launched_by="visible Evaluate result button → AnalysisSetupDialog → Start analysis")
        for part in run["partitions"]:
            job_id = part["manifest"]["job_id"]
            env.worker.allowed_jobs.add(job_id)
            assert env.worker.run_once(job_id=job_id)
        completed = remote.server_analysis_run(project.id, run_id)
        assert completed["state"] == "completed" and completed["completed_frames"] == 2 and completed["failed_frames"] == 0
        measurements = remote.server_analysis_frames(project.id, run_id, "iou")
        assert len(measurements) == 2
        for row in measurements:
            assert row["raw_value"] == pytest.approx(expected[(row["x"], row["y"])])
        panel = workspace.analysis_runs_panel
        coordinator = workspace._analysis_coordinator
        if not panel.isVisible():
            QTest.mouseClick(workspace.analysis_history_button, Qt.MouseButton.LeftButton)
        _choose_visible_combo(panel.metric_combo, "iou")
        qtbot.waitUntil(lambda: coordinator._runs.get(run_id, {}).get("state") == "completed", timeout=15000)
        qtbot.waitUntil(lambda: panel.result_table.rowCount() == 2, timeout=15000)
        for row in range(2):
            xy = (int(panel.result_table.item(row, 1).text()), int(panel.result_table.item(row, 2).text()))
            assert float(panel.result_table.item(row, 4).text()) == pytest.approx(expected[xy])
        panel.result_table.selectRow(0)
        frame_id = panel.selected_frame_id()
        assert panel.map_button.isVisible() and panel.map_button.isEnabled()
        QTest.mouseClick(panel.map_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: bool(remote.server_analysis_run(project.id, run_id).get("maps", {}).get(frame_id, {}).get("manifest")),
            timeout=15000)
        map_entry = remote.server_analysis_run(project.id, run_id)["maps"][frame_id]
        map_job = map_entry["manifest"]["job_id"]
        env.worker.allowed_jobs.add(map_job)
        assert env.worker.run_once(job_id=map_job)
        qtbot.waitUntil(lambda: workspace.findChild(QDialog, "remoteAnalysisMapDialog") is not None, timeout=15000)
        dialog = workspace.findChild(QDialog, "remoteAnalysisMapDialog")
        pixmap = dialog.findChild(QLabel, "analysisMapImage").pixmap()
        assert dialog.isVisible() and pixmap.width() == 64 and pixmap.height() == 64
        map_bytes = remote.server_analysis_map_bytes(project.id, run_id, frame_id)
        final = remote.server_analysis_run(project.id, run_id)
        blob = final["maps"][frame_id]["blob"]
        assert len(map_bytes) == blob["size_bytes"] and hashlib.sha256(map_bytes).hexdigest() == blob["sha256"]
        assert shell.grab().save(str(env.output / "hub-analysis.png"))
        dialog.close()
        env.receipt.update(metrics=[dict(row) for row in measurements], expected_iou={str(k): v for k,v in expected.items()},
            worker_runtime=final["worker_runtime"], map_sha256=blob["sha256"], map_bytes=blob["size_bytes"])
    finally:
        qtbot.waitUntil(lambda: not controller._background_workers, timeout=15000)
        controller._workspace._analysis_timer.stop()
        controller._agent_timer.stop()
        shell.close_guard = None
        shell.close()
        dual.stop_sync()
    reopened = _connect(env.config, env.tmp_path / "reopened")
    assert reopened.server_analysis_run(project.id, run_id) == final
    assert reopened.server_analysis_frames(project.id, run_id, "iou") == measurements
    assert reopened.server_analysis_map_bytes(project.id, run_id, frame_id) == map_bytes
    shell, controller, dual = _open_project(qtbot, reopened, project, env.output, "reopen")
    try:
        workspace = controller._workspace
        if not workspace.analysis_runs_panel.isVisible():
            QTest.mouseClick(workspace.analysis_history_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: workspace.analysis_runs_panel.run_table.rowCount() > 0, timeout=15000)
        assert workspace.analysis_runs_panel.run_table.item(0, 0).text() == "completed"
        assert shell.grab().save(str(env.output / "hub-analysis-reopened.png"))
    finally:
        qtbot.waitUntil(lambda: not controller._background_workers, timeout=15000)
        workspace._analysis_timer.stop()
        controller._agent_timer.stop()
        shell.close_guard = None
        shell.close()
        dual.stop_sync()
    env.receipt.update(checks=["actual visible GUI creation and matrix frame selection", "real machine Karakal subprocess",
        "two IoU values independently verified from pinned PNG bytes", "actual GUI metrics and discrepancy map",
        "map hash/size and fresh HTTP+Hub history reread"], quality_assessed=False, status="passed")


@pytest.mark.parametrize("live_chain", ["stitch"], indirect=True)
def test_stitch_real_file_dialog_machine_compute_and_hub_block_view(live_chain, qtbot):
    import numpy as np
    from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog, QLineEdit, QPushButton
    from kraken_core.plugin_protocol import MOSAIC_MEDIA_TYPE
    from kraken_hub.presentation.qt.stitching_dialog import StitchingDialog
    from kraken_manager.domain.multilayer import MultilayerModel
    from kraken_manager.infrastructure.mosaic import MosaicReader
    from tests.test_live_ui_workflow import modal

    env = live_chain
    remote, project, state = env.remote, env.project, env.state
    principal = remote.auth.principal
    layer_id = state["layer_id"]
    current = remote.get_multilayer_model(project.id)
    remote.mutate_multilayer_model(project_id=project.id, principal=principal,
        expected_revision=current["model"]["revision"], operation="set_stack",
        arguments={"layers": [{"layer_id": layer_id, "layer_type": "metal", "z_m": 0, "thickness_m": 0}]},
        idempotency_key=str(uuid4()))
    cells = sorted(remote.frame_cells(project.id, layer_id, state["image_representation_id"]), key=lambda c:c.x)
    assert len(cells) == 2
    tiles, images = [], []
    for cell in cells:
        path = remote.managed_artifact_path(project.id, cell.artifact_version_id)
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB")))
        tile = {"frame_id": cell.frame_id, "artifact_version_id": cell.artifact_version_id, "width": 64, "height": 64,
                "pixel_size_m": [1e-6, 1e-6], "y_axis": "down", "catalog_xy": [cell.x, cell.y]}
        if cell.x == 2:
            tile.update(declared_affine=[1, 0, 64e-6, 0, 1, 0], note="Controlled adjacent synthetic tiles, not real stage calibration")
        tiles.append(tile)
    document = env.output / "observations.json"
    document.write_text(json.dumps({"layer_id": layer_id, "reference_frame_id": cells[0].frame_id,
        "tiles": tiles, "max_error_m": 1e-12}), encoding="utf-8")
    shell, controller, dual = _open_project(qtbot, remote, project, env.output, "compute")
    outer = StitchingDialog(controller, layer_id)
    qtbot.addWidget(outer)
    outer.show()
    old_native = QApplication.testAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    dialog = viewer = None
    try:
        computed_button = outer.findChild(QPushButton, "computedStitchButton")
        assert computed_button.isVisible() and computed_button.isEnabled()
        QTest.mouseClick(computed_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: hasattr(outer, "_computed"))
        dialog = outer._computed
        qtbot.waitUntil(lambda: dialog._worker is None and dialog.snapshot is not None, timeout=15000)

        def choose_file(file_dialog):
            assert isinstance(file_dialog, QFileDialog) and file_dialog.isVisible()
            edit = file_dialog.findChild(QLineEdit, "fileNameEdit")
            assert edit is not None and edit.isVisible()
            file_dialog.setDirectory(str(document.parent))
            file_dialog.selectFile(document.name)
            QTest.qWait(100)
            assert file_dialog.selectedFiles() and Path(file_dialog.selectedFiles()[0]).resolve() == document.resolve()
            button = file_dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Open)
            assert button.isEnabled()
            file_dialog.grab().save(str(env.output / "file-dialog-before-open.png"))
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

        modal(lambda: QTest.mouseClick(dialog.import_button, Qt.MouseButton.LeftButton), choose_file)
        qtbot.waitUntil(lambda: dialog._worker is None and dialog._pending_import is None
            and bool(dialog.snapshot["model"]["stitches"]), timeout=15000)
        latest_geometry = MultilayerModel.from_dict(dialog.snapshot).stitches[-1]
        assert tuple(dialog.plans.currentData()) == (latest_geometry.id, latest_geometry.revision)
        _choose_visible_combo(dialog.mode, "rgb")
        assert dialog.run_button.isEnabled() and dialog.run_button.isVisible()
        QTest.mouseClick(dialog.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: dialog._worker is None and bool(dialog.snapshot["model"]["stitch_renders"]), timeout=15000)
        job_id = dialog.snapshot["model"]["stitch_renders"][-1]["job_id"]
        env.worker.allowed_jobs.add(job_id)
        env.receipt.update(project_id=str(project.id), layer_id=layer_id, job_id=job_id,
            launched_by="StitchingDialog computed button → real QFileDialog JSON → Compute")
        assert env.worker.run_once(job_id=job_id)
        qtbot.waitUntil(lambda: dialog._worker is None
            and dialog.snapshot["model"]["stitch_renders"][-1]["state"] in {"imported", "failed", "cancelled"}, timeout=15000)
        result = dialog.snapshot["model"]["stitch_renders"][-1]
        assert result["state"] == "imported", result.get("error")
        model = MultilayerModel.from_dict(dialog.snapshot)
        render = model.stitch_renders[-1]
        geometry = next(s for s in model.stitches if s.id == render.stitch_id and s.revision == render.stitch_revision)
        version = remote.artifact_version(project.id, render.output.version_id)
        path = remote.managed_artifact_path(project.id, version.id)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == version.sha256
        assert path.stat().st_size == version.size_bytes > 0 and version.media_type == MOSAIC_MEDIA_TYPE
        assert set(map(str, version.input_version_ids)) == {str(c.artifact_version_id) for c in cells} | {geometry.plan.version_id}
        with MosaicReader(path) as reader:
            reader.validate(expected_job_id=job_id)
            assert (reader.metadata["render"]["width"], reader.metadata["render"]["height"]) == (128, 64)
            block = reader.read_block(0, 0)
            assert np.array_equal(block.pixels, np.concatenate(images, axis=1))
            assert np.all(block.coverage == 1)
        QTest.mouseClick(dialog.view_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: hasattr(dialog, "_viewer") and dialog._viewer._worker is None
            and dialog._viewer._metadata is not None, timeout=15000)
        viewer = dialog._viewer
        assert viewer.isVisible() and not viewer.image.pixmap().isNull()
        assert viewer._metadata["job_id"] == job_id and "plan" not in viewer._metadata
        QTest.mouseClick(viewer.coverage, Qt.MouseButton.LeftButton)
        show_block = next(b for b in viewer.findChildren(QPushButton) if b.text() == "Показать блок")
        QTest.mouseClick(show_block, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: viewer._worker is None, timeout=15000)
        assert viewer.image.pixmap().toImage().pixelColor(10, 10).getRgb()[:3] == (255, 255, 255)
        assert viewer.grab().save(str(env.output / "hub-mosaic-block.png"))
        env.receipt.update(output_version_id=str(version.id), sha256=version.sha256, bytes=version.size_bytes,
            media_type=version.media_type, input_version_ids=list(map(str, version.input_version_ids)),
            dimensions=[128, 64], bounded_viewer_metadata=True)
    finally:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, old_native)
        if viewer is not None:
            viewer.close()
        if dialog is not None:
            dialog.timer.stop()
            dialog.close()
        outer.close()
        qtbot.waitUntil(lambda: not controller._background_workers, timeout=15000)
        controller._workspace._analysis_timer.stop()
        controller._agent_timer.stop()
        shell.close_guard = None
        shell.close()
        dual.stop_sync()
    reopened = _connect(env.config, env.tmp_path / "new-stitch-session")
    snapshot = reopened.get_multilayer_model(project.id)
    reread = MultilayerModel.from_dict(snapshot).stitch_renders[-1]
    assert reread == render
    reread_path = reopened.managed_artifact_path(project.id, version.id)
    assert hashlib.sha256(reread_path.read_bytes()).hexdigest() == version.sha256
    env.receipt.update(checks=["actual JSON file dialog import and GUI compute request", "native stitch machine subprocess",
        "immutable mosaic SHA/size/type and exact input lineage", "128x64 pixels equal independently concatenated inputs",
        "real bounded block viewer and coverage toggle", "fresh server session rereads same render/version"], status="passed")


@pytest.mark.parametrize("live_chain", ["stitch"], indirect=True)
def test_stitch_hub_reread_only_opens_saved_mosaic_without_compute(live_chain, qtbot):
    from PyQt6.QtWidgets import QPushButton
    from kraken_hub.presentation.qt.stitching_dialog import StitchingDialog

    env = live_chain
    saved = json.loads((env.output.parent / "receipt.json").read_text(encoding="utf-8"))
    assert saved["status"] == "passed" and saved["project_id"] == str(env.project.id)
    shell, controller, dual = _open_project(qtbot, env.remote, env.project, env.output, "reopen")
    outer = StitchingDialog(controller, saved["layer_id"])
    qtbot.addWidget(outer)
    outer.show()
    dialog = viewer = None
    try:
        QTest.mouseClick(outer.findChild(QPushButton, "computedStitchButton"), Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: hasattr(outer, "_computed"))
        dialog = outer._computed
        qtbot.waitUntil(lambda: dialog._worker is None and dialog.snapshot is not None, timeout=15000)
        assert dialog.view_button.isVisible() and dialog.view_button.isEnabled()
        QTest.mouseClick(dialog.view_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: hasattr(dialog, "_viewer") and dialog._viewer._worker is None
            and dialog._viewer._metadata is not None, timeout=15000)
        viewer = dialog._viewer
        assert viewer.isVisible() and viewer._metadata["job_id"] == saved["job_id"]
        assert not viewer.image.pixmap().isNull()
        path = env.remote.managed_artifact_path(env.project.id, saved["output_version_id"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == saved["sha256"]
        assert viewer.grab().save(str(env.output / "hub-mosaic-reopened.png"))
    finally:
        if viewer is not None:
            viewer.close()
        if dialog is not None:
            dialog.timer.stop()
            dialog.close()
        outer.close()
        qtbot.waitUntil(lambda: not controller._background_workers, timeout=15000)
        controller._workspace._analysis_timer.stop()
        controller._agent_timer.stop()
        shell.close_guard = None
        shell.close()
        dual.stop_sync()
    env.receipt.update(project_id=str(env.project.id), job_id=saved["job_id"], output_version_id=saved["output_version_id"],
        sha256=saved["sha256"], checks=["no computation repeated", "new Hub session opens original mosaic in actual block viewer"], status="passed")
