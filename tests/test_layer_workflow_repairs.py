from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMainWindow

from contour.application.frame_layers import BaseFrameRecord
from contour.application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
from contour.batch_processor import BatchQueueRunnable
from contour.kraken_bridge import (
    ContourKrakenSession,
    KrakenBridgeError,
    _ensure_processing_finished,
    recognition_parameters,
)
from kraken_core.plugin_protocol import PluginFrameInput, PluginJobManifest


def test_contour_close_waits_for_nested_background_workers_without_blocking_gui(app):
    import threading
    import time

    from PyQt6.QtCore import QObject, QRunnable, QThreadPool
    from contour.widget import _background_pools_are_idle

    owner = QObject()
    nested = QObject(owner)
    pool = QThreadPool(nested)
    entered = threading.Event()
    release = threading.Event()

    class Worker(QRunnable):
        def run(self):
            entered.set()
            release.wait(10)

    pool.start(Worker())
    try:
        assert entered.wait(5)
        started = time.monotonic()
        assert not _background_pools_are_idle(owner)
        assert time.monotonic() - started < 1
    finally:
        release.set()
        assert pool.waitForDone(5000)
    assert _background_pools_are_idle(owner)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("operation", ["frames.vectorize.v1", "frames.dataset.prepare.v1"])
def test_local_contour_uses_manifest_coordinates_without_dropping_same_row_frames(tmp_path, app, operation):
    payload = b"image"
    inputs = []
    for index, (x, y) in enumerate(((2, 1), (1, 1), (3, 4))):
        relative = f"inputs/{x}_{y}.png"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
        inputs.append(
            PluginFrameInput(
                frame_id=f"frame-{index}",
                artifact_version_id=f"version-{index}",
                relative_path=relative,
                sha256=hashlib.sha256(payload).hexdigest(),
                media_type="image/png",
                x=x,
                y=y,
            )
        )
    manifest = PluginJobManifest(
        job_id="job",
        actor_id="owner",
        project_id="project",
        layer_id="layer",
        target_representation_id="target",
        operation=operation,
        inputs=tuple(inputs),
        parameters={"project_width": 10},
    )
    job_path = tmp_path / "job.json"
    job_path.write_text(manifest.to_json(), encoding="utf-8")
    session = ContourKrakenSession.load(
        job_manifest=job_path,
        result_manifest=tmp_path / "result.json",
        staging_root=tmp_path,
    )
    window = QMainWindow()
    records = []

    def load(paths, **_kwargs):
        records.extend(BaseFrameRecord(p, Path(p).stem, window._widget._source_frame_numbers[p]) for p in paths)

    window._widget = SimpleNamespace(load_images=load)
    session.attach_return_action(window)
    assert [r.frame_number for r in records] == [1, 0, 32]
    assert len({r.path for r in records}) == len(inputs)
    assert tuple(r.path for r in records) == tuple(map(str, session.input_paths))
    window.close()


@pytest.mark.parametrize("count,workers,chunk", [(0, 4, 16), (1, 4, 16), (35, 2, 4), (100, 4, 8)])
def test_batch_tail_is_distributed_without_reordering_or_repeating_frames(count, workers, chunk):
    paths = [f"frame-{index}.png" for index in range(count)]
    runner = BatchQueueRunnable(
        paths,
        {"steps": []},
        ContourExtractionSettings(),
        None,
        SaveOptions(),
        DisplaySettings(),
        workers,
        "en",
        chunk_size=chunk,
    )
    requests = runner._build_chunk_requests()
    assert [path for request in requests for path in request.image_paths] == paths
    assert [r.chunk_id for r in requests] == list(range(1, len(requests) + 1))
    assert all(len(r.image_paths) <= chunk for r in requests)
    if requests:
        assert len(requests[-1].image_paths) == 1


def test_recognition_parameters_use_contour_import_format():
    settings = ContourExtractionSettings(min_area=42)
    window = SimpleNamespace(
        _widget=SimpleNamespace(
            get_pipeline=lambda: {"steps": []},
            _current_contour_settings=lambda: settings,
        )
    )
    payload = recognition_parameters(window)
    assert payload["format"] == "contour-recognition-settings"
    assert ContourExtractionSettings.from_dict(payload["recognition_settings"]).min_area == 42


@pytest.mark.parametrize("batch,preview", [(True, None), (False, 7), (True, 7)])
def test_return_rejects_unfinished_processing(batch, preview):
    window = SimpleNamespace(
        _widget=SimpleNamespace(
            _batch_processor=SimpleNamespace(is_running=batch),
            _preview_running_request_id=preview,
        )
    )
    with pytest.raises(KrakenBridgeError, match="Дождитесь"):
        _ensure_processing_finished(window)
    window._widget._batch_processor.is_running = False
    window._widget._preview_running_request_id = None
    _ensure_processing_finished(window)


def test_training_parameters_preserve_native_import_format_and_reject_ambiguous_runs(tmp_path):
    import json
    from neuralimage.kraken_bridge import KrakenBridgeError as NeuralBridgeError, _training_parameters
    from neuralimage.application.dto import MainWindowState, SettingsState
    from neuralimage.infrastructure.config.state_store import create_workflow_snapshot_payload, load_workflow_snapshot

    payload = create_workflow_snapshot_payload(MainWindowState(), SettingsState())
    path = tmp_path / "neuralimage_workflow.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = _training_parameters(tmp_path)
    assert restored == payload
    exported = tmp_path / "parameters.json"
    exported.write_text(json.dumps(restored), encoding="utf-8")
    main, settings = load_workflow_snapshot(exported)
    assert main.epochs == MainWindowState().epochs
    other = tmp_path / "other"
    other.mkdir()
    (other / path.name).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NeuralBridgeError, match="одного обучения"):
        _training_parameters(tmp_path)


@pytest.mark.parametrize("editor_current", [False, True])
@pytest.mark.parametrize("save_method", ["dataset", "result"])
def test_saving_during_frame_switch_never_uses_previous_frames_vectors(monkeypatch, editor_current, save_method):
    from contour.domain import PolygonData
    from contour.application.processing import ImageProcessingState
    from contour.application.services.workspace_session import WorkspaceSession
    from contour.widget_parts.processing_mixin_parts.export_dataset_frame_state import ExportDatasetFrameStateMixin
    from contour.widget_parts.processing_mixin_parts import save_current_result as save_module

    correct = PolygonData(id=1, points=[(1, 1), (5, 1), (5, 5)])
    stale = PolygonData(id=2, points=[(100, 100), (200, 100), (200, 200)])
    workspace = WorkspaceSession()
    workspace._current_image_path = "new.png"
    workspace._current_state = ImageProcessingState(image_path="new.png", polygons=[correct])
    captured = []
    widget = SimpleNamespace(
        _workspace=workspace,
        _editor_polygons_are_current_frame=lambda: editor_current,
        get_polygons=lambda: [stale],
        _update_frame_item_status=lambda *_: None,
        _export_dataset_frame_for_state=lambda path, state, polygons, dataset_directory: captured.extend(polygons),
    )
    if save_method == "dataset":
        ExportDatasetFrameStateMixin.export_current_frame_to_dataset(widget, "dataset")
    else:
        widget._current_contour_settings = lambda: ContourExtractionSettings()
        widget._current_save_options = lambda: SaveOptions()
        widget._display_settings = DisplaySettings()
        widget.get_pipeline = lambda: {"steps": []}
        monkeypatch.setattr(save_module, "save_result_bundle", lambda **kw: captured.extend(kw["polygons"]) or {})
        save_module.SaveCurrentResultMixin.save_current_result(widget, "results")
    assert [p.id for p in captured] == ([2] if editor_current else [1])
    assert [p.id for p in workspace.current_state.polygons] == ([2] if editor_current else [1])


def test_optimized_cif_slit_matches_original_search_including_ties():
    import math
    from shapely.geometry import LineString, Polygon
    from contour.serializers_parts.boundary_vertices_by_distance import (
        _boundary_vertices_by_distance,
        _point_distance_squared,
        _rightmost_point_index,
    )
    from contour.serializers_parts.select_vertex_slit_hole import _select_vertex_slit_for_hole

    def original(hole, solid, outer, linked):
        best = None
        anchor = _rightmost_point_index(hole)
        for hole_index in [anchor] + [i for i in range(len(hole)) if i != anchor]:
            for key, ring in [(-1, outer), *linked]:
                for target, index in _boundary_vertices_by_distance(ring, hole[hole_index], prefer_same_x_side=False):
                    distance = _point_distance_squared(hole[hole_index], target)
                    if best is not None and distance >= best[0]:
                        continue
                    if solid.covers(LineString([target, hole[hole_index]])):
                        best = distance, key, target, index, hole_index, [target, hole[hole_index]]
        return None if best is None else best[1:]

    for vertices in (4, 19, 128):
        outer = [
            (100 * math.cos(i * 2 * math.pi / vertices), 100 * math.sin(i * 2 * math.pi / vertices))
            for i in range(vertices)
        ]
        for shift in (-30, 0, 30):
            hole = [(shift - 5, -5), (shift + 5, -5), (shift + 5, 5), (shift - 5, 5)]
            sibling = [(shift - 5, 15), (shift + 5, 15), (shift + 5, 25), (shift - 5, 25)]
            solid = Polygon(outer, holes=[hole, sibling])
            for linked in ([], [(0, sibling)]):
                expected = original(hole, solid, outer, linked)
                assert _select_vertex_slit_for_hole(hole, solid, outer=outer, linked_hole_rings=linked) == expected


def test_inference_parameter_export_restores_effective_runtime_settings():
    from neuralimage.kraken_bridge import HeadlessOptions
    from neuralimage.application.dto import MainWindowState, SettingsState
    from neuralimage.application.services.inference_settings import restore_inference_settings
    from neuralimage.application.services.workflow_mapper import build_workflow_parameters

    options = HeadlessOptions(Path("model.pth"), "a" * 64, "test", (128, 256), 3, 12, False, 0.73, True, True, 5)
    current = SettingsState(batch_size=16, train_batch_size=19, model="keep-training-model")
    restored = restore_inference_settings(options.provenance(), current)
    assert restored.train_batch_size == 19
    assert restored.model == "keep-training-model"
    assert current.recognition_threshold == 0.5
    _, _, recognition = build_workflow_parameters(MainWindowState(work_mode="recognition_only"), restored)
    assert recognition.part_size == (128, 256)
    assert recognition.batch_size == 3
    assert recognition.threshold == 0.73
    assert recognition.overlap == 12


@pytest.mark.parametrize(
    "key,value",
    [
        ("patch_size", [True, 256]),
        ("threshold", float("nan")),
        ("batch_size", 0),
        ("tta", "false"),
        ("overlap", 256),
        ("version", 999),
    ],
)
def test_inference_import_rejects_invalid_parameters_without_changing_settings(key, value):
    from neuralimage.kraken_bridge import HeadlessOptions
    from neuralimage.application.dto import SettingsState
    from neuralimage.application.services.inference_settings import restore_inference_settings

    payload = HeadlessOptions(
        Path("model.pth"), "a" * 64, "test", (256, 256), 3, 12, False, 0.73, True, True, 5
    ).provenance()
    payload[key] = value
    current = SettingsState()
    with pytest.raises(ValueError):
        restore_inference_settings(payload, current)
    assert current == SettingsState()


def test_theme_preferences_apply_and_restore_all_choices(tmp_path, monkeypatch, app):
    from kraken_hub.presentation.qt import preferences_dialog as preferences
    from kraken_core.theme import apply_app_theme, shared_theme_stylesheet

    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(preferences, "program_settings", lambda: settings)
    monkeypatch.setattr(preferences, "load_program_accounts", lambda *_args: [])
    monkeypatch.setattr(preferences, "save_program_accounts", lambda *_args: None)
    previous = app.property("krakenTheme") or "dark"
    try:
        for theme in ("system", "light", "dark"):
            dialog = preferences.ProgramPreferencesDialog()
            assert [dialog._theme.itemText(i) for i in range(3)] == ["Как в системе", "Светлая", "Темная"]
            dialog._theme.setCurrentIndex(dialog._theme.findData(theme))
            assert dialog._apply()
            assert settings.value("appearance/theme") == theme
            assert app.styleSheet() == shared_theme_stylesheet(theme)
            dialog.close()
            restored = preferences.ProgramPreferencesDialog()
            assert restored._theme.currentData() == theme
            restored.close()
    finally:
        apply_app_theme(previous, app)


def test_light_theme_repaints_graphics_scenes(app):
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QColor, QImage, QPainter

    from kraken_core.frame_matrix.qt import FrameMatrixView
    from kraken_core.theme import apply_app_theme, scene_color
    from kraken_manager.presentation.qt.layer_management_parts.pipeline_scene import PipelineScene
    from kraken_manager.presentation.qt.role_management import RoleGraphView

    previous = app.property("krakenTheme") or "dark"
    matrix = FrameMatrixView()
    pipeline = PipelineScene()
    roles = RoleGraphView()
    try:
        apply_app_theme("light", app)
        assert matrix.backgroundBrush().color().name().lower() == QColor(scene_color("matrix")).name().lower()
        assert matrix._status_colors["empty"] == scene_color("empty")
        for surface in (pipeline, roles):
            image = QImage(8, 8, QImage.Format.Format_ARGB32)
            image.fill(0)
            painter = QPainter(image)
            surface.drawBackground(painter, QRectF(0, 0, 8, 8))
            painter.end()
            assert image.pixelColor(3, 3).name().lower() == QColor(scene_color("pipeline")).name().lower()
        apply_app_theme("dark", app)
        assert matrix.backgroundBrush().color().name().lower() == "#0b1120"
        image = QImage(8, 8, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        pipeline.drawBackground(painter, QRectF(0, 0, 8, 8))
        painter.end()
        assert image.pixelColor(3, 3).name().lower() == "#1b1d1b"
    finally:
        apply_app_theme(previous, app)
        matrix.deleteLater()
        roles.deleteLater()


def test_theme_cancel_does_not_apply(tmp_path, monkeypatch, app):
    from kraken_hub.presentation.qt import preferences_dialog as preferences

    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    settings.setValue("appearance/theme", "light")
    monkeypatch.setattr(preferences, "program_settings", lambda: settings)
    monkeypatch.setattr(preferences, "load_program_accounts", lambda *_args: [])
    dialog = preferences.ProgramPreferencesDialog()
    dialog._theme.setCurrentIndex(dialog._theme.findData("dark"))
    dialog.reject()
    assert settings.value("appearance/theme") == "light"


def test_pipeline_uses_applied_parameters_and_human_titles():
    from datetime import UTC, datetime
    from kraken_hub.presentation.qt.manager_app import DesktopController
    from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose

    now = datetime.now(UTC)
    source = SimpleNamespace(
        id="source",
        name="Images",
        note="Source image note",
        kind=RepresentationKind.IMAGE,
        purpose=RepresentationPurpose.SOURCE,
        active=True,
        source="managed-import",
        created_at=now,
    )
    manifest = {"layer_id": "layer", "capability": "frames.vectorize.v1", "parameters": {"min_area": 1}}
    job = {"id": "job", "layer_id": "layer", "state": "succeeded"}
    applied = {"recognition_settings": {"min_area": 42}}
    events = (
        SimpleNamespace(
            event_type="PluginJobCreated",
            event_id="event",
            recorded_at=now,
            actor=SimpleNamespace(display_name="Owner"),
            payload={"plugin_job_id": "job", "manifest": manifest, "job": job},
        ),
        SimpleNamespace(
            event_type="PluginResultImported",
            event_id="result",
            recorded_at=now,
            actor=SimpleNamespace(display_name="Owner"),
            payload={"plugin_job_id": "job", "job": job, "result": {"parameters_applied": applied}},
        ),
    )
    controller = object.__new__(DesktopController)
    controller.service = SimpleNamespace(
        list_representations=lambda *_: (source,),
        history=lambda *_: events,
        latest_karakal_analysis=lambda *_: None,
        list_derived_runs=lambda *_: (),
    )
    snapshot = controller._pipeline_snapshot("project", "layer")
    assert snapshot.lanes[0].nodes[0].details["note"] == "Source image note"
    node = next(n for n in snapshot.lanes[0].nodes if n.node_id == "job")
    assert node.title == "Векторизация изображений"
    assert node.subtitle == "Завершено"
    assert node.details["параметры"] == applied
