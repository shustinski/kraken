from __future__ import annotations

import cProfile

import numpy as np

from contour.domain import PolygonData, compute_polygon_metrics
from contour.infrastructure import profiling
from contour.infrastructure.polygon_change_profiler import PolygonChangeProfile


def _some_polygon_work() -> int:
    return sum(index * index for index in range(200))


def test_polygon_change_profile_reports_operation_and_phases() -> None:
    session = PolygonChangeProfile.begin(operation="erase_brush", polygon_count_before=3)
    assert _some_polygon_work() > 0
    started_at = session.started_at
    session.note_phase("boolean", started_at)
    session.set_metadata(changed=True, result_polygons=2, overlapping=1)
    session.finish()

    summary = session.format_summary(status="completed")

    assert "[contour polygon change profiling]" in summary
    assert "operation=erase_brush" in summary
    assert "polygons_before=3" in summary
    assert "changed=True" in summary
    assert "result_polygons=2" in summary
    assert "boolean=" in summary
    assert "_some_polygon_work" in session.format_stats()


def test_editor_scene_emits_polygon_change_profile(monkeypatch, capsys) -> None:
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QApplication, QGraphicsView

    from contour.graphics.editor_scene import PolygonEditorScene

    app = QApplication.instance() or QApplication([])
    view = QGraphicsView()
    scene = PolygonEditorScene(view)
    view.setScene(scene)

    outer_points = [(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)]
    area, perimeter, bbox = compute_polygon_metrics(outer_points)
    outer = PolygonData(
        id=1,
        points=outer_points,
        area=area,
        perimeter=perimeter,
        bbox=bbox,
    )
    scene.set_image_pixmap(QPixmap(100, 100))
    scene.set_polygons([outer], emit_signal=False)

    monkeypatch.setattr(profiling, "POLYGON_CHANGE_PROFILING_ENABLED", True)

    changed = scene._subtract_shape_from_scene(
        points=[(30.0, 30.0), (50.0, 30.0), (50.0, 50.0), (30.0, 50.0)],
        thickness=None,
        label="Erase rectangle",
    )

    captured = capsys.readouterr().out
    assert changed
    assert "[contour polygon change profiling]" in captured
    assert "operation=erase_rectangle" in captured
    assert "[contour polygon change profiling stats]" in captured


def test_move_vertex_tool_activation_emits_console_profile(monkeypatch, capsys) -> None:
    from PyQt6.QtWidgets import QApplication

    from contour.graphics.editor_view import PolygonEditorView
    from contour.graphics.tools import EditorTool
    from contour.infrastructure import profiling

    app = QApplication.instance() or QApplication([])
    view = PolygonEditorView()
    view.set_image(np.zeros((100, 100), dtype=np.uint8))
    view.set_polygons(
        [
            PolygonData(
                id=1,
                points=[(10.0, 10.0), (70.0, 10.0), (70.0, 70.0), (10.0, 70.0)],
                area=3600.0,
                perimeter=240.0,
                bbox=(10, 10, 60, 60),
            )
        ]
    )

    monkeypatch.setattr(profiling, "MOVE_VERTEX_TOOL_PROFILING_ENABLED", True)

    view.set_tool(EditorTool.MOVE_VERTEX)
    app.processEvents()

    captured = capsys.readouterr().out
    assert "[contour move-vertex-tool profiling] started" in captured
    assert "[contour move-vertex-tool profiling] status=displayed" in captured
    assert "polygons=1" in captured
    assert "vertices=4" in captured
    assert "sync_vertices=" in captured
    assert "[contour move-vertex-tool profiling stats]" in captured
