from __future__ import annotations

import cProfile

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


def test_editor_scene_emits_polygon_change_profile(monkeypatch) -> None:
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

    messages: list[str] = []
    scene.logRequested.connect(messages.append)
    monkeypatch.setattr(profiling, "POLYGON_CHANGE_PROFILING_ENABLED", True)

    changed = scene._subtract_shape_from_scene(
        points=[(30.0, 30.0), (50.0, 30.0), (50.0, 50.0), (30.0, 50.0)],
        thickness=None,
        label="Erase rectangle",
    )

    assert changed
    assert any("[contour polygon change profiling]" in message for message in messages)
    assert any("operation=erase_rectangle" in message for message in messages)
    assert any("[contour polygon change profiling stats]" in message for message in messages)
