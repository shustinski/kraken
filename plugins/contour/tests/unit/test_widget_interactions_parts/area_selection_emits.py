"""test_widget_interactions: test area selection emits."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from contour.graphics_view import (
    EditorTool,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import (
    _all_points_within,
    _rectangle_polygon,
)


class AreaSelectionEmitsMixin:
    def test_finished_point_polygon_is_cropped_to_image_bounds(self) -> None:
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.ADD_POLYGON)
        for point in (
            QPointF(-20.0, -20.0),
            QPointF(60.0, -20.0),
            QPointF(60.0, 60.0),
            QPointF(-20.0, 60.0),
        ):
            self.view._editor_scene.append_pending_point(point)

        self.view._editor_scene.finish_pending_polygon()
        self._app.processEvents()

        polygons = self.view.get_polygons()
        self.assertTrue(polygons)
        self.assertTrue(_all_points_within(polygons, 0.0, 0.0, 100.0, 100.0))

    def test_finished_rectangle_polygon_is_cropped_to_image_bounds(self) -> None:
        self.view.set_polygons([])

        added = self.view._editor_scene.add_rectangle_polygon(QPointF(-20.0, -20.0), QPointF(60.0, 60.0))
        self._app.processEvents()

        polygons = self.view.get_polygons()
        self.assertTrue(added)
        self.assertTrue(polygons)
        self.assertTrue(_all_points_within(polygons, 0.0, 0.0, 100.0, 100.0))

    def test_delete_key_still_deletes_selected_polygon(self) -> None:
        self.view._editor_scene.select_polygon(1)

        QTest.keyClick(self.view.viewport(), Qt.Key.Key_Delete)
        self._app.processEvents()

        self.assertEqual(self.view.get_polygons(), [])

    def test_contact_delete_key_emits_separate_profiling_boundaries(self) -> None:
        via = _rectangle_polygon(20, 20, 40, 40)
        via.category = "via"
        via.shape_hint = "box"
        self.view.set_polygons([via])
        self.view._editor_scene.select_polygon(1)
        started: list[int] = []
        finished: list[int] = []
        self.view.contactDeletionStarted.connect(started.append)
        self.view.contactDeletionFinished.connect(finished.append)

        QTest.keyClick(self.view.viewport(), Qt.Key.Key_Delete)
        self._app.processEvents()

        self.assertEqual(started, [1])
        self.assertEqual(finished, [1])
        self.assertEqual(self.view.get_polygons(), [])

    def test_contact_undo_and_redo_emit_profiling_boundaries(self) -> None:
        via = _rectangle_polygon(20, 20, 40, 40)
        via.category = "via"
        via.shape_hint = "box"
        self.view.set_polygons([via])
        self.view._editor_scene.select_polygon(1)
        self.assertTrue(self.view._editor_scene.delete_polygon())
        undo_started: list[bool] = []
        undo_finished: list[tuple[bool, int]] = []
        redo_started: list[bool] = []
        redo_finished: list[tuple[bool, int]] = []
        self.view.contactUndoStarted.connect(lambda: undo_started.append(True))
        self.view.contactUndoFinished.connect(
            lambda applied, count: undo_finished.append((applied, count))
        )
        self.view.contactRedoStarted.connect(lambda: redo_started.append(True))
        self.view.contactRedoFinished.connect(
            lambda applied, count: redo_finished.append((applied, count))
        )

        self.view.undo()
        self.view.redo()

        self.assertEqual(undo_started, [True])
        self.assertEqual(undo_finished, [(True, 1)])
        self.assertEqual(redo_started, [True])
        self.assertEqual(redo_finished, [(True, 1)])

    def test_area_selection_emits_selected_contact_count(self) -> None:
        first = _rectangle_polygon(20, 20, 35, 35)
        first.category = "via"
        first.shape_hint = "box"
        second = _rectangle_polygon(55, 55, 70, 70)
        second.id = 2
        second.category = "via"
        second.shape_hint = "box"
        self.view.set_polygons([first, second])
        started: list[bool] = []
        finished: list[int] = []
        self.view.contactMultiSelectionStarted.connect(lambda: started.append(True))
        self.view.contactMultiSelectionFinished.connect(finished.append)
        press_pos = self.view.mapFromScene(QPointF(5.0, 5.0))
        release_pos = self.view.mapFromScene(QPointF(85.0, 85.0))

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            press_pos,
        )
        QTest.mouseMove(self.view.viewport(), release_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            release_pos,
        )
        self._app.processEvents()

        self.assertEqual(started, [True])
        self.assertEqual(finished, [2])
