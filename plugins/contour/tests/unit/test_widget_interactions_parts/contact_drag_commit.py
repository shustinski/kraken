"""test_widget_interactions: test contact drag commit."""
from __future__ import annotations
from unittest.mock import patch
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class ContactDragCommitMixin:
    def test_contact_drag_commit_updates_only_contact_and_is_undoable(self) -> None:
        contact = _rectangle_polygon(20, 20, 40, 40)
        contact.category = "via"
        contact.shape_hint = "box"
        contact.recognition_score = 87.0
        self.view.set_polygons([contact])
        scene = self.view._editor_scene
        scene.select_polygon(1)
        original_points = scene.polygon_points(1)
        moved_points = [(x + 15, y + 10) for x, y in original_points]
        scene.preview_polygon_move(1, moved_points)
        self.view._drag_kind = "polygon"
        self.view._drag_polygon_id = 1
        self.view._drag_origin_points = original_points
        self.view._drag_start_scene_pos = QPointF(20.0, 20.0)
        self.view._drag_polygons_snapshot = None
        self.view._drag_polygon_is_contact = True

        with patch.object(
            scene,
            "_bulk_restore_polygons",
            wraps=scene._bulk_restore_polygons,
        ) as bulk_restore:
            QTest.mouseRelease(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                self.view.mapFromScene(QPointF(55.0, 50.0)),
            )
            self._app.processEvents()

        bulk_restore.assert_not_called()
        self.assertEqual(scene.polygon_points(1), moved_points)
        self.assertEqual(scene.polygon_snapshot(1).recognition_score, 87.0)

        scene.undo_stack.undo()
        self.assertEqual(scene.polygon_points(1), original_points)
        scene.undo_stack.redo()
        self.assertEqual(scene.polygon_points(1), moved_points)

    def test_contact_drag_skips_hover_hit_testing_on_mouse_move(self) -> None:
        contact = _rectangle_polygon(20, 20, 40, 40)
        contact.category = "via"
        contact.shape_hint = "box"
        self.view.set_polygons([contact])
        self.view._drag_kind = "polygon"
        self.view._drag_polygon_id = None
        self.view._drag_polygon_is_contact = True

        with patch.object(
            self.view._editor_scene,
            "sync_conductor_hover_highlight",
            wraps=self.view._editor_scene.sync_conductor_hover_highlight,
        ) as hover_hit_test:
            QTest.mouseMove(
                self.view.viewport(),
                self.view.mapFromScene(QPointF(30.0, 30.0)),
            )
            self._app.processEvents()

        hover_hit_test.assert_not_called()
