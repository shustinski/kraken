"""test_widget_interactions: test zoom uses visible."""
from __future__ import annotations
from unittest.mock import patch
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsView,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class ZoomUsesVisibleMixin:
    def test_zoom_uses_visible_composite_contact_layer_and_restores_items(self) -> None:
        scene = self.view._editor_scene
        contact_a = _rectangle_polygon(20, 20, 28, 28)
        contact_a.category = "via"
        contact_a.shape_hint = "box"
        contact_b = _rectangle_polygon(400, 400, 408, 408)
        contact_b.id = 2
        contact_b.category = "via"
        contact_b.shape_hint = "box"
        self.view.set_polygons([contact_a, contact_b])

        enabled = scene.begin_zoom_vector_render_mode(minimum_contacts=2)

        self.assertTrue(enabled)
        self.assertTrue(scene.polygon_overlays_visible())
        self.assertTrue(scene._zoom_contact_composite_items)
        self.assertEqual(len(scene._zoom_contact_composite_items), 1)
        self.assertIsInstance(
            scene._zoom_contact_composite_items[0],
            QGraphicsPixmapItem,
        )
        self.assertFalse(
            scene._zoom_contact_composite_items[0].pixmap().isNull()
        )
        self.assertTrue(all(item.isVisible() for item in scene._zoom_contact_composite_items))
        self.assertFalse(scene._polygon_items[1].isVisible())
        self.assertFalse(scene._polygon_items[2].isVisible())

        scene.set_polygon_overlays_visible(False)
        self.assertFalse(scene._zoom_contact_composite_items[0].isVisible())
        scene.set_polygon_overlays_visible(True)
        self.assertTrue(scene._zoom_contact_composite_items[0].isVisible())
        self.assertFalse(scene._polygon_items[1].isVisible())
        self.assertFalse(scene._polygon_items[2].isVisible())

        scene.end_zoom_vector_render_mode()

        self.assertFalse(scene._zoom_contact_composite_items)
        self.assertTrue(scene._polygon_items[1].isVisible())
        self.assertTrue(scene._polygon_items[2].isVisible())
        self.assertTrue(scene.polygon_overlays_visible())

    def test_zoom_requests_one_managed_viewport_update_per_frame(self) -> None:
        viewport = self.view.viewport()
        self.assertEqual(
            self.view._wheel_zoom_timer.timerType(),
            Qt.TimerType.PreciseTimer,
        )
        self.assertEqual(
            self.view._zoom_animation_timer.timerType(),
            Qt.TimerType.PreciseTimer,
        )
        with patch.object(viewport, "update") as viewport_update:
            self.view._enter_zoom_render_mode()
            self.assertEqual(
                self.view.viewportUpdateMode(),
                QGraphicsView.ViewportUpdateMode.NoViewportUpdate,
            )

            self.view._zoom_animation_viewport_pixel = QPoint(30, 30)
            self.view._zoom_animation_target_zoom = self.view.zoom_factor() * 2.0
            self.view._advance_zoom_animation()

            viewport_update.assert_called_once_with()
            self.view._leave_zoom_render_mode()

        self.assertEqual(
            self.view.viewportUpdateMode(),
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate,
        )

    def test_zoom_skips_hover_hit_testing_on_mouse_move(self) -> None:
        self.view._zoom_animation_timer.start(60_000)
        try:
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
        finally:
            self.view._zoom_animation_timer.stop()
