"""test_widget_interactions: test paste refreshes emits."""
from __future__ import annotations
from unittest.mock import patch
from PyQt6.QtCore import QPointF
from contour.application.processing import (
    DisplaySettings,
)
from contour.domain import PolygonData
from contour.graphics_items import EditablePolygonItem
from contour.graphics_view import (
    PolygonEditorScene,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class PasteRefreshesEmitsMixin:
    def test_fully_filtered_contact_paste_does_not_create_undo_entry(self) -> None:
        scene = PolygonEditorScene()
        existing = _rectangle_polygon(0, 0, 10, 10)
        existing.category = "via"
        existing.shape_hint = "box"
        duplicate = existing.clone()
        scene.set_polygons([existing])
        scene.set_minimum_contact_distance(20.0)
        undo_count = scene.undo_stack.count()
        changed: list[None] = []
        scene.polygonsChanged.connect(lambda: changed.append(None))

        pasted = scene.add_cloned_polygons_at(
            [duplicate],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(pasted, [])
        self.assertEqual(scene.undo_stack.count(), undo_count)
        self.assertEqual(changed, [])
        self.assertEqual(len(scene.get_polygons()), 1)

    def test_paste_refreshes_and_emits_once_per_undo_operation(self) -> None:
        scene = PolygonEditorScene()
        contacts = []
        for index in range(10):
            contact = _rectangle_polygon(index * 20, 10, index * 20 + 10, 20)
            contact.id = index + 1
            contact.category = "via"
            contact.shape_hint = "box"
            contacts.append(contact)
        changed: list[None] = []
        scene.polygonsChanged.connect(lambda: changed.append(None))

        with patch.object(scene, "_refresh_all_items", wraps=scene._refresh_all_items) as refresh:
            pasted = scene.add_cloned_polygons_at(
                contacts,
                QPointF(95.0, 15.0),
                QPointF(95.0, 45.0),
            )
            self.assertEqual(len(pasted), len(contacts))
            # New items are refreshed directly; unchanged scene items are not.
            self.assertEqual(refresh.call_count, 0)
            self.assertEqual(len(changed), 1)

            scene.undo_stack.undo()
            self.assertEqual(refresh.call_count, 0)
            self.assertEqual(len(changed), 2)
            self.assertEqual(scene.get_polygons(), [])

            scene.undo_stack.redo()
            self.assertEqual(refresh.call_count, 0)
            self.assertEqual(len(changed), 3)
            self.assertEqual(len(scene.get_polygons()), len(contacts))

    def test_contact_recognition_mode_protects_only_recognized_vias(self) -> None:
        scene = PolygonEditorScene()
        manual = _rectangle_polygon(10, 10, 20, 20)
        manual.category = "via"
        manual.shape_hint = "box"
        automatic = _rectangle_polygon(30, 30, 40, 40)
        automatic.id = 2
        automatic.category = "via"
        automatic.shape_hint = "box"
        automatic.recognition_score = 82.0
        scene.set_polygons([manual, automatic])
        scene.set_protect_recognized_vias(True)

        self.assertFalse(scene.delete_via_at(QPointF(35.0, 35.0)))
        self.assertTrue(scene.delete_via_at(QPointF(15.0, 15.0)))
        self.assertEqual([polygon.id for polygon in scene.get_polygons()], [2])

    def test_manual_via_uses_configured_color_and_recognized_via_keeps_score_color(self) -> None:
        scene = PolygonEditorScene()
        scene.set_display_settings(
            DisplaySettings(
                external_color="#7C3AED",
                via_selection_color="#FACC15",
            )
        )
        manual = _rectangle_polygon(10, 10, 20, 20)
        manual.category = "via"
        manual.shape_hint = "box"
        automatic = _rectangle_polygon(30, 30, 40, 40)
        automatic.id = 2
        automatic.category = "via"
        automatic.shape_hint = "box"
        automatic.recognition_score = 100.0
        scene.set_polygons([manual, automatic])

        self.assertEqual(scene._polygon_items[1].pen().color().name().lower(), "#7c3aed")
        self.assertNotEqual(scene._polygon_items[2].pen().color().name().lower(), "#7c3aed")
        scene.select_polygon(1)
        self.assertEqual(scene._polygon_items[1].pen().color().name().lower(), "#facc15")

    def test_via_display_mode_switches_between_circle_and_rectangle_paths(self) -> None:
        polygon = PolygonData(
            id=1,
            points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            category="via",
            shape_hint="box",
        )

        circle_item = EditablePolygonItem(polygon, DisplaySettings(via_display_mode="circle"))
        rectangle_item = EditablePolygonItem(polygon, DisplaySettings(via_display_mode="rectangle"))

        self.assertFalse(circle_item.path().contains(QPointF(1.0, 1.0)))
        self.assertTrue(rectangle_item.path().contains(QPointF(1.0, 1.0)))

    def test_via_cursor_turns_red_when_placement_overlaps_existing_via(self) -> None:
        scene = PolygonEditorScene()
        scene.add_via_at(QPointF(20.0, 20.0), 10.0, 10.0)

        scene.set_via_cursor(QPointF(24.0, 20.0), 10.0, 10.0, True)
        blocked_color = scene._via_cursor_item.pen().color().name().lower()
        scene.set_via_cursor(QPointF(40.0, 20.0), 10.0, 10.0, True)
        allowed_color = scene._via_cursor_item.pen().color().name().lower()

        self.assertEqual(blocked_color, "#ef4444")
        self.assertEqual(allowed_color, "#a78bfa")
