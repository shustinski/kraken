"""test_widget_interactions: test contact drag preview."""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
from PyQt6.QtCore import QPointF
from contour.graphics_view import (
    PolygonEditorScene,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class ContactDragPreviewMixin:
    def test_contact_drag_preview_refreshes_only_moved_contact(self) -> None:
        scene = PolygonEditorScene()
        contacts = []
        for index in range(100):
            contact = _rectangle_polygon(index * 15, 0, index * 15 + 8, 8)
            contact.id = index + 1
            contact.category = "via"
            contact.shape_hint = "box"
            contacts.append(contact)
        scene.set_polygons(contacts)
        moved_points = [(x, y + 20) for x, y in scene.polygon_points(1)]

        with (
            patch.object(
                scene,
                "_refresh_all_items",
                wraps=scene._refresh_all_items,
            ) as refresh_all,
            patch.object(
                scene,
                "_refresh_polygon_item",
                wraps=scene._refresh_polygon_item,
            ) as refresh_item,
        ):
            scene.preview_polygon_move(1, moved_points)

        self.assertEqual(refresh_all.call_count, 0)
        self.assertEqual(refresh_item.call_count, 0)
        self.assertNotEqual(scene.polygon_points(1), moved_points)
        self.assertEqual(scene.polygon_edit_preview_points(1), moved_points)
        self.assertTrue(scene._polygon_edit_preview_item.isVisible())

    def test_selection_refreshes_only_previous_and_next_contacts(self) -> None:
        scene = PolygonEditorScene()
        contacts = []
        for index in range(100):
            contact = _rectangle_polygon(index * 15, 0, index * 15 + 8, 8)
            contact.id = index + 1
            contact.category = "via"
            contact.shape_hint = "box"
            contacts.append(contact)
        scene.set_polygons(contacts)
        scene.select_polygon(20)

        with patch.object(
            scene,
            "_refresh_polygon_selection_item",
            wraps=scene._refresh_polygon_selection_item,
        ) as refresh_item:
            scene.select_polygon(80)

        self.assertEqual(refresh_item.call_count, 2)

    def test_multi_selection_does_not_resolve_edit_families_for_contacts(self) -> None:
        scene = PolygonEditorScene()
        contacts = []
        for index in range(100):
            contact = _rectangle_polygon(index * 15, 0, index * 15 + 8, 8)
            contact.id = index + 1
            contact.category = "via"
            contact.shape_hint = "box"
            contacts.append(contact)
        scene.set_polygons(contacts)

        with patch.object(
            scene,
            "_polygon_edit_family_ids",
            wraps=scene._polygon_edit_family_ids,
        ) as edit_family:
            scene.select_polygons([contact.id for contact in contacts])

        edit_family.assert_not_called()
        self.assertEqual(len(scene.selected_polygons()), len(contacts))

    def test_delete_via_at_deletes_only_vias(self) -> None:
        scene = PolygonEditorScene()
        conductor = _rectangle_polygon(0, 0, 20, 20)
        via = _rectangle_polygon(35, 35, 45, 45)
        via.id = 2
        via.category = "via"
        via.shape_hint = "box"
        scene.set_polygons([conductor, via])

        self.assertFalse(scene.delete_via_at(QPointF(10.0, 10.0)))
        self.assertEqual(len(scene.get_polygons()), 2)
        self.assertTrue(scene.delete_via_at(QPointF(40.0, 40.0)))
        self.assertEqual(len(scene.get_polygons()), 1)

    def test_via_deletion_and_undo_skip_full_scene_refresh(self) -> None:
        scene = PolygonEditorScene()
        contacts = []
        for index in range(100):
            contact = _rectangle_polygon(index * 15, 0, index * 15 + 8, 8)
            contact.id = index + 1
            contact.category = "via"
            contact.shape_hint = "box"
            contacts.append(contact)
        scene.set_polygons(contacts)

        with patch.object(
            scene,
            "_refresh_all_items",
            wraps=scene._refresh_all_items,
        ) as refresh:
            self.assertTrue(scene.delete_via_at(QPointF(4.0, 4.0)))
            self.assertEqual(refresh.call_count, 0)
            self.assertEqual(len(scene.get_polygons()), 99)

            scene.undo_stack.undo()
            self.assertEqual(refresh.call_count, 0)
            self.assertEqual(len(scene.get_polygons()), 100)

    def test_polygon_and_via_creation_are_mutually_exclusive(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))

        self.assertTrue(scene.add_via_at(QPointF(20.0, 20.0), 10.0, 10.0))
        self.assertFalse(scene.add_rectangle_polygon(QPointF(40.0, 40.0), QPointF(60.0, 60.0)))

        scene.set_polygons([_rectangle_polygon(10, 10, 30, 30)])
        self.assertFalse(scene.add_via_at(QPointF(60.0, 60.0), 10.0, 10.0))

    def test_paste_cannot_create_mixed_content(self) -> None:
        scene = PolygonEditorScene()
        via = _rectangle_polygon(10, 10, 20, 20)
        via.category = "via"
        via.shape_hint = "box"
        scene.set_polygons([via])

        pasted = scene.add_cloned_polygons_at(
            [_rectangle_polygon(30, 30, 50, 50)],
            QPointF(40.0, 40.0),
            QPointF(70.0, 70.0),
        )

        self.assertEqual(pasted, [])
        self.assertEqual(len(scene.get_polygons()), 1)

    def test_paste_merges_intersecting_conductors_and_is_undoable(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))
        existing = _rectangle_polygon(10, 10, 40, 40)
        scene.set_polygons([existing])

        pasted = scene.add_cloned_polygons_at(
            [_rectangle_polygon(30, 10, 60, 40)],
            QPointF(0.0, 0.0),
            QPointF(0.0, 0.0),
        )

        self.assertEqual(len(pasted), 1)
        polygons = scene.get_polygons()
        self.assertEqual(len(polygons), 1)
        self.assertEqual(min(x for x, _y in polygons[0].points), 10)
        self.assertEqual(max(x for x, _y in polygons[0].points), 60)
        self.assertAlmostEqual(polygons[0].area, 1500.0)

        scene.undo_stack.undo()
        restored = scene.get_polygons()
        self.assertEqual(len(restored), 1)
        self.assertEqual(max(x for x, _y in restored[0].points), 40)

        scene.undo_stack.redo()
        self.assertEqual(max(x for x, _y in scene.get_polygons()[0].points), 60)

    def test_nonintersecting_paste_preserves_points_without_shapely_rebuild(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((100, 100), dtype=np.uint8))
        source = _rectangle_polygon(10, 10, 30, 30)

        with patch(
            "contour.graphics.editor_scene.shapely_to_polygon_data_list"
        ) as rebuild:
            pasted = scene.add_cloned_polygons_at(
                [source],
                QPointF(0.0, 0.0),
                QPointF(40.0, 0.0),
            )

        self.assertEqual(len(pasted), 1)
        self.assertEqual(scene.get_polygons()[0].points, [(50, 10), (70, 10), (70, 30), (50, 30)])
        rebuild.assert_not_called()
