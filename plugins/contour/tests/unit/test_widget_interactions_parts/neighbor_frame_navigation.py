"""test_widget_interactions: test neighbor frame navigation."""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
)
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.domain import PolygonData
from contour.graphics_view import (
    EditorTool,
    PolygonEditorView,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import (
    _app,
    _rectangle_polygon,
)


class NeighborFrameNavigationMixin:
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.view = PolygonEditorView()
        self.view._editor_scene.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=0.0, min_hole_area_to_remove_px2=0.0)
        )
        self.view.resize(320, 320)
        self.view.set_image(np.zeros((100, 100), dtype=np.uint8))
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view.set_tool(EditorTool.SELECT)
        self.view.show()
        self._app.processEvents()

    def test_neighbor_frame_vectors_are_rendered_over_neighbor_image(self) -> None:
        neighbor = QImage(25, 25, QImage.Format.Format_RGB32)
        neighbor.fill(0)
        polygon = _rectangle_polygon(10, 10, 40, 40)

        self.view.set_neighbor_frames([(1, 0, neighbor, "neighbor.jpg", [polygon], (100, 100))], 0.5)

        scene = self.view._editor_scene
        vector_items = [item for item in scene._neighbor_frame_items if isinstance(item, QGraphicsPathItem)]
        self.assertEqual(len(vector_items), 1)
        self.assertEqual(scene.neighbor_frame_path_at(QPointF(20, 20)), None)
        self.assertEqual(scene.neighbor_frame_path_at(QPointF(120, 20)), "neighbor.jpg")

    def test_select_click_computes_additive_mode_before_picking_polygon(self) -> None:
        scene = self.view._editor_scene
        with patch.object(scene, "polygon_at", return_value=1) as polygon_at:
            self._press_editor(Qt.MouseButton.LeftButton, QPointF(50.0, 50.0))

        polygon_at.assert_called_once()
        self.assertTrue(polygon_at.call_args.kwargs["cycle"])
        self.assertEqual(scene.selected_polygon_id(), 1)

    def test_conductor_recognition_mode_blocks_vector_edit_tools(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        self.assertEqual(self.view.current_tool, EditorTool.BRUSH)

        self.view.set_conductor_recognition_mode(True)
        self.assertTrue(self.view.vector_edits_locked())
        self.assertEqual(self.view.current_tool, EditorTool.SELECT)
        self.assertNotIn(EditorTool.BRUSH, self.view.available_tools())
        self.assertNotIn(EditorTool.ADD_POLYGON, self.view.available_tools())
        self.view.set_tool(EditorTool.MOVE_VERTEX)
        self.assertEqual(self.view.current_tool, EditorTool.SELECT)

        self.view.set_conductor_recognition_mode(False)
        self.assertFalse(self.view.vector_edits_locked())
        self.assertIn(EditorTool.BRUSH, self.view.available_tools())

    def test_neighbor_frame_path_at_overlap_prefers_neighbor_without_main_vectors(self) -> None:
        neighbor = QImage(100, 100, QImage.Format.Format_RGB32)
        neighbor.fill(0)

        self.view.set_polygons([])
        self.view.set_neighbor_frames([(1, 0, neighbor, "neighbor.jpg", [], (100, 100))], 0.5, overlap_pixels=20)

        scene = self.view._editor_scene
        self.assertEqual(scene.neighbor_frame_path_at(QPointF(90, 50)), "neighbor.jpg")
        self.assertIsNone(scene.neighbor_frame_path_at(QPointF(40, 50)))

    def test_neighbor_frame_navigation_requires_double_click(self) -> None:
        neighbor = QImage(25, 25, QImage.Format.Format_RGB32)
        neighbor.fill(0)
        self.view.set_neighbor_frames(
            [(1, 0, neighbor, "neighbor.jpg", [], (100, 100))],
            0.5,
        )
        activated: list[str] = []
        self.view.neighborFrameActivated.connect(activated.append)
        position = self.view.mapFromScene(QPointF(120.0, 20.0))

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            position,
        )
        self._app.processEvents()
        self.assertEqual(activated, [])

        QTest.mouseDClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            position,
        )
        self._app.processEvents()
        self.assertEqual(activated, ["neighbor.jpg"])

    def test_contact_tool_click_on_existing_via_requests_diagnostics(self) -> None:
        via = _rectangle_polygon(40, 40, 60, 60)
        via.category = "via"
        via.shape_hint = "box"
        requested: list[PolygonData] = []
        self.view.set_polygons([via])
        self.view.set_contact_recognition_mode(True)
        self.view.set_via_debug_inspection_enabled(True)
        self.view.set_tool(EditorTool.ADD_VIA)
        self.view.viaDebugRequested.connect(requested.append)

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(50.0, 50.0)),
        )
        self._app.processEvents()

        self.assertEqual([polygon.id for polygon in requested], [1])
        self.assertEqual(len(self.view.get_polygons()), 1)

    def test_copy_and_paste_emit_profile_boundaries_and_counts(self) -> None:
        self.view._editor_scene.select_polygon(1)
        copy_started: list[None] = []
        copy_finished: list[int] = []
        paste_started: list[int] = []
        paste_finished: list[int] = []
        self.view.contactCopyStarted.connect(lambda: copy_started.append(None))
        self.view.contactCopyFinished.connect(copy_finished.append)
        self.view.contactPasteStarted.connect(paste_started.append)
        self.view.contactPasteFinished.connect(paste_finished.append)

        self.view.copy_selected()
        self.view.start_paste_mode()
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(15.0, 15.0)),
        )
        self._app.processEvents()

        self.assertEqual(len(copy_started), 1)
        self.assertEqual(copy_finished, [1])
        self.assertEqual(paste_started, [1])
        self.assertEqual(paste_finished, [1])
        self.assertEqual(len(self.view.get_polygons()), 1)

    def test_paste_preview_reuses_items_while_pointer_moves(self) -> None:
        self.view._editor_scene.select_polygon(1)
        self.view.copy_selected()
        self.view.start_paste_mode()
        initial_items = list(self.view._paste_preview_items)

        self.view._update_paste_preview(QPointF(30.0, 30.0))
        self.view._update_paste_preview(QPointF(60.0, 70.0))

        self.assertEqual(len(initial_items), 1)
        self.assertEqual(self.view._paste_preview_items, initial_items)
        self.assertEqual(
            initial_items[0].pos(),
            QPointF(
                60.0 - self.view._clipboard_anchor.x(),
                70.0 - self.view._clipboard_anchor.y(),
            ),
        )
