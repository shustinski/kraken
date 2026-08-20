from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from contour.graphics.minimap_geometry import MINIMAP_VIEWPORT_MARGIN_PX, minimap_point_to_scene
from contour.graphics.tools import EditorTool
from contour.graphics_view import PolygonEditorView


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class PolygonEditorMinimapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.view = PolygonEditorView()
        self.view.resize(420, 420)
        self.view.show()
        self._app.processEvents()

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        self._app.processEvents()

    def test_minimap_hidden_without_image(self) -> None:
        self.assertFalse(self.view._minimap.isVisible())
        self.assertFalse(self.view._minimap.has_image())

    def test_minimap_appears_bottom_right_after_set_image(self) -> None:
        self.view.set_image(np.zeros((80, 120), dtype=np.uint8))
        self._app.processEvents()
        minimap = self.view._minimap
        self.assertTrue(minimap.isVisible())
        self.assertTrue(minimap.has_image())
        viewport = self.view.viewport()
        self.assertIsNotNone(viewport)
        assert viewport is not None
        viewport_geom = viewport.geometry()
        minimap_geom = minimap.geometry()
        self.assertGreater(minimap_geom.left(), viewport_geom.center().x())
        self.assertGreater(minimap_geom.top(), viewport_geom.center().y())
        self.assertLessEqual(minimap_geom.right(), viewport_geom.right())
        self.assertLessEqual(minimap_geom.bottom(), viewport_geom.bottom())
        self.assertAlmostEqual(
            viewport_geom.right() - minimap_geom.right(),
            MINIMAP_VIEWPORT_MARGIN_PX,
            delta=4,
        )
        self.assertAlmostEqual(
            viewport_geom.bottom() - minimap_geom.bottom(),
            MINIMAP_VIEWPORT_MARGIN_PX,
            delta=4,
        )

    def test_minimap_click_centers_viewport_on_image_point(self) -> None:
        self.view.set_image(np.zeros((100, 100), dtype=np.uint8))
        self.view.fit_to_view()
        self.view.scale(4.0, 4.0)
        self.view._update_navigation_scene_rect()
        self._app.processEvents()
        minimap = self.view._minimap
        local = QPoint(minimap.width() // 4, minimap.height() // 4)
        expected_x, expected_y = minimap_point_to_scene(
            (float(local.x()), float(local.y())),
            (0.0, 0.0, 100.0, 100.0),
            (0.0, 0.0, float(minimap.width()), float(minimap.height())),
        )
        QTest.mouseClick(minimap, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, local)
        self._app.processEvents()
        viewport = self.view.viewport()
        self.assertIsNotNone(viewport)
        assert viewport is not None
        center = self.view.mapToScene(viewport.rect().center())
        self.assertAlmostEqual(center.x(), expected_x, delta=8.0)
        self.assertAlmostEqual(center.y(), expected_y, delta=8.0)

    def test_minimap_click_does_not_reach_scene_tools(self) -> None:
        self.view.set_image(np.zeros((64, 64), dtype=np.uint8))
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self._app.processEvents()
        minimap = self.view._minimap
        QTest.mouseClick(
            minimap,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(minimap.width() // 2, minimap.height() // 2),
        )
        self._app.processEvents()
        self.assertFalse(self.view._editor_scene.has_pending_polygon())

    def test_minimap_works_in_conductor_recognition_mode(self) -> None:
        self.view.set_image(np.zeros((80, 80), dtype=np.uint8))
        self.view.set_conductor_recognition_mode(True)
        self.view.scale(3.0, 3.0)
        self.view._update_navigation_scene_rect()
        self._app.processEvents()
        minimap = self.view._minimap
        self.assertTrue(minimap.isVisible())
        local = QPoint(8, 8)
        QTest.mouseClick(minimap, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, local)
        self._app.processEvents()
        viewport = self.view.viewport()
        self.assertIsNotNone(viewport)
        assert viewport is not None
        center = self.view.mapToScene(viewport.rect().center())
        self.assertLess(center.x(), 30.0)
        self.assertLess(center.y(), 30.0)

    def test_minimap_hidden_after_clearing_image(self) -> None:
        self.view.set_image(np.zeros((24, 24), dtype=np.uint8))
        self._app.processEvents()
        self.assertTrue(self.view._minimap.isVisible())
        self.view.set_image(None)
        self._app.processEvents()
        self.assertFalse(self.view._minimap.isVisible())
        self.assertFalse(self.view._minimap.has_image())
