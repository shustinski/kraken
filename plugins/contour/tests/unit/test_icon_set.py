from __future__ import annotations

import unittest
from importlib import resources

from kraken_core.theme import apply_app_theme
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QColor, QIcon, QImage
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication, QToolButton

from contour.graphics.tools import EditorTool
from contour.ui import icon_set
from contour.ui.editor_icons import create_editor_action_icon, create_editor_tool_icon, refresh_editor_icons

_ICON_PX = 24


def _app() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


def _stroke_color(icon: QIcon, mode: QIcon.Mode = QIcon.Mode.Normal, state: QIcon.State = QIcon.State.Off) -> QColor:
    """Return the colour of the most opaque pixel, i.e. a stroke centre."""

    image = icon.pixmap(QSize(_ICON_PX, _ICON_PX), mode, state).toImage().convertToFormat(QImage.Format.Format_ARGB32)
    best = QColor(0, 0, 0, 0)
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() > best.alpha():
                best = color
    return best


class IconSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def tearDown(self) -> None:
        self._app.setStyleSheet("")
        self._app.setProperty("krakenTheme", None)

    def test_every_editor_tool_has_a_visible_icon(self) -> None:
        for tool in EditorTool:
            with self.subTest(tool=tool):
                icon = create_editor_tool_icon(tool)
                self.assertFalse(icon.isNull())
                self.assertGreater(_stroke_color(icon).alpha(), 200)

    def test_editor_actions_have_visible_icons(self) -> None:
        for action in ("undo", "redo", "zoom_in", "zoom_out", "fit", "antialias", "antialias_all"):
            with self.subTest(action=action):
                self.assertGreater(_stroke_color(create_editor_action_icon(action)).alpha(), 200)

    def test_icon_files_match_the_declared_names_and_follow_the_grid(self) -> None:
        folder = resources.files("contour.ui").joinpath("icons")
        stems = sorted(entry.name.removesuffix(".svg") for entry in folder.iterdir() if entry.name.endswith(".svg"))
        self.assertEqual(stems, sorted(icon_set.ICON_NAMES))
        for name in icon_set.ICON_NAMES:
            with self.subTest(icon=name):
                source = folder.joinpath(f"{name}.svg").read_text(encoding="utf-8")
                self.assertIn('viewBox="0 0 24 24"', source)
                self.assertIn('stroke="currentColor"', source)
                self.assertIn('stroke-width="1.75"', source)
                self.assertTrue(QSvgRenderer(source.replace("currentColor", "#000000").encode()).isValid())

    def test_modes_use_text_muted_and_accent_colours(self) -> None:
        colors = icon_set.IconColors(normal=QColor("#101010"), disabled=QColor("#808080"), on=QColor("#2060f0"))
        icon = icon_set.icon("select", colors)

        self.assertEqual(_stroke_color(icon).name(), "#101010")
        self.assertEqual(_stroke_color(icon, QIcon.Mode.Disabled).name(), "#808080")
        self.assertEqual(_stroke_color(icon, QIcon.Mode.Normal, QIcon.State.On).name(), "#2060f0")

    def test_theme_change_recolours_icons(self) -> None:
        apply_app_theme("dark", self._app)
        dark = _stroke_color(create_editor_tool_icon(EditorTool.SELECT))
        apply_app_theme("light", self._app)
        light = _stroke_color(create_editor_tool_icon(EditorTool.SELECT))

        self.assertNotEqual(dark.name(), light.name())
        self.assertGreater(dark.lightness(), light.lightness())

    def test_refresh_editor_icons_rebuilds_button_icons(self) -> None:
        class _Host:
            pass

        host = _Host()
        tool_button = QToolButton()
        tool_button.setCheckable(True)
        host._tool_buttons = {EditorTool.BRUSH: tool_button}
        host.undo_button = QToolButton()

        apply_app_theme("dark", self._app)
        refresh_editor_icons(host)
        dark = _stroke_color(tool_button.icon())
        dark_undo = _stroke_color(host.undo_button.icon())
        apply_app_theme("light", self._app)
        refresh_editor_icons(host)

        self.assertNotEqual(dark.name(), _stroke_color(tool_button.icon()).name())
        self.assertNotEqual(dark_undo.name(), _stroke_color(host.undo_button.icon()).name())
        self.assertIn("QToolButton:checked", tool_button.styleSheet())
        self.assertNotIn("QToolButton:checked", host.undo_button.styleSheet())


if __name__ == "__main__":
    unittest.main()
