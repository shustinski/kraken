"""Monochrome line icons drawn from the bundled SVG set.

Every file in ``icons/`` is a 24x24 outline drawn with ``stroke="currentColor"``.
An icon is rendered on demand in the colour that matches its mode and state:

* ``Normal`` / ``Active`` — the theme text colour;
* ``Disabled`` — the text colour blended halfway into the window colour;
* ``On`` (a checked tool button) — the theme accent (``palette().highlight()``).

The colours are read once, when the icon is built. Rebuild icons after a theme
change (see :func:`contour.ui.editor_icons.refresh_editor_icons`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from importlib import resources

from PyQt6.QtCore import QByteArray, QPoint, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication, QLineEdit, QWidget

ICON_GRID_PX = 24

ICON_NAMES = (
    "add_polygon",
    "add_vertex",
    "add_via",
    "antialias",
    "antialias_all",
    "brush",
    "close",
    "delete_polygon",
    "delete_vertex",
    "files",
    "fit",
    "folder",
    "images_source",
    "move_vertex",
    "pan",
    "play",
    "play_all",
    "redo",
    "ruler",
    "save",
    "select",
    "select_area",
    "stop",
    "trace_pen",
    "undo",
    "vectors_source",
    "zoom_in",
    "zoom_out",
)


@dataclass(frozen=True)
class IconColors:
    normal: QColor
    disabled: QColor
    on: QColor

    def for_mode(self, mode: QIcon.Mode, state: QIcon.State) -> QColor:
        if mode == QIcon.Mode.Disabled:
            return self.disabled
        if state == QIcon.State.On:
            return self.on
        return self.normal


def _blend(foreground: QColor, background: QColor, amount: float) -> QColor:
    return QColor(
        round(foreground.red() * (1.0 - amount) + background.red() * amount),
        round(foreground.green() * (1.0 - amount) + background.green() * amount),
        round(foreground.blue() * (1.0 - amount) + background.blue() * amount),
    )


def icon_colors() -> IconColors:
    """Return icon colours for the active theme.

    Kraken themes are application stylesheets, so the application palette
    keeps platform colours. Polished probe widgets report what the stylesheet
    really paints: a plain widget gives text and window colours, a line edit
    gives the selection (accent) colour.
    """

    if QApplication.instance() is None:
        palette = QGuiApplication.palette()
        text = palette.color(QPalette.ColorRole.WindowText)
        window = palette.color(QPalette.ColorRole.Window)
        accent = palette.color(QPalette.ColorRole.Highlight)
    else:
        probe = QWidget()
        probe.ensurePolished()
        text = probe.palette().color(QPalette.ColorRole.WindowText)
        window = probe.palette().color(QPalette.ColorRole.Window)
        accent_probe = QLineEdit()
        accent_probe.ensurePolished()
        accent = accent_probe.palette().color(QPalette.ColorRole.Highlight)
    return IconColors(normal=text, disabled=_blend(text, window, 0.55), on=accent)


@cache
def _svg_source(name: str) -> str:
    return resources.files("contour.ui").joinpath("icons", f"{name}.svg").read_text(encoding="utf-8")


@lru_cache(maxsize=512)
def _renderer(name: str, color_name: str) -> QSvgRenderer:
    svg = _svg_source(name).replace("currentColor", color_name)
    return QSvgRenderer(QByteArray(svg.encode("utf-8")))


class _SvgIconEngine(QIconEngine):
    def __init__(self, name: str, colors: IconColors) -> None:
        super().__init__()
        self._name = name
        self._colors = colors

    def paint(self, painter: QPainter | None, rect: QRect, mode: QIcon.Mode, state: QIcon.State) -> None:
        if painter is None:
            return
        side = min(rect.width(), rect.height())
        target = QRectF(
            rect.x() + (rect.width() - side) / 2.0,
            rect.y() + (rect.height() - side) / 2.0,
            side,
            side,
        )
        color = self._colors.for_mode(mode, state)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # QtSvg does not read #AARRGGBB, so pass RGB and the alpha separately.
        painter.setOpacity(painter.opacity() * color.alphaF())
        _renderer(self._name, color.name(QColor.NameFormat.HexRgb)).render(painter, target)
        painter.restore()

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State, scale: float) -> QPixmap:
        pixmap = QPixmap(max(1, round(size.width() * scale)), max(1, round(size.height() * scale)))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(QPoint(0, 0), pixmap.size()), mode, state)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        return pixmap

    def actualSize(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QSize:
        return size

    def clone(self) -> QIconEngine:
        return _SvgIconEngine(self._name, self._colors)

    def key(self) -> str:
        return "contour-svg"

    def isNull(self) -> bool:
        return False


def icon(name: str, colors: IconColors | None = None) -> QIcon:
    """Return the themed icon ``name`` (a file stem in ``icons/``)."""

    if name not in ICON_NAMES:
        raise KeyError(f"unknown icon: {name}")
    return QIcon(_SvgIconEngine(name, colors or icon_colors()))


__all__ = ["ICON_GRID_PX", "ICON_NAMES", "IconColors", "icon", "icon_colors"]
