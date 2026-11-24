"""Reusable Qt stylesheets for the polygon extraction widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtGui import QColor

    from .icon_set import IconColors

COMPACT_UI_STYLE = """
#polygonExtractionWidget {
    font-size: 12px;
}
#polygonExtractionWidget QLabel,
#polygonExtractionWidget QCheckBox,
#polygonExtractionWidget QGroupBox {
    font-size: 12px;
}
#polygonExtractionWidget QGroupBox {
    margin-top: 2px;
    padding-top: 2px;
}
#polygonExtractionWidget QPushButton {
    min-height: 24px;
    padding: 2px 6px;
    font-size: 12px;
}
#polygonExtractionWidget QToolButton {
    padding: 0;
}
#polygonExtractionWidget QToolButton:checked {
    background-color: rgba(128, 128, 128, 46);
    border: 1px solid palette(highlight);
    border-radius: 6px;
}
#polygonExtractionWidget QToolButton:checked:hover {
    background-color: rgba(128, 128, 128, 64);
}
#polygonExtractionWidget QLineEdit,
#polygonExtractionWidget QComboBox,
#polygonExtractionWidget QSpinBox,
#polygonExtractionWidget QDoubleSpinBox {
    min-height: 22px;
    padding: 1px 3px;
    font-size: 12px;
}
#polygonExtractionWidget QTabBar::tab {
    min-height: 20px;
    padding: 2px 6px;
    font-size: 12px;
}
#polygonExtractionWidget QListWidget {
    font-size: 12px;
}
#polygonExtractionWidget QProgressBar {
    min-height: 18px;
    max-height: 18px;
}
"""

RECOGNITION_SCENE_FRAME_STYLE = """
#editorSceneFrame {
    border: 3px solid #DC2626;
}
"""


def _rgba(color: QColor, alpha: int) -> str:
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"


def toolbar_button_stylesheet(colors: IconColors, *, checkable: bool) -> str:
    """Flat style for icon-only editor buttons.

    Set on each button: a widget-level stylesheet replaces the ancestor rules,
    so the Kraken theme's filled QToolButton look would otherwise win. The
    active tool gets a translucent accent backdrop and an accent outline.
    """

    rules = [
        "QToolButton { padding: 0; margin: 0; background: transparent;"
        " border: 1px solid transparent; border-radius: 6px; }",
        f"QToolButton:hover {{ background-color: {_rgba(colors.normal, 28)}; }}",
        f"QToolButton:pressed {{ background-color: {_rgba(colors.normal, 46)}; }}",
        "QToolButton:disabled { background: transparent; border-color: transparent; }",
    ]
    if checkable:
        rules.append(
            f"QToolButton:checked {{ background-color: {_rgba(colors.on, 46)};"
            f" border: 1px solid {_rgba(colors.on, 150)}; }}"
        )
        rules.append(f"QToolButton:checked:hover {{ background-color: {_rgba(colors.on, 64)}; }}")
    return " ".join(rules)


__all__ = ["COMPACT_UI_STYLE", "RECOGNITION_SCENE_FRAME_STYLE", "toolbar_button_stylesheet"]
