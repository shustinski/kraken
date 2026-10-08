"""One look for the standalone Karakal on every Windows.

Without an explicit style Qt picks the system one: «windows11» (dark-aware) on Windows 11 and
the old light «windowsvista» on Windows 10, so the frame window, dialogs and message boxes
looked different. Karakal therefore uses Fusion with its own dark palette, the widget style
sheet for every window and dark window title bars. Inside Kraken the host keeps its own look.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory

from .ui_constants import EXTEND_ROOT_OBJECT_NAME, widget_stylesheet

_ACTIVE_COLORS = {
    QPalette.ColorRole.Window: "#15191f",
    QPalette.ColorRole.WindowText: "#edf3fb",
    QPalette.ColorRole.Base: "#10151c",
    QPalette.ColorRole.AlternateBase: "#18212b",
    QPalette.ColorRole.Text: "#edf3fb",
    QPalette.ColorRole.Button: "#1d2733",
    QPalette.ColorRole.ButtonText: "#edf3fb",
    QPalette.ColorRole.BrightText: "#ffffff",
    QPalette.ColorRole.Highlight: "#275fbb",
    QPalette.ColorRole.HighlightedText: "#ffffff",
    QPalette.ColorRole.ToolTipBase: "#1d2733",
    QPalette.ColorRole.ToolTipText: "#edf3fb",
    QPalette.ColorRole.PlaceholderText: "#6f7a86",
    QPalette.ColorRole.Link: "#6fa8ff",
    QPalette.ColorRole.LinkVisited: "#a98bff",
    QPalette.ColorRole.Light: "#3a4a5c",
    QPalette.ColorRole.Midlight: "#2c3a4a",
    QPalette.ColorRole.Mid: "#232e3a",
    QPalette.ColorRole.Dark: "#0b0f14",
    QPalette.ColorRole.Shadow: "#000000",
}
_DISABLED_COLORS = {
    QPalette.ColorRole.WindowText: "#6f7a86",
    QPalette.ColorRole.Text: "#6f7a86",
    QPalette.ColorRole.ButtonText: "#6f7a86",
    QPalette.ColorRole.Button: "#161b22",
    QPalette.ColorRole.Base: "#12171d",
    QPalette.ColorRole.Highlight: "#2f3a47",
    QPalette.ColorRole.HighlightedText: "#9aa6b2",
}

# Windows that are not inside the main widget: menus, tooltips, message boxes, frame window.
_EXTRA_STYLESHEET = """
QToolTip { background-color: #1d2733; color: #edf3fb; border: 1px solid #3a5068; padding: 4px 6px; }
QMenu { background-color: #11161d; color: #edf3fb; border: 1px solid #30445a; padding: 4px 0px; }
QMenu::item { padding: 5px 24px 5px 22px; background-color: transparent; }
QMenu::item:selected { background-color: #275fbb; color: #ffffff; }
QMenu::item:disabled { color: #6f7a86; }
QMenu::separator { height: 1px; background-color: #30445a; margin: 4px 8px; }
QMenuBar { background-color: #11161d; color: #edf3fb; }
QMenuBar::item { background-color: transparent; padding: 4px 10px; }
QMenuBar::item:selected { background-color: #1d2a38; }
QMessageBox QLabel { background-color: transparent; }
QDialogButtonBox QPushButton { min-width: 84px; padding: 5px 12px; }
QDialog QLineEdit, QDialog QComboBox, QDialog QSpinBox, QDialog QDoubleSpinBox { padding: 3px 8px; min-height: 22px; border-radius: 6px; }
QTextBrowser, QPlainTextEdit, QTextEdit { background-color: #10151c; color: #edf3fb; border: 1px solid #30445a; border-radius: 8px; }
"""


def app_stylesheet() -> str:
    """The main widget's style sheet for every window, plus menus, tooltips and dialogs."""

    scoped = widget_stylesheet()
    lines = []
    for line in scoped.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"#{EXTEND_ROOT_OBJECT_NAME} {{"):
            continue
        lines.append(line.replace(f"#{EXTEND_ROOT_OBJECT_NAME} ", ""))
    return "\n".join(lines) + _EXTRA_STYLESHEET


def karakal_palette() -> QPalette:
    palette = QPalette()
    for role, color in _ACTIVE_COLORS.items():
        palette.setColor(role, QColor(color))
    for role, color in _DISABLED_COLORS.items():
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(color))
    return palette


def apply_karakal_app_theme(app: QApplication) -> None:
    """Fusion, the dark Karakal palette and style sheet, dark title bars: same on Windows 10 and 11."""

    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):
        # Dark window frames (title bars) on Windows 10 and 11.
        hints.setColorScheme(Qt.ColorScheme.Dark)
    style = QStyleFactory.create("Fusion")
    if style is not None:
        app.setStyle(style)
    app.setPalette(karakal_palette())
    app.setStyleSheet(app_stylesheet())
