"""One look on Windows 10 and 11: Fusion, the Karakal palette and style sheet for every window."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication

from karakal.ui.app_theme import app_stylesheet, apply_karakal_app_theme, karakal_palette
from karakal.ui.ui_constants import EXTEND_ROOT_OBJECT_NAME, widget_stylesheet


def test_widget_stylesheet_has_bundled_arrows() -> None:
    sheet = widget_stylesheet()
    urls = [part.split('")', 1)[0] for part in sheet.split('url("')[1:]]
    assert len(urls) >= 2
    assert all(Path(url).is_file() for url in urls)


def test_app_stylesheet_covers_every_window() -> None:
    sheet = app_stylesheet()
    assert f"#{EXTEND_ROOT_OBJECT_NAME}" not in sheet
    assert "QToolTip" in sheet and "QMenu::item:selected" in sheet
    assert "QComboBox::down-arrow" in sheet


def test_palette_is_dark() -> None:
    palette = karakal_palette()
    assert palette.color(QPalette.ColorRole.Window).lightness() < 40
    assert palette.color(QPalette.ColorRole.WindowText).lightness() > 200


def test_theme_makes_fusion_the_style(qtbot, monkeypatch) -> None:
    app = QApplication.instance()
    previous_palette = QPalette(app.palette())
    previous_sheet = app.styleSheet()
    previous_scheme = app.styleHints().colorScheme()
    styles: list[str] = []
    # Other tests share this application: record the style instead of switching it.
    monkeypatch.setattr(app, "setStyle", lambda style: styles.append(style.name()))
    try:
        apply_karakal_app_theme(app)
        assert [name.lower() for name in styles] == ["fusion"]
        assert app.palette().color(QPalette.ColorRole.Window) == karakal_palette().color(QPalette.ColorRole.Window)
        assert app.styleSheet() == app_stylesheet()
    finally:
        app.setStyleSheet(previous_sheet)
        app.setPalette(previous_palette)
        if hasattr(app.styleHints(), "unsetColorScheme"):
            app.styleHints().unsetColorScheme()
        else:
            app.styleHints().setColorScheme(previous_scheme)
