from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu

from .styles import load_shared_stylesheet

THEME_DARK = "dark"
THEME_LIGHT = "light"
THEME_SYSTEM = "system"
THEME_STYLESHEETS = {
    THEME_DARK: "dark_modern.qss",
    THEME_LIGHT: "style.qss",
}


def normalize_theme(theme: str | None) -> str:
    value = str(theme or "").strip().lower()
    if value == THEME_SYSTEM:
        return THEME_SYSTEM
    if value in {"light", "светлая", "светлый"}:
        return THEME_LIGHT
    return THEME_DARK


def active_theme(theme: str | None = None) -> str:
    """Return dark or light, resolving the system choice and the saved application theme."""

    if theme is None:
        app = QApplication.instance()
        theme = None if app is None else str(app.property("krakenTheme") or "")
    normalized = normalize_theme(theme)
    if normalized == THEME_SYSTEM:
        app = QApplication.instance()
        dark = app is not None and app.styleHints().colorScheme() == Qt.ColorScheme.Dark
        return THEME_DARK if dark else THEME_LIGHT
    return normalized


_SCENE_COLORS = {
    THEME_DARK: {
        "matrix": "#0b1120",
        "pipeline": "#1b1d1b",
        "grid_minor": "#272a27",
        "grid_major": "#343834",
        "empty": "#1f2937",
        "selection": "#f8fafc",
        "label": "#f8fafc",
        "minimap": "#0b1120",
        "minimap_matrix": "#1e293b",
        "minimap_border": "#475569",
    },
    THEME_LIGHT: {
        "matrix": "#f4f6f8",
        "pipeline": "#f7f8f6",
        "grid_minor": "#e6e8e6",
        "grid_major": "#d5d8d5",
        "empty": "#e8edf2",
        "selection": "#0f172a",
        "label": "#111827",
        "minimap": "#f8fafc",
        "minimap_matrix": "#e8eef5",
        "minimap_border": "#94a3b8",
    },
}


def scene_color(name: str, theme: str | None = None) -> str:
    return _SCENE_COLORS[active_theme(theme)][name]


def shared_theme_stylesheet(theme: str | None) -> str:
    return load_shared_stylesheet(THEME_STYLESHEETS[active_theme(theme)])


def apply_app_theme(theme: str | None, app: QApplication | None = None) -> str:
    normalized = normalize_theme(theme)
    target = app or QApplication.instance()
    if target is not None:
        target.setProperty("krakenTheme", normalized)
        if not target.property("krakenThemeListener"):
            target.styleHints().colorSchemeChanged.connect(
                lambda _scheme: apply_app_theme(THEME_SYSTEM, target)
                if target.property("krakenTheme") == THEME_SYSTEM else None
            )
            target.setProperty("krakenThemeListener", True)
        target.setStyleSheet(shared_theme_stylesheet(normalized))
        for widget in target.allWidgets():
            refresh = getattr(widget, "apply_kraken_scene_theme", None)
            if callable(refresh):
                refresh()
    return normalized


def add_theme_menu(
    window: QMainWindow,
    *,
    initial_theme: str = THEME_DARK,
    on_theme_changed: Callable[[str], None] | None = None,
    title: str = "Тема",
    dark_text: str = "Темная",
    light_text: str = "Светлая",
) -> QMenu:
    menu = window.menuBar().addMenu(title)
    group = QActionGroup(window)
    group.setExclusive(True)

    def _add_action(text: str, theme: str) -> QAction:
        action = QAction(text, window)
        action.setCheckable(True)
        action.setData(theme)
        action.setChecked(normalize_theme(initial_theme) == theme)
        group.addAction(action)
        menu.addAction(action)
        action.triggered.connect(lambda _checked=False, selected=theme: _apply(selected))
        return action

    def _apply(theme: str) -> None:
        if on_theme_changed is not None:
            on_theme_changed(theme)
        else:
            apply_app_theme(theme)
        for action in group.actions():
            action.setChecked(action.data() == normalize_theme(theme))

    dark_action = _add_action(dark_text, THEME_DARK)
    light_action = _add_action(light_text, THEME_LIGHT)

    # Keep Qt objects alive on PyQt builds that do not retain wrapper refs reliably.
    setattr(window, "_kraken_theme_menu", menu)
    setattr(window, "_kraken_theme_action_group", group)
    setattr(window, "_kraken_theme_dark_action", dark_action)
    setattr(window, "_kraken_theme_light_action", light_action)
    return menu
