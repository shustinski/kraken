from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from kraken_core.styles import load_shared_stylesheet, load_stylesheet


class ThemeManager:
    _THEME_FILES = {
        "Dark": "dark_modern.qss",
        "Light": "style.qss",
    }

    def __init__(self, settings_path: Path | None = None):
        if settings_path is None:
            settings_path = Path.cwd() / ".krona_ui.json"
        self._settings_path = settings_path
        self._theme_dir = Path(__file__).parent

    def available_themes(self) -> list[str]:
        return list(self._THEME_FILES.keys())

    def _read_settings(self) -> dict:
        if not self._settings_path.exists():
            return {}
        try:
            payload = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_settings(self, payload: dict) -> None:
        self._settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_saved_theme(self, default: str = "Dark") -> str:
        payload = self._read_settings()
        value = payload.get("theme")
        if value in self._THEME_FILES:
            return value
        return default

    def save_theme(self, theme_name: str) -> None:
        if theme_name not in self._THEME_FILES:
            return
        payload = self._read_settings()
        payload["theme"] = theme_name
        self._write_settings(payload)

    def load_saved_language(self, default: str = "English") -> str:
        payload = self._read_settings()
        value = payload.get("language")
        return value if isinstance(value, str) and value else default

    def save_language(self, language_name: str) -> None:
        payload = self._read_settings()
        payload["language"] = language_name
        self._write_settings(payload)

    def stylesheet_for(self, theme_name: str) -> str:
        filename = self._THEME_FILES.get(theme_name)
        if not filename:
            return ""
        path = self._theme_dir / filename
        qss = load_shared_stylesheet(filename)
        return qss or load_stylesheet(path)

    def apply_theme(self, app: QApplication, theme_name: str) -> bool:
        qss = self.stylesheet_for(theme_name)
        if not qss:
            return False
        app.setStyleSheet(qss)
        return True
