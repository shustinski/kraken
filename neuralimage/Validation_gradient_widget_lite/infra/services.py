"""Provide small persisted-settings helpers for the lite widget."""
from __future__ import annotations

import json

from PyQt6.QtCore import QSettings

from ..ui.ui_constants import SETTINGS_BUILD_KEY, SETTINGS_DETAILS_VIEW_KEY, SETTINGS_FOLDERS_KEY, SETTINGS_LANGUAGE_KEY


class ValidationGradientLiteSettingsService:
    """Read and write persisted UI state for the lite widget."""

    def __init__(self, settings: QSettings) -> None:
        self._settings = settings

    def load_folder_manager_payload(self) -> dict:
        return self._load_payload(SETTINGS_FOLDERS_KEY)

    def save_folder_manager_payload(self, payload: dict) -> None:
        self._save_payload(SETTINGS_FOLDERS_KEY, payload)

    def load_build_settings_payload(self) -> dict:
        return self._load_payload(SETTINGS_BUILD_KEY)

    def save_build_settings_payload(self, payload: dict) -> None:
        self._save_payload(SETTINGS_BUILD_KEY, payload)

    def load_details_view_payload(self) -> dict:
        return self._load_payload(SETTINGS_DETAILS_VIEW_KEY)

    def save_details_view_payload(self, payload: dict) -> None:
        self._save_payload(SETTINGS_DETAILS_VIEW_KEY, payload)

    def load_language(self) -> str | None:
        value = self._settings.value(SETTINGS_LANGUAGE_KEY, "", str)
        return str(value) if value else None

    def save_language(self, language: str) -> None:
        self._settings.setValue(SETTINGS_LANGUAGE_KEY, str(language))

    def sync(self) -> None:
        self._settings.sync()

    def _load_payload(self, key: str) -> dict:
        raw = self._settings.value(key, "", str)
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_payload(self, key: str, payload: dict) -> None:
        self._settings.setValue(key, json.dumps(payload, ensure_ascii=False))


