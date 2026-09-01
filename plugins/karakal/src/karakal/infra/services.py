"""Provide small persisted-settings helpers for Karakal."""
from __future__ import annotations

import json
import logging

from PyQt6.QtCore import QSettings

from ..core.performance import PerformanceConfig, load_performance_config
from ..ui.ui_constants import (
    SETTINGS_BUILD_KEY,
    SETTINGS_ANALYSIS_PROFILE_KEY,
    SETTINGS_DETAILS_VIEW_KEY,
    SETTINGS_FOLDERS_KEY,
    SETTINGS_LANGUAGE_KEY,
    SETTINGS_PERFORMANCE_KEY,
    SETTINGS_VALIDATION_MASK_KEY,
)


_LOGGER = logging.getLogger(__name__)
_LEGACY_MANAGEMENT_SETTINGS_KEY = "ui/management_settings"
_LEGACY_MANAGER_PREFIXES = ("manager_", "management_", "primary_labeling_", "labeling_priority_")
_LEGACY_MANAGER_MODES = {"manager", "management", "manager_mode"}


class KarakalSettingsService:
    """Read and write persisted UI state for Karakal."""

    def __init__(self, settings: QSettings) -> None:
        self._settings = settings
        self._migrate_legacy_manager_settings()

    def load_folder_manager_payload(self) -> dict:
        return self._load_payload(SETTINGS_FOLDERS_KEY)

    def save_folder_manager_payload(self, payload: dict) -> None:
        self._save_payload(SETTINGS_FOLDERS_KEY, payload)

    def load_build_settings_payload(self) -> dict:
        return self._load_payload(SETTINGS_BUILD_KEY)

    def save_build_settings_payload(self, payload: dict) -> None:
        self._save_payload(SETTINGS_BUILD_KEY, payload)

    def load_analysis_profile_payload(self) -> dict:
        return self._load_payload(SETTINGS_ANALYSIS_PROFILE_KEY)

    def save_analysis_profile_payload(self, payload: dict) -> None:
        self._save_payload(SETTINGS_ANALYSIS_PROFILE_KEY, payload)

    def load_details_view_payload(self) -> dict:
        return self._load_payload(SETTINGS_DETAILS_VIEW_KEY)

    def save_details_view_payload(self, payload: dict) -> None:
        self._save_payload(SETTINGS_DETAILS_VIEW_KEY, payload)

    def load_validation_mask_payload(self) -> dict:
        return self._load_payload(SETTINGS_VALIDATION_MASK_KEY)

    def save_validation_mask_payload(self, payload: dict) -> None:
        self._save_payload(SETTINGS_VALIDATION_MASK_KEY, payload)

    def load_performance_config(self) -> PerformanceConfig:
        return load_performance_config(self._load_payload(SETTINGS_PERFORMANCE_KEY))

    def save_performance_config(self, config: PerformanceConfig) -> None:
        self._save_payload(SETTINGS_PERFORMANCE_KEY, config.to_payload())

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
        except (TypeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_payload(self, key: str, payload: dict) -> None:
        self._settings.setValue(key, json.dumps(payload, ensure_ascii=False))

    def _migrate_legacy_manager_settings(self) -> None:
        """Drop only Manager Mode state and preserve all unrelated user settings."""

        changed = False
        if self._settings.contains(_LEGACY_MANAGEMENT_SETTINGS_KEY):
            self._settings.remove(_LEGACY_MANAGEMENT_SETTINGS_KEY)
            changed = True
        for key in (SETTINGS_BUILD_KEY, SETTINGS_FOLDERS_KEY, SETTINGS_DETAILS_VIEW_KEY):
            payload = self._load_payload(key)
            if not payload:
                continue
            migrated = dict(payload)
            for field in tuple(migrated):
                normalized = str(field).strip().lower()
                if normalized.startswith(_LEGACY_MANAGER_PREFIXES):
                    migrated.pop(field, None)
                    changed = True
            for field in ("app_mode", "selected_mode", "current_mode"):
                if str(migrated.get(field, "")).strip().lower() in _LEGACY_MANAGER_MODES:
                    migrated[field] = "validation"
                    changed = True
            if migrated != payload:
                self._save_payload(key, migrated)
        if changed:
            _LOGGER.warning("Removed obsolete Karakal Manager Mode settings; validation remains inactive until requested")
            self._settings.sync()
