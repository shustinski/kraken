from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PyQt6.QtCore import QSettings

from contour.infrastructure.settings_store import (
    WidgetAppearanceSettingsStore,
    WidgetDisplaySettingsStore,
    WidgetGamificationProfileStore,
    WidgetSessionSettingsStore,
)


class WidgetDisplaySettingsStoreTests(unittest.TestCase):
    def test_display_settings_round_trip_through_qsettings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.ini"

            def _settings_factory() -> QSettings:
                return QSettings(str(settings_path), QSettings.Format.IniFormat)

            store = WidgetDisplaySettingsStore(settings_factory=_settings_factory)
            store.save(
                {
                    "external_color": "#112233",
                    "conductor_hover_highlight_color": "#445566",
                    "line_width": 3.5,
                    "show_vertices": False,
                    "random_object_colors": True,
                    "autosave_on_frame_transition": True,
                    "show_frame_matrix": False,
                    "show_frame_matrix_thumbnails": False,
                    "show_neighbor_frames": True,
                    "neighbor_columns": 6,
                    "neighbor_max_grid": 5,
                    "neighbor_opacity": 0.45,
                    "neighbor_overlap_pixels": 12,
                }
            )

            payload = store.load()

            self.assertEqual(payload["external_color"], "#112233")
            self.assertEqual(payload["conductor_hover_highlight_color"], "#445566")
            self.assertEqual(payload["line_width"], 3.5)
            self.assertFalse(payload["show_vertices"])
            self.assertTrue(payload["random_object_colors"])
            self.assertTrue(payload["autosave_on_frame_transition"])
            self.assertFalse(payload["show_frame_matrix"])
            self.assertFalse(payload["show_frame_matrix_thumbnails"])
            self.assertTrue(payload["show_neighbor_frames"])
            self.assertEqual(payload["neighbor_columns"], 6)
            self.assertEqual(payload["neighbor_max_grid"], 5)
            self.assertEqual(payload["neighbor_opacity"], 0.45)
            self.assertEqual(payload["neighbor_overlap_pixels"], 12)


class WidgetAppearanceSettingsStoreTests(unittest.TestCase):
    def test_language_and_theme_round_trip_through_qsettings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.ini"

            def _settings_factory() -> QSettings:
                return QSettings(str(settings_path), QSettings.Format.IniFormat)

            store = WidgetAppearanceSettingsStore(settings_factory=_settings_factory)

            self.assertEqual(store.load_language(), "ru")
            self.assertEqual(store.load_theme(), "dark")

            store.save_language("en")
            store.save_theme("light")

            self.assertEqual(store.load_language(), "en")
            self.assertEqual(store.load_theme(), "light")

    def test_invalid_language_and_theme_values_fall_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.ini"

            def _settings_factory() -> QSettings:
                return QSettings(str(settings_path), QSettings.Format.IniFormat)

            settings = _settings_factory()
            settings.setValue("appearance/language", "de")
            settings.setValue("appearance/theme", "unknown")
            settings.sync()

            store = WidgetAppearanceSettingsStore(settings_factory=_settings_factory)

            self.assertEqual(store.load_language(), "ru")
            self.assertEqual(store.load_theme(), "dark")


class WidgetGamificationProfileStoreTests(unittest.TestCase):
    def test_gamification_profile_payload_round_trip_through_qsettings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.ini"

            def _settings_factory() -> QSettings:
                return QSettings(str(settings_path), QSettings.Format.IniFormat)

            store = WidgetGamificationProfileStore(settings_factory=_settings_factory)
            payload = {"wallet_balance": 12, "selected_pet": "kraken"}

            store.save_payload(payload)

            self.assertEqual(store.load_payload(), payload)


class WidgetSessionSettingsStoreTests(unittest.TestCase):
    def test_current_image_path_round_trip_through_qsettings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.ini"

            def _settings_factory() -> QSettings:
                return QSettings(str(settings_path), QSettings.Format.IniFormat)

            store = WidgetSessionSettingsStore(settings_factory=_settings_factory)

            store.save_current_image_path("frame_001.png")
            self.assertEqual(store.load_current_image_path(), str(Path("frame_001.png")))

            store.save_current_image_path(None)
            self.assertIsNone(store.load_current_image_path())


if __name__ == "__main__":
    unittest.main()
