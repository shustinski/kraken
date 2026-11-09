from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PyQt6.QtCore import QSettings

from contour.application.dto import PersistedPaths
from contour.infrastructure.settings_store import (
    IMAGE_LIST_MODE_EXPLICIT,
    WidgetAppearanceSettingsStore,
    WidgetDisplaySettingsStore,
    WidgetGamificationProfileStore,
    WidgetPathSettingsStore,
    WidgetSessionSettingsStore,
)

# Regression tests for defect D13 (docs/quality/defects.md) live in this module.
kraken_defect = "D13"


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
                    "via_selection_color": "#ABCDEF",
                    "conductor_hover_highlight_color": "#445566",
                    "line_width": 3.5,
                    "show_vertices": False,
                    "random_object_colors": True,
                    "autosave_on_frame_transition": True,
                    "show_frame_matrix": False,
                    "show_frame_matrix_thumbnails": False,
                    "show_neighbor_frames": True,
                    "show_neighbor_vectors": True,
                    "neighbor_columns": 6,
                    "neighbor_max_grid": 5,
                    "neighbor_opacity": 0.45,
                    "neighbor_overlap_pixels": 12,
                }
            )

            payload = store.load()

            self.assertEqual(payload["external_color"], "#112233")
            self.assertEqual(payload["via_selection_color"], "#ABCDEF")
            self.assertEqual(payload["conductor_hover_highlight_color"], "#445566")
            self.assertEqual(payload["line_width"], 3.5)
            self.assertFalse(payload["show_vertices"])
            self.assertTrue(payload["random_object_colors"])
            self.assertTrue(payload["autosave_on_frame_transition"])
            self.assertFalse(payload["show_frame_matrix"])
            self.assertFalse(payload["show_frame_matrix_thumbnails"])
            self.assertTrue(payload["show_neighbor_frames"])
            self.assertTrue(payload["show_neighbor_vectors"])
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

    def test_session_image_and_vector_paths_round_trip_through_qsettings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.ini"

            def _settings_factory() -> QSettings:
                return QSettings(str(settings_path), QSettings.Format.IniFormat)

            store = WidgetSessionSettingsStore(settings_factory=_settings_factory)

            store.save_image_paths(["frame_001.png", Path("frame_002.jpg"), "frame_001.png"])
            store.save_vector_paths(["frame_001.cif", Path("frame_002.cv"), "frame_001.cif"])

            self.assertEqual(
                store.load_image_paths(),
                [str(Path("frame_001.png")), str(Path("frame_002.jpg"))],
            )
            self.assertEqual(
                store.load_vector_paths(),
                [str(Path("frame_001.cif")), str(Path("frame_002.cv"))],
            )

            store.save_image_paths([])
            store.save_vector_paths([])

            self.assertEqual(store.load_image_paths(), [])
            self.assertEqual(store.load_vector_paths(), [])


class ManagedJobSettingsStoreTests(unittest.TestCase):
    """D13: a managed Kraken job (KRAKEN_JOB_MANIFEST set by the agent runner) must
    not persist job staging paths into the host user's settings profile."""

    @staticmethod
    def _stores(directory: str) -> tuple[WidgetPathSettingsStore, WidgetSessionSettingsStore, Path]:
        settings_path = Path(directory) / "settings.ini"

        def _settings_factory() -> QSettings:
            return QSettings(str(settings_path), QSettings.Format.IniFormat)

        return (
            WidgetPathSettingsStore(settings_factory=_settings_factory),
            WidgetSessionSettingsStore(settings_factory=_settings_factory),
            settings_path,
        )

    def _job_environment(self, settings_path: Path):
        return mock.patch.dict(os.environ, {"KRAKEN_JOB_MANIFEST": str(settings_path / "job.json")})

    def test_managed_job_does_not_persist_job_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path_store, session_store, settings_path = self._stores(directory)
            with self._job_environment(settings_path):
                path_store.save(
                    PersistedPaths(
                        input_directory=str(Path(directory) / "staging" / "inputs"),
                        cif_directory=str(Path(directory) / "staging" / "vectors"),
                        output_directory=str(Path(directory) / "staging" / "outputs"),
                        dataset_directory=str(Path(directory) / "staging" / "dataset"),
                    )
                )
                session_store.save_image_paths([str(Path(directory) / "staging" / "inputs" / "1_1.jpg")])
                session_store.save_vector_paths([str(Path(directory) / "staging" / "vectors" / "1_1.cif")])
                session_store.save_current_image_path(str(Path(directory) / "staging" / "inputs" / "1_1.jpg"))
                session_store.save_viewed_image_paths([str(Path(directory) / "staging" / "inputs" / "1_1.jpg")])
                session_store.save_image_list_mode(IMAGE_LIST_MODE_EXPLICIT)

            settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
            written_keys = set(settings.allKeys())
            for key in (
                "paths/input_directory",
                "paths/cif_directory",
                "paths/output_directory",
                "paths/dataset_directory",
                "session/image_paths",
                "session/vector_paths",
                "session/viewed_image_paths",
                "session/current_image_path",
                "session/image_list_mode",
            ):
                self.assertNotIn(key, written_keys)
            settings.sync()

    def test_persistence_still_writes_outside_managed_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path_store, session_store, settings_path = self._stores(directory)
            with mock.patch.dict(os.environ, {"KRAKEN_JOB_MANIFEST": ""}):
                path_store.save(
                    PersistedPaths(
                        input_directory=str(Path(directory) / "inputs"),
                        output_directory=str(Path(directory) / "outputs"),
                    )
                )
                session_store.save_image_paths([str(Path(directory) / "inputs" / "1_1.jpg")])
                session_store.save_current_image_path(str(Path(directory) / "inputs" / "1_1.jpg"))
                session_store.save_image_list_mode(IMAGE_LIST_MODE_EXPLICIT)

            settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
            self.assertEqual(settings.value("paths/input_directory", "", type=str), str(Path(directory) / "inputs"))
            self.assertEqual(settings.value("paths/output_directory", "", type=str), str(Path(directory) / "outputs"))
            self.assertEqual(settings.value("session/image_paths", "", type=str) != "", True)
            self.assertEqual(settings.value("session/current_image_path", "", type=str), str(Path(directory) / "inputs" / "1_1.jpg"))
            self.assertEqual(settings.value("session/image_list_mode", "", type=str), IMAGE_LIST_MODE_EXPLICIT)
            settings.sync()


if __name__ == "__main__":
    unittest.main()
