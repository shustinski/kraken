from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PyQt6.QtCore import QSettings

from polygon_widget.infrastructure.settings_store import WidgetDisplaySettingsStore


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
                    "line_width": 3.5,
                    "show_vertices": False,
                    "show_neighbor_frames": True,
                    "neighbor_columns": 6,
                    "neighbor_max_grid": 5,
                    "neighbor_opacity": 0.45,
                    "neighbor_overlap_pixels": 12,
                }
            )

            payload = store.load()

            self.assertEqual(payload["external_color"], "#112233")
            self.assertEqual(payload["line_width"], 3.5)
            self.assertFalse(payload["show_vertices"])
            self.assertTrue(payload["show_neighbor_frames"])
            self.assertEqual(payload["neighbor_columns"], 6)
            self.assertEqual(payload["neighbor_max_grid"], 5)
            self.assertEqual(payload["neighbor_opacity"], 0.45)
            self.assertEqual(payload["neighbor_overlap_pixels"], 12)


if __name__ == "__main__":
    unittest.main()
