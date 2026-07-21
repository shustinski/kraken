from __future__ import annotations

import json

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QListWidgetItem

from contour.application.extraction_profiles import default_contour_settings_profiles
from contour.infrastructure.settings_store import WidgetViaPresetSettingsStore
from contour.ui.item_status_painting import FRAME_STATUS_ROLE, paint_image_row_item


def test_default_extraction_profiles_keep_expected_modes() -> None:
    profiles = default_contour_settings_profiles()

    assert sorted(profiles) == ["conductors", "vias"]
    assert profiles["conductors"].object_type == "conductor"
    assert profiles["conductors"].output_mode == "polygon"
    assert profiles["vias"].object_type == "via"
    assert profiles["vias"].output_mode == "box"


def test_via_preset_store_ignores_invalid_payload(tmp_path) -> None:
    settings_path = tmp_path / "settings.ini"

    def settings_factory() -> QSettings:
        return QSettings(str(settings_path), QSettings.Format.IniFormat)

    settings = settings_factory()
    settings.setValue("via_search/user_presets", "[")
    settings.sync()

    assert WidgetViaPresetSettingsStore(settings_factory).load() == {}


def test_via_preset_store_round_trips_payload(tmp_path) -> None:
    settings_path = tmp_path / "settings.ini"

    def settings_factory() -> QSettings:
        return QSettings(str(settings_path), QSettings.Format.IniFormat)

    store = WidgetViaPresetSettingsStore(settings_factory)
    store.save({"fast vias": {"via_search_mode": "bright_tophat_dog", "debug_enabled": True}})

    assert store.load() == {"fast vias": {"via_search_mode": "bright_tophat_dog", "debug_enabled": True}}
    raw = settings_factory().value("via_search/user_presets", "", type=str)
    assert json.loads(raw)["fast vias"]["via_search_mode"] == "bright_tophat_dog"


def test_paint_image_row_item_sets_status_and_text() -> None:
    item = QListWidgetItem()

    paint_image_row_item(
        item,
        r"d:\frames\frame_001.png",
        image_has_changes=False,
        has_vector_overlay=True,
        extraction_enabled=False,
        viewed=True,
        persisted_highlight=False,
    )

    assert item.data(FRAME_STATUS_ROLE) is not None
    assert item.text() == "frame_001"
