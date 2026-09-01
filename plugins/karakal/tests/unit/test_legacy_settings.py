from __future__ import annotations

import json

from PyQt6.QtCore import QSettings

from karakal.infra.services import KarakalSettingsService


def test_legacy_manager_settings_are_removed_without_resetting_other_state(tmp_path, caplog) -> None:
    path = tmp_path / "legacy.ini"
    settings = QSettings(str(path), QSettings.Format.IniFormat)
    settings.setValue("ui/management_settings", json.dumps({"primary_labeling_target_ratio": 0.1}))
    settings.setValue(
        "ui/build_settings",
        json.dumps({
            "app_mode": "management",
            "management_scenario": "primary_labeling_selection",
            "metric_key": "dice_score",
            "frames_per_row": 42,
        }),
    )
    settings.setValue(
        "ui/model_folders",
        json.dumps({"folders": [{"path": "C:/data/model", "checked": True}], "export_folder": "C:/exports"}),
    )
    settings.sync()

    with caplog.at_level("INFO"):
        service = KarakalSettingsService(settings)

    migrated = service.load_build_settings_payload()
    assert migrated["app_mode"] == "validation"
    assert "management_scenario" not in migrated
    assert migrated["metric_key"] == "dice_score"
    assert migrated["frames_per_row"] == 42
    assert service.load_folder_manager_payload()["export_folder"] == "C:/exports"
    assert not settings.contains("ui/management_settings")
    messages = [record.message for record in caplog.records if "Manager Mode settings" in record.message]
    assert messages == ["Removed obsolete Karakal Manager Mode settings; validation remains inactive until requested"]


def test_unknown_legacy_mode_is_not_interpreted_as_manager_mode(tmp_path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("ui/build_settings", json.dumps({"app_mode": "grid_inspection", "metric_key": "dice_score"}))

    service = KarakalSettingsService(settings)

    assert service.load_build_settings_payload() == {"app_mode": "grid_inspection", "metric_key": "dice_score"}
