from __future__ import annotations

from PyQt6.QtCore import QSettings

import karakal.app.main_window as main_window


def test_widget_exposes_only_supported_application_modes(tmp_path, monkeypatch, qtbot) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(main_window, "QSettings", lambda *_args: settings)
    widget = main_window.KarakalWidget()
    qtbot.addWidget(widget)

    modes = {str(widget.app_mode_combo.itemData(index)) for index in range(widget.app_mode_combo.count())}
    switch_modes = set(widget.analysis_setup_panel._mode_buttons)

    assert modes == {"validation", "grid_inspection"}
    assert switch_modes == modes
    assert not hasattr(widget, "mode_toggle_button")
    assert not widget.app_mode_combo.isVisibleTo(widget)
    assert widget.main_mode_stack.count() == 2
    assert not any("manager" in action.text().lower() for action in widget.findChildren(main_window.QAction))


def test_validation_profile_follows_the_sources() -> None:
    from kraken_core.analysis_protocol import AnalysisProfileKind as Kind

    from karakal.core.analysis_profiles import auto_validation_profile

    assert auto_validation_profile(2, 0) == Kind.MODEL_COMPARISON
    assert auto_validation_profile(3, 3) == Kind.MODEL_COMPARISON
    assert auto_validation_profile(1, 1) == Kind.CONFIDENCE_AUDIT
    assert auto_validation_profile(1, 0) == Kind.SINGLE_RESULT_RISK
    # Without models nothing jumps; cell inspection is never picked for validation.
    assert auto_validation_profile(0, 0, Kind.SINGLE_RESULT_RISK) == Kind.SINGLE_RESULT_RISK
    assert auto_validation_profile(0, 0, Kind.GRID_DEFECTS) == Kind.MODEL_COMPARISON


def _add_folder(tmp_path, presenter, name: str, *, confidence: bool = False):
    import numpy as np

    from karakal.core.image_io import _grayscale_array_to_qimage
    from karakal.ui.ui_constants import FOLDER_CONFIDENCE_ROLE

    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    assert _grayscale_array_to_qimage(np.full((8, 8), 255, dtype=np.uint8)).save(str(folder / "F_0001.png"))
    item = presenter._append_folder_item(folder, checked=True)
    if confidence:
        item.setData(FOLDER_CONFIDENCE_ROLE, str(folder))
    presenter._refresh_folder_rows()
    presenter._sync_action_buttons()
    return item


def test_mode_switch_sets_mode_and_picks_validation_analysis(tmp_path, qtbot) -> None:
    from kraken_core.analysis_protocol import AnalysisProfileKind as Kind

    from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE

    settings_path = tmp_path / "karakal.ini"
    widget = main_window.KarakalWidget(settings=QSettings(str(settings_path), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    panel = widget.analysis_setup_panel

    first = _add_folder(tmp_path, presenter, "model_a", confidence=True)
    assert presenter._analysis_profile == Kind.CONFIDENCE_AUDIT
    _add_folder(tmp_path, presenter, "model_b")
    assert presenter._analysis_profile == Kind.MODEL_COMPARISON
    assert widget.app_mode_combo.currentData() == "validation"
    assert widget._t("profile.model_comparison.title") in panel.profile_description_label.text()

    panel._mode_buttons["grid_inspection"].click()
    assert presenter._analysis_profile == Kind.GRID_DEFECTS
    assert widget.app_mode_combo.currentData() == "grid_inspection"
    assert widget.main_mode_stack.currentIndex() == 1
    # Sources changing in cell inspection keep the mode.
    first.setData(FOLDER_CHECKED_ROLE, False)
    presenter._sync_action_buttons()
    assert presenter._analysis_profile == Kind.GRID_DEFECTS

    panel._mode_buttons["validation"].click()
    assert presenter._analysis_profile == Kind.SINGLE_RESULT_RISK
    assert widget.main_mode_stack.currentIndex() == 0
    assert not widget._matrix_analysis_mode_row.isVisibleTo(widget)

    panel._mode_buttons["grid_inspection"].click()
    presenter._persist_state()
    restored = main_window.KarakalWidget(settings=QSettings(str(settings_path), QSettings.Format.IniFormat))
    qtbot.addWidget(restored)
    assert restored._presenter._analysis_profile == Kind.GRID_DEFECTS
    assert restored.app_mode_combo.currentData() == "grid_inspection"
    assert restored.analysis_setup_panel._mode_buttons["grid_inspection"].isChecked()


def test_setting_the_mode_directly_aligns_the_profile(tmp_path, qtbot) -> None:
    from kraken_core.analysis_protocol import AnalysisProfileKind as Kind

    widget = main_window.KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    assert widget._presenter._analysis_profile == Kind.GRID_DEFECTS
    assert widget.analysis_setup_panel._mode_buttons["grid_inspection"].isChecked()
