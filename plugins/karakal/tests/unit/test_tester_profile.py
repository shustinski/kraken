"""Tester profile hides unfinished UI and ignores saved extra modes."""
from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QCheckBox, QPushButton
from kraken_core.analysis_protocol import AnalysisProfileKind

from karakal.app.main_window import KarakalWidget
from karakal.core.features import show_class_conflict
from karakal.core.features import tester_build as profile_is_tester
from karakal.core.grid_anomaly import GridDamageAnalysisConfig


def test_dev_profile_is_the_default(monkeypatch) -> None:
    monkeypatch.delenv("KARAKAL_BUILD_PROFILE", raising=False)
    assert profile_is_tester() is False
    assert show_class_conflict() is True
    from karakal.core.features import show_confidence_defects

    assert show_confidence_defects() is True


def test_confidence_defects_enabled_in_tester_profile(monkeypatch) -> None:
    monkeypatch.setenv("KARAKAL_BUILD_PROFILE", "tester")
    from karakal.core.features import show_confidence_defects

    assert show_confidence_defects() is True
    assert profile_is_tester() is True
    assert show_class_conflict() is False


def test_tester_profile_hides_unfinished_controls(tmp_path, qtbot, monkeypatch) -> None:
    monkeypatch.setenv("KARAKAL_BUILD_PROFILE", "tester")
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)

    profile_buttons = [
        button
        for button in widget.analysis_setup_panel.findChildren(QPushButton)
        if bool(button.property("analysisProfile"))
    ]
    visible_profiles = [button for button in profile_buttons if button.isVisibleTo(widget)]
    assert len(profile_buttons) == 4
    assert visible_profiles == []
    assert widget._presenter._analysis_profile == AnalysisProfileKind.GRID_DEFECTS
    assert widget.app_mode_combo.currentData() == "grid_inspection"
    assert not widget.mode_toggle_button.isVisibleTo(widget)
    assert not widget.pair_matrix_group.isVisibleTo(widget)
    assert not widget._grid_reference_frame_row.isVisibleTo(widget)
    # Confidence fill defects are enabled in tester; the compute checkbox stays available.
    assert not widget.grid_layer_compute_checks["confidence"].isHidden()
    assert widget.grid_layer_compute_checks["confidence"].isChecked()
    assert widget.grid_layer_display_combo.findData("confidence") >= 0
    assert not widget.grid_layer_compute_checks["derived_conflict"].isVisibleTo(widget)
    assert not widget.grid_error_type_checks["class_conflict"].isVisibleTo(widget)
    assert not widget.grid_error_type_checks["conductor_zone"].isVisibleTo(widget)
    assert not widget.btn_export_layer.isVisibleTo(widget)
    payload = widget._presenter._grid_inspection_config_payload()
    assert "derived_conflict" not in payload["requested_layers"]
    assert "confidence" in payload["requested_layers"]
    assert "binary" in payload["requested_layers"]
    assert "class_conflict" not in payload["enabled_error_types"]
    assert "calibration_examples" not in payload
    state = SimpleNamespace(grid_inspection_reference_record_key="saved-reference")
    assert widget._presenter._grid_inspection_reference_profile_for_state(state, GridDamageAnalysisConfig()) is None


def test_tester_profile_stays_hidden_after_two_models(tmp_path, qtbot, monkeypatch) -> None:
    monkeypatch.setenv("KARAKAL_BUILD_PROFILE", "tester")
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    for name in ("model_one", "model_zero"):
        folder = tmp_path / name
        folder.mkdir()
        presenter._append_folder_item(folder, checked=True)

    def hidden() -> None:
        assert not widget.pair_matrix_group.isVisibleTo(widget)
        assert not widget.grid_layer_compute_checks["confidence"].isHidden()
        assert widget.grid_layer_display_combo.findData("confidence") >= 0
        assert not widget.grid_layer_compute_checks["derived_conflict"].isVisibleTo(widget)
        assert widget.grid_layer_display_combo.findData("derived_conflict") < 0
        tabs = getattr(widget, "grid_inspection_layer_tabs", None)
        if tabs is not None:
            conflict_index = presenter._grid_inspection_layer_keys().index("derived_conflict")
            assert not tabs.isTabVisible(conflict_index)
        assert not widget._grid_reference_frame_row.isVisibleTo(widget)
        profiles = [
            button
            for button in widget.analysis_setup_panel.findChildren(QPushButton)
            if bool(button.property("analysisProfile")) and button.isVisibleTo(widget)
        ]
        assert profiles == []

    presenter._sync_mode_controls()
    hidden()
    widget.grid_inspection_content_tabs.setCurrentIndex(1)
    widget.grid_inspection_content_tabs.setCurrentIndex(0)
    hidden()
    widget._toggle_language()
    presenter._sync_mode_controls()
    hidden()
    presenter._append_folder_item(tmp_path / "model_one", checked=True)
    presenter._sync_mode_controls()
    hidden()
    payload = presenter._grid_inspection_config_payload()
    assert "derived_conflict" not in payload["requested_layers"]
    assert "confidence" in payload["requested_layers"]

    from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
    from karakal.ui.details_dialog import ExtendFrameDetailsDialog

    record = FrameRecord("frame-1", "frame_0001.jpg")
    dialog = ExtendFrameDetailsDialog(
        record,
        BuildResult(records=(record,), options=BuildOptions()),
        allowed_result_kinds=("grid_cell_defects",),
    )
    qtbot.addWidget(dialog)
    dialog.show()
    dialog._set_grid_layer_controls_visible(True)
    dialog._set_grid_detail_layer_rows(True)
    dialog.details_control_tabs.setCurrentIndex(1)
    assert not dialog.grid_confidence_layer_row.isHidden()
    dialog.details_control_tabs.setCurrentIndex(0)
    assert not dialog.grid_error_type_checks["class_conflict"].isVisibleTo(dialog)
    assert not dialog.comparison_score_card.isVisibleTo(dialog)


def test_tester_launch_switches_a_saved_other_profile(tmp_path, qtbot, monkeypatch) -> None:
    monkeypatch.delenv("KARAKAL_BUILD_PROFILE", raising=False)
    settings_path = tmp_path / "karakal.ini"
    widget = KarakalWidget(settings=QSettings(str(settings_path), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    widget._presenter._on_analysis_profile_changed(AnalysisProfileKind.SINGLE_RESULT_RISK.value)
    widget._presenter._persist_state()
    widget.close()

    monkeypatch.setenv("KARAKAL_BUILD_PROFILE", "tester")
    restored = KarakalWidget(settings=QSettings(str(settings_path), QSettings.Format.IniFormat))
    qtbot.addWidget(restored)
    assert restored._presenter._analysis_profile == AnalysisProfileKind.GRID_DEFECTS
    assert restored.app_mode_combo.currentData() == "grid_inspection"


def test_small_artifact_has_no_separate_inaccuracy_checkbox(tmp_path, qtbot) -> None:
    from karakal.core.domain import BuildOptions, BuildResult, FrameRecord
    from karakal.ui.details_dialog import ExtendFrameDetailsDialog
    from karakal.ui.i18n import set_current_language

    set_current_language("ru")
    record = FrameRecord("frame-1", "frame_0001.jpg")
    dialog = ExtendFrameDetailsDialog(
        record,
        BuildResult(records=(record,), options=BuildOptions()),
        allowed_result_kinds=("grid_cell_defects",),
    )
    qtbot.addWidget(dialog)
    dialog._set_grid_layer_controls_visible(True)
    assert not dialog.grid_suspicious_visible.isVisibleTo(dialog)
    visible_text = " ".join(
        checkbox.text()
        for checkbox in dialog.findChildren(QCheckBox)
        if checkbox.isVisibleTo(dialog)
    )
    assert "Возможная неточность" not in visible_text
    assert dialog.grid_error_type_checks["small_artifact"].toolTip().startswith("Кусок маски")
