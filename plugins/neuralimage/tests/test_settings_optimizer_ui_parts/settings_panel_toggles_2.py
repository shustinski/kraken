"""test_settings_optimizer_ui: test settings panel toggles 2."""
from __future__ import annotations
import pytest
from PyQt6.QtWidgets import QMessageBox
from neuralimage.lib.data_interfaces import WorkMode, build_synthetic_defect_generator_parameters
from neuralimage.view.settings_panel import SettingsPanel

def test_settings_panel_toggles_pcb_defect_controls(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.expert_groupbox.setChecked(True)

    panel.synthetic_defect_generator_check_box.setChecked(False)
    panel._sync_synthetic_defect_generator_controls()
    assert panel.synthetic_defect_generator_groupbox.isHidden() is False
    assert panel._field_rows[panel.pcb_defects_probability_spinbox].isEnabled() is False
    assert panel._field_rows[panel.pcb_defects_min_count_spinbox].isEnabled() is False
    assert panel._field_rows[panel.pcb_defect_type_checkboxes['break']].isEnabled() is False
    assert panel._field_rows[panel.pcb_defect_type_spinboxes['break']].isEnabled() is False

    panel.synthetic_defect_generator_check_box.setChecked(True)
    panel._sync_synthetic_defect_generator_controls()
    assert panel.synthetic_defect_generator_groupbox.isHidden() is False
    assert panel._field_rows[panel.pcb_defects_probability_spinbox].isEnabled() is True
    assert panel._field_rows[panel.pcb_defects_max_count_spinbox].isEnabled() is True
    assert panel._field_rows[panel.pcb_defect_type_checkboxes['via']].isEnabled() is True
    assert panel._field_rows[panel.pcb_defect_type_spinboxes['via']].isEnabled() is True

    panel.pcb_defect_type_checkboxes['via'].setChecked(False)
    panel._sync_synthetic_defect_generator_controls()
    assert panel._field_rows[panel.pcb_defect_type_spinboxes['via']].isEnabled() is False

def test_settings_panel_shows_ic_synthetic_defect_controls(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.expert_groupbox.setChecked(True)
    panel.synthetic_defect_generator_check_box.setChecked(True)
    panel.set_synthetic_topology_domain_value('ic')
    panel._sync_synthetic_domain_controls()
    panel._sync_synthetic_defect_generator_controls()

    assert panel._field_rows[panel.ic_defect_type_checkboxes['line_break']].isHidden() is False
    assert panel._field_rows[panel.ic_defect_type_spinboxes['line_break']].isHidden() is False
    assert panel._field_rows[panel.pcb_defect_type_checkboxes['break']].isHidden() is True
    assert panel._field_rows[panel.ic_defect_type_checkboxes['line_break']].isEnabled() is True
    assert panel._field_rows[panel.ic_defect_type_spinboxes['line_break']].isEnabled() is True

def test_settings_panel_saves_ic_synthetic_defect_toggles(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls(WorkMode.train_only.value)
    panel.synthetic_defect_generator_check_box.setChecked(True)
    panel.set_synthetic_topology_domain_value('ic')
    panel.ic_defect_type_checkboxes['bridge'].setChecked(False)
    panel.ic_defect_type_checkboxes['via_open'].setChecked(True)
    panel.ic_defect_type_spinboxes['via_open'].setValue(90)

    config = build_synthetic_defect_generator_parameters(panel.get_synthetic_defect_generator_config())

    assert config.topology_domain == 'ic'
    assert config.ic_defects.defect_probabilities['bridge'] == pytest.approx(0.0)
    assert config.ic_defects.defect_probabilities['via_open'] == pytest.approx(1.0)
    assert config.ic_defects.defect_severities['via_open'] == pytest.approx(0.9)

def test_settings_panel_emits_validation_settings_changed(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()
    calls = {'count': 0}
    panel.validation_settings_changed.connect(lambda: calls.__setitem__('count', calls['count'] + 1))

    panel.validation_check_box.setChecked(True)
    panel.validation_spinbox.setValue(25)

    assert calls['count'] >= 2

def test_settings_panel_emits_reset_defaults_requested_after_confirmation(qapp, monkeypatch):
    panel = SettingsPanel()
    panel.connect_internal_signals()
    calls = {'count': 0}
    panel.reset_defaults_requested.connect(lambda: calls.__setitem__('count', calls['count'] + 1))
    monkeypatch.setattr(
        QMessageBox,
        'question',
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    panel.reset_defaults_button.click()

    assert calls['count'] == 1

def test_settings_panel_emits_augmentation_preview_requested(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()
    panel.expert_groupbox.setChecked(True)
    calls = {'count': 0}
    panel.augmentation_preview_requested.connect(lambda: calls.__setitem__('count', calls['count'] + 1))

    panel.augmentation_preview_button.click()

    assert calls['count'] == 1
