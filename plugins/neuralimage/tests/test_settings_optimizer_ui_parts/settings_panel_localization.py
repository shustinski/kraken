"""test_settings_optimizer_ui: test settings panel localization."""
from __future__ import annotations
import pytest
from neuralimage.configuration import get_sem_preset
from neuralimage.view.settings_panel import SettingsPanel

def test_settings_panel_sem_target_toggle_marks_custom_and_syncs_head(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls('train_only')
    panel.expert_groupbox.setChecked(True)

    panel.sem_segmentation_controls['target_corner'].setChecked(True)
    payload = panel.get_sem_segmentation_config()

    assert payload['preset'] == 'custom'
    assert payload['targets']['geometry']['corner'] is True
    assert payload['heads']['enabled'] == ['corner']

def test_settings_panel_disables_sdf_boundary_weight_without_sdf(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls('train_only')
    panel.expert_groupbox.setChecked(True)
    panel.set_sem_segmentation_config(get_sem_preset('sem_topology_experimental_v1').to_dict())

    panel.sem_segmentation_controls['target_sdf'].setChecked(False)
    payload = panel.get_sem_segmentation_config()

    assert not panel.sem_segmentation_controls['target_distance_boundary_weight'].isEnabled()
    assert payload['targets']['distance_boundary_weight'] == 0.0

def test_settings_panel_emits_optimizer_settings_changed(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()

    calls = {'count': 0}
    panel.optimizer_settings_changed.connect(lambda: calls.__setitem__('count', calls['count'] + 1))

    panel.optimizer_type.setCurrentText('adamw')
    panel.mixed_precision_type.setCurrentText('off')
    panel.learning_rate_spinbox.setValue(0.0002)
    panel.weight_decay_spinbox.setValue(0.01)

    # Setting an already-selected combo value does not emit currentIndexChanged.
    assert calls['count'] >= 3

def test_settings_panel_localization_resources_cover_runtime_fallback_keys():
    import json
    from pathlib import Path

    required_keys = {
        'recognition_group',
        'save_validation_binary_images',
        'save_validation_binary_images_tip',
        'validation_source_label',
        'validation_source_split',
        'validation_source_external',
        'validation_source_tip',
        'validation_image_path_label',
        'validation_image_path_tip',
        'validation_label_path_label',
        'validation_label_path_tip',
        'scheduler_name_label',
        'scheduler_plateau_factor_tip',
        'scheduler_plateau_patience_tip',
        'scheduler_plateau_threshold_tip',
        'scheduler_plateau_min_lr_tip',
        'scheduler_plateau_cooldown_tip',
        'scheduler_cosine_t_max_tip',
        'scheduler_cosine_eta_min_tip',
        'scheduler_one_cycle_max_lr_tip',
        'scheduler_one_cycle_pct_start_tip',
        'scheduler_one_cycle_anneal_strategy_tip',
        'scheduler_one_cycle_div_factor_tip',
        'scheduler_one_cycle_final_div_factor_tip',
        'scheduler_one_cycle_three_phase_tip',
        'scheduler_step_lr_step_size_tip',
        'scheduler_step_lr_gamma_tip',
        'cutout_probability_tip',
        'cutout_holes_tip',
        'cutout_size_ratio_tip',
        'mixup_probability_tip',
        'mixup_alpha_tip',
        'pcb_defects_group',
        'pcb_defects_group_tip',
        'pcb_defects_probability_tip',
        'pcb_defects_min_count_tip',
        'pcb_defects_max_count_tip',
        'pcb_defects_use_input_mask',
        'pcb_defects_use_input_mask_tip',
        'pcb_defects_use_defect_mask_as_label',
        'pcb_defects_use_defect_mask_as_label_tip',
        'pcb_break_severity_tip',
        'pcb_short_severity_tip',
        'pcb_missing_copper_severity_tip',
        'pcb_excess_copper_severity_tip',
        'pcb_pinhole_severity_tip',
        'pcb_spurious_copper_severity_tip',
        'pcb_via_severity_tip',
        'pcb_misalignment_severity_tip',
        'loss_function',
        'sample_size_tip',
        'shift_tip',
        'augmentation_preview_button',
        'augmentation_preview_button_tip',
    }

    for path in ('resources/ui_texts_ru.json', 'resources/ui_texts_en.json'):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        settings_panel = data.get('settings_panel', {})
        missing = sorted(required_keys - set(settings_panel))
        assert not missing, f'{path} missing keys: {missing}'

def test_settings_panel_optimizer_presets_apply_values_and_highlight_active(qapp):
    panel = SettingsPanel()
    panel.connect_internal_signals()

    adam_btn, adamw_btn, adamw_muon_btn = panel.optimizer_preset_buttons
    assert len(panel.optimizer_preset_buttons) == 3
    assert panel.optimizer_advanced_content_widget.isHidden() is True
    assert panel.loss_advanced_content_widget.isHidden() is True

    adamw_muon_btn.click()
    assert panel.optimizer_type.currentText() == 'adamw_muon'
    assert panel.learning_rate_spinbox.value() == pytest.approx(0.0003)
    assert panel.weight_decay_spinbox.value() == pytest.approx(0.02)
    assert adamw_muon_btn.property('selectionRole') == 'mode'
    assert adamw_muon_btn.property('presetRole') == 'quickPreset'
    assert adamw_muon_btn.isChecked() is True
    assert adam_btn.isChecked() is False
    assert adamw_btn.isChecked() is False

    adamw_btn.click()
    assert panel.optimizer_type.currentText() == 'adamw'
    assert panel.learning_rate_spinbox.value() == pytest.approx(0.0005)
    assert panel.weight_decay_spinbox.value() == pytest.approx(0.01)
    assert adamw_btn.property('selectionRole') == 'mode'
    assert adamw_btn.property('presetRole') == 'quickPreset'
    assert adamw_btn.isChecked() is True
    assert adam_btn.isChecked() is False
    assert adamw_muon_btn.isChecked() is False

    panel.learning_rate_spinbox.setValue(0.00031)
    assert adam_btn.isChecked() is False
    assert adamw_btn.isChecked() is False
    assert adamw_muon_btn.isChecked() is False

def test_settings_panel_toggles_validation_spinbox(qapp):
    panel = SettingsPanel()
    panel.expert_groupbox.setChecked(True)

    panel.validation_check_box.setChecked(False)
    assert panel.validation_spinbox.isEnabled() is False

    panel.validation_check_box.setChecked(True)
    assert panel.validation_spinbox.isEnabled() is True

def test_settings_panel_shows_only_selected_scheduler_fields(qapp):
    panel = SettingsPanel()

    panel.set_scheduler_value('off')
    panel._sync_scheduler_controls()
    assert panel._field_rows[panel.scheduler_step_lr_step_size_spinbox].isHidden() is True
    assert panel._field_rows[panel.scheduler_one_cycle_max_lr_spinbox].isHidden() is True

    panel.set_scheduler_value('step_lr')
    panel._sync_scheduler_controls()
    assert panel._field_rows[panel.scheduler_step_lr_step_size_spinbox].isHidden() is False
    assert panel._field_rows[panel.scheduler_step_lr_gamma_spinbox].isHidden() is False
    assert panel._field_rows[panel.scheduler_plateau_factor_spinbox].isHidden() is True

    panel.set_scheduler_value('one_cycle')
    panel._sync_scheduler_controls()
    assert panel._field_rows[panel.scheduler_one_cycle_max_lr_spinbox].isHidden() is False
    assert panel._field_rows[panel.scheduler_one_cycle_three_phase_check_box].isHidden() is False
    assert panel._field_rows[panel.scheduler_step_lr_step_size_spinbox].isHidden() is True
