"""test_settings_optimizer_ui: test main presenter applies."""
from __future__ import annotations
import types
import pytest
from neuralimage.lib.data_interfaces import build_synthetic_defect_generator_parameters
from neuralimage.view.settings_panel import SettingsPanel
from neuralimage.view.window_dataclasses import SettingsState

from test_settings_optimizer_ui_parts.preprocessing_groupbox_controls import _import_main_presenter_with_stubs

def test_main_presenter_applies_augmentation_preview_payload_to_settings_panel(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    from PyQt6.QtCore import QObject
    QObject.__init__(presenter)
    presenter.settings_panel = SettingsPanel()
    presenter.settings_panel.connect_internal_signals()
    presenter.settings_state = SettingsState()
    presenter.view = types.SimpleNamespace(is_batch_preview_enabled=lambda: False)

    payload = {
        'horizontal_rotation': True,
        'vertical_rotation': False,
        'flip_x': True,
        'flip_y': False,
        'random_crop': True,
        'crops_per_image': 23,
        'scale_augmentation': True,
        'scale_augmentation_strength': 0.37,
        'additional_augmentation': True,
        'augmentation_brightness_strength': 0.21,
        'augmentation_contrast_strength': 0.0,
        'augmentation_gamma_strength': 0.18,
        'augmentation_noise_probability': 0.44,
        'augmentation_noise_sigma': 0.013,
        'augmentation_blur_probability': 0.0,
        'augmentation_blur_radius': 0.0,
        'synthetic_defect_generator': {
            'enabled': True,
            'epoch_size_factor': 1.8,
            'trace_count_range': [8, 10],
            'segment_count_range': [3, 7],
            'trace_half_width_range': [2, 5],
            'background_noise_sigma_range': [0.01, 0.04],
            'trace_noise_sigma_range': [0.02, 0.05],
            'defects': {
                'enabled': True,
                'defect_probability': 0.52,
                'min_defects': 2,
                'max_defects': 5,
                'max_attempts_per_defect': 13,
                'use_input_mask': False,
                'use_defect_mask_as_label': False,
                'defect_probabilities': {
                    'break': 1.0,
                    'short': 0.0,
                    'missing_copper': 1.0,
                    'excess_copper': 0.0,
                    'pinhole': 1.0,
                    'spurious_copper': 0.0,
                    'via': 1.0,
                    'misalignment': 1.0,
                },
                'defect_severities': {
                    'break': 0.7,
                    'short': 0.0,
                    'missing_copper': 0.4,
                    'excess_copper': 0.0,
                    'pinhole': 0.6,
                    'spurious_copper': 0.0,
                    'via': 0.85,
                    'misalignment': 0.2,
                },
            },
        },
        'cutout_enabled': True,
        'cutout_probability': 0.81,
        'cutout_holes': 3,
        'cutout_size_ratio': 0.29,
        'random_artifacts_enabled': True,
        'random_artifacts_probability': 0.63,
        'random_artifacts_count': 2,
        'random_artifacts_size_ratio': 0.18,
        'random_artifacts_dust_enabled': True,
        'random_artifacts_resist_residue_enabled': False,
        'random_artifacts_etch_residue_enabled': True,
        'random_artifacts_particle_cluster_enabled': False,
        'random_artifacts_flake_enabled': True,
        'mixup_enabled': True,
        'mixup_probability': 0.58,
        'mixup_alpha': 0.47,
    }

    module.MainPresenter._apply_augmentation_preview_settings(presenter, payload)

    panel = presenter.settings_panel
    assert panel.horizontal_rotation.isChecked() is True
    assert panel.vertical_rotation.isChecked() is False
    assert panel.flip_x.isChecked() is True
    assert panel.flip_y.isChecked() is False
    assert panel.random_crop_check_box.isChecked() is True
    assert panel.crops_per_image_spinbox.value() == 23
    assert panel.scale_augmentation_check_box.isChecked() is True
    assert panel.scale_augmentation_strength_spinbox.value() == pytest.approx(0.37)
    assert panel.additional_augmentation_check_box.isChecked() is True
    assert panel.augmentation_brightness_spinbox.value() == pytest.approx(0.21)
    assert panel.augmentation_contrast_spinbox.value() == pytest.approx(0.0)
    assert panel.augmentation_gamma_spinbox.value() == pytest.approx(0.18)
    assert panel.augmentation_noise_probability_spinbox.value() == pytest.approx(0.44)
    assert panel.augmentation_noise_sigma_spinbox.value() == pytest.approx(0.013)
    assert panel.augmentation_blur_probability_spinbox.value() == pytest.approx(0.0)
    assert panel.cutout_check_box.isChecked() is True
    assert panel.cutout_holes_spinbox.value() == 3
    assert panel.random_artifacts_check_box.isChecked() is True
    assert panel.random_artifact_type_checkboxes['resist_residue'].isChecked() is False
    assert panel.random_artifact_type_checkboxes['flake'].isChecked() is True
    assert panel.mixup_check_box.isChecked() is True
    assert panel.mixup_alpha_spinbox.value() == pytest.approx(0.47)
    assert panel.synthetic_defect_generator_check_box.isChecked() is True
    assert panel.synthetic_dataset_factor_spinbox.value() == pytest.approx(1.8)
    assert panel.synthetic_trace_count_min_spinbox.value() == 8
    assert panel.synthetic_trace_count_max_spinbox.value() == 10
    assert panel.synthetic_segment_count_min_spinbox.value() == 3
    assert panel.synthetic_segment_count_max_spinbox.value() == 7
    assert panel.synthetic_trace_half_width_min_spinbox.value() == 2
    assert panel.synthetic_trace_half_width_max_spinbox.value() == 5
    assert panel.pcb_defects_probability_spinbox.value() == pytest.approx(0.52)
    assert panel.pcb_defect_type_checkboxes['short'].isChecked() is False
    assert panel.pcb_defect_type_spinboxes['short'].value() == 0
    assert panel.pcb_defect_type_checkboxes['via'].isChecked() is True
    assert panel.pcb_defect_type_spinboxes['via'].value() == 85

    synthetic_generator = build_synthetic_defect_generator_parameters(panel.get_synthetic_defect_generator_config())
    assert synthetic_generator.enabled is True
    assert synthetic_generator.epoch_size_factor == pytest.approx(1.8)
    assert synthetic_generator.trace_count_range == (8, 10)
    assert synthetic_generator.segment_count_range == (3, 7)
    assert synthetic_generator.trace_half_width_range == (2, 5)
    assert synthetic_generator.defects.defect_probability == pytest.approx(0.52)
    assert synthetic_generator.defects.max_attempts_per_defect == 13
    assert synthetic_generator.defects.use_input_mask is True
    assert synthetic_generator.defects.defect_probabilities['break'] == pytest.approx(1.0)
    assert synthetic_generator.defects.defect_probabilities['short'] == pytest.approx(0.0)
    assert synthetic_generator.defects.defect_severities['break'] == pytest.approx(0.7)
    assert synthetic_generator.defects.defect_severities['via'] == pytest.approx(0.85)

    assert presenter.settings_state.random_crop is True
    assert presenter.settings_state.flip_x is True
    assert presenter.settings_state.flip_y is False
    assert presenter.settings_state.crops_per_image == 23
    assert presenter.settings_state.scale_augmentation is True
    assert presenter.settings_state.additional_augmentation is True
    state_synthetic_generator = build_synthetic_defect_generator_parameters(
        presenter.settings_state.synthetic_defect_generator
    )
    assert state_synthetic_generator.defects.defect_probabilities['via'] == pytest.approx(1.0)
    assert state_synthetic_generator.defects.defect_severities['via'] == pytest.approx(0.85)

def test_main_presenter_updates_main_view_sample_count_when_calculated(qapp):
    module = _import_main_presenter_with_stubs()
    presenter = module.MainPresenter.__new__(module.MainPresenter)
    from PyQt6.QtCore import QObject
    QObject.__init__(presenter)
    presenter._sample_count_worker_thread = None
    presenter._sample_count_cache_path = None
    presenter._sample_count_cache_sizes = None
    presenter._latest_sample_count_request_id = 7
    presenter._start_pending_sample_count_request_if_needed = lambda: None
    presenter.settings_panel = types.SimpleNamespace(set_samples_count=lambda value: setattr(presenter, '_panel_count', value))
    presenter.view = types.SimpleNamespace(set_samples_count=lambda value: setattr(presenter, '_view_count', value))

    module.MainPresenter._on_sample_count_calculated(presenter, 7, 'D:/sample', [], 123)

    assert presenter._panel_count == 123
    assert presenter._view_count == 123
