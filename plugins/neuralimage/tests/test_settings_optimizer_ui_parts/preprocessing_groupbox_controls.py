"""test_settings_optimizer_ui: test preprocessing groupbox controls."""
from __future__ import annotations
import importlib
import sys
import types
import pytest
from PyQt6.QtWidgets import QApplication
from neuralimage.configuration import get_sem_preset
from neuralimage.view.settings_panel import SettingsPanel

@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def _import_main_presenter_with_stubs():
    return importlib.import_module('neuralimage.presenter.main_presenter')

def test_settings_panel_round_trips_sem_topology_configuration(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls('train_only')
    panel.expert_groupbox.setChecked(True)
    payload = get_sem_preset('sem_topology_experimental_v1').to_dict()

    panel.set_sem_segmentation_config(payload)
    panel.hard_mining_check_box.setChecked(True)

    assert not hasattr(panel, 'sem_segmentation_config_editor')
    assert not hasattr(panel, 'sem_segmentation_tabs')
    assert panel.sem_segmentation_controls['target_skeleton'].isChecked()
    actual = panel.get_sem_segmentation_config()
    assert actual['targets'] == payload['targets']
    assert actual['heads'] == payload['heads']
    assert panel.configuration_status_button.text() == 'OK'

def test_settings_panel_loads_legacy_sem_preset_without_exposing_preset_ui(qapp):
    panel = SettingsPanel()
    panel.sync_business_logic_controls('train_only')
    panel.expert_groupbox.setChecked(True)
    panel.set_sem_segmentation_config(get_sem_preset('sem_topology_experimental_v1').to_dict())

    assert panel.preprocessing_groupbox.isChecked()
    assert panel.sem_segmentation_controls['pre_mode'].currentData() == 'per_image_percentile'
    assert panel.sem_segmentation_controls['aug_enabled'].isChecked()
    assert panel.sem_segmentation_controls['target_boundary'].isChecked()
    assert panel.sem_segmentation_controls['uncertainty_enabled'].isChecked()
    assert not hasattr(panel, 'sem_segmentation_preset_combo')
    assert not hasattr(panel, 'sem_segmentation_apply_preset_button')

def test_settings_panel_merges_sem_fields_into_existing_groups_with_tooltips(qapp):
    panel = SettingsPanel()

    groups = panel.sem_segmentation_section_groupboxes
    assert panel._settings_cards['preprocessing'].isAncestorOf(groups['preprocessing'])
    assert groups['preprocessing'].isCheckable()
    assert panel.training_augmentation_editor.isAncestorOf(panel.sem_segmentation_controls['aug_charging_strength'])
    assert panel.loss_advanced_form.indexOf(groups['losses']) >= 0
    assert panel.hard_mining_form.indexOf(groups['hard_mining']) >= 0
    assert panel._settings_cards['inference_uncertainty'].isAncestorOf(
        panel.sem_segmentation_controls['uncertainty_method']
    )
    assert panel._settings_cards['validation_metrics'].isAncestorOf(groups['validation'])
    assert panel._settings_cards['supervision'].isAncestorOf(groups['basic_targets'])
    assert panel._settings_cards['supervision'].isAncestorOf(groups['geometry_targets'])
    assert panel.sem_segmentation_controls['uncertainty_method'].toolTip()
    assert panel.sem_segmentation_controls['hard_exploration'].toolTip()

def test_preprocessing_groupbox_controls_normalization_mode(qapp):
    panel = SettingsPanel()
    mode = panel.sem_segmentation_controls['pre_mode']

    assert panel.preprocessing_groupbox.isCheckable()
    assert not panel.preprocessing_groupbox.isChecked()
    assert mode.currentData() == 'none'
    assert 'train' in mode.toolTip().lower()

    panel.preprocessing_groupbox.setChecked(True)
    assert mode.currentData() == 'per_image_percentile'

    dataset_index = mode.findData('dataset_zscore')
    mode.setCurrentIndex(dataset_index)
    assert panel.preprocessing_groupbox.isChecked()
    assert panel.get_sem_segmentation_config()['preprocessing']['mode'] == 'dataset_zscore'

    panel.preprocessing_groupbox.setChecked(False)
    assert mode.currentData() == 'none'
