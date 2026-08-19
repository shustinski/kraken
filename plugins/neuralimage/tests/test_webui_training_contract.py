from __future__ import annotations

import json
import os

import pytest

django = pytest.importorskip('django')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'neuralimage.webui_project.settings')
django.setup()

from neuralimage.application.dto import SettingsState
from neuralimage.configuration import get_sem_preset, sem_config_to_form_values
from neuralimage.webui.forms import SettingsForm, defaults_from_settings_state


def _valid_settings_data(**updates: str) -> dict[str, str]:
    data = {
        'step': '64',
        'sample_x': '256',
        'sample_y': '256',
        'model': 'M 720k',
        'color_mode': 'RGB',
        'sample_cut_mode': 'online',
        'batch_size': '4',
        'overlap': '16',
        'log_update_frequency': '0',
        'optimizer_name': 'adam',
        'mixed_precision': 'off',
        'loss_function': 'bce',
        'learning_rate': '0.001',
        'weight_decay': '0.0',
        'warmup_epochs': '3',
        'warmup_start_factor': '0.1',
    }
    data.update(updates)
    return data


def test_web_form_exposes_only_automatic_early_stopping_toggle():
    fields = SettingsForm.base_fields
    assert 'early_stopping_enabled' in fields
    assert 'early_stopping_patience' not in fields
    assert 'early_stopping_min_delta' not in fields
    assert 'early_stopping_restore_best_weights' not in fields
    assert 'hard_mining_strength' not in fields
    assert 'hard_mining_ema_alpha' not in fields


def test_web_random_patch_size_maps_independent_online_ranges():
    form = SettingsForm(data=_valid_settings_data(
        random_patch_size_enabled='on',
        random_patch_min_x='65',
        random_patch_min_y='97',
        random_patch_max_x='160',
        random_patch_max_y='224',
    ))
    assert form.is_valid(), form.errors

    state = form.to_state()

    assert state.random_patch_size_enabled
    assert state.random_patch_min_size == (65, 97)
    assert state.random_patch_max_size == (160, 224)


def test_new_runtime_defaults_are_off_in_web_contract():
    initial = defaults_from_settings_state(SettingsState())
    assert initial['mixed_precision'] == 'off'
    assert initial['use_multi_gpu'] is False
    assert initial['torch_compile_enabled'] is False


def test_web_synthetic_topology_maps_only_when_enabled():
    form = SettingsForm(data=_valid_settings_data(
        synthetic_defect_generator_enabled='on',
        synthetic_topology_domain='ic',
        synthetic_epoch_size_factor='0.5',
        synthetic_image_width='640',
        synthetic_image_height='384',
    ))
    assert form.is_valid(), form.errors

    payload = form.to_state().synthetic_defect_generator

    assert payload['enabled'] is True
    assert payload['topology_domain'] == 'ic'
    assert payload['epoch_size_factor'] == pytest.approx(0.5)
    assert payload['image_size_xy'] == [640, 384]


def test_web_sem_configuration_uses_typed_fields_and_round_trips_preset():
    assert 'sem_segmentation_config' not in SettingsForm.base_fields
    assert 'sem__target_skeleton' in SettingsForm.base_fields
    preset = get_sem_preset('sem_topology_experimental_v1').to_dict()
    flattened = sem_config_to_form_values(preset)
    posted = _valid_settings_data()
    for name, value in flattened.items():
        if isinstance(value, bool):
            if value:
                posted[name] = 'on'
        else:
            posted[name] = str(value)

    form = SettingsForm(data=posted)

    assert form.is_valid(), form.errors
    assert form.to_state().sem_segmentation_config == preset


def test_web_form_accepts_legacy_json_post_without_rendering_json_field():
    preset = get_sem_preset('sem_topology_experimental_v1').to_dict()
    form = SettingsForm(
        data=_valid_settings_data(sem_segmentation_config=json.dumps(preset)),
    )

    assert form.is_valid(), form.errors
    assert form.to_state().sem_segmentation_config == preset


def test_web_form_reports_invalid_legacy_json_post():
    form = SettingsForm(
        data=_valid_settings_data(sem_segmentation_config='{invalid'),
    )

    assert not form.is_valid()
    assert 'Invalid legacy SEM configuration' in str(form.errors['sem_preset'])
