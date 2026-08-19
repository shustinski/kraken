import json

import pytest

from neuralimage.configuration import (
    available_sem_presets,
    build_sem_segmentation_config,
    get_sem_preset,
    sem_config_from_form_values,
    sem_config_to_form_values,
)
from neuralimage.main_code_version import _load_settings


def test_legacy_preset_keeps_quality_affecting_features_disabled():
    config = get_sem_preset('legacy_v1')
    assert not config.preprocessing.any_enabled()
    assert not config.augmentation.enabled
    assert not config.targets.any_enabled()
    assert config.heads.enabled == ()


def test_experimental_preset_round_trips_and_has_stable_hash():
    config = get_sem_preset('sem_topology_experimental_v1')
    restored = build_sem_segmentation_config(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert restored.stable_hash() == config.stable_hash()
    assert set(config.heads.enabled) == {'boundary', 'skeleton', 'sdf'}
    assert config.experiment.topology_first


def test_typed_ui_values_round_trip_and_derive_matching_heads():
    preset = get_sem_preset('sem_topology_experimental_v1').to_dict()
    values = sem_config_to_form_values(preset)

    restored = sem_config_from_form_values(values, preset='sem_topology_experimental_v1')

    assert restored == preset
    assert restored['heads']['enabled'] == ['boundary', 'skeleton', 'sdf']


def test_typed_legacy_defaults_keep_compact_compatibility_payload():
    values = sem_config_to_form_values({})

    assert sem_config_from_form_values(values, preset='legacy_v1') == {}


def test_typed_ui_clears_distance_boundary_weight_when_sdf_is_disabled():
    values = sem_config_to_form_values(get_sem_preset('sem_topology_experimental_v1').to_dict())
    values['sem__target_sdf'] = False

    restored = sem_config_from_form_values(values, preset='custom')

    assert restored['targets']['basic']['sdf'] is False
    assert restored['targets']['distance_boundary_weight'] == 0.0
    assert 'sdf' not in restored['heads']['enabled']


def test_cross_field_validation_rejects_missing_sdf_head():
    with pytest.raises(ValueError, match='Heads and targets'):
        build_sem_segmentation_config(
            {
                'targets': {'basic': {'sdf': True}},
                'heads': {'enabled': []},
            }
        )


@pytest.mark.parametrize(
    'payload, message',
    [
        ({'hard_mining': {'refresh_epochs': 0}}, 'refresh_epochs'),
        ({'validation': {'confidence_bins': 1}}, 'confidence_bins'),
        ({'preprocessing': {'denoise_strength': -1}}, 'denoise_strength'),
    ],
)
def test_strict_validation_rejects_invalid_ranges(payload, message):
    with pytest.raises(ValueError, match=message):
        build_sem_segmentation_config(payload)


def test_recommended_preset_is_gated_until_ablation():
    assert 'sem_topology_recommended_v1' not in available_sem_presets()
    with pytest.raises(ValueError, match='three-seed'):
        get_sem_preset('sem_topology_recommended_v1')


@pytest.mark.parametrize('training_key,recognition_key', [
    ('training_parameters', 'recognition_parameters'),
    ('tranining_parameters', 'recogniton_parameters'),
])
def test_cli_accepts_correct_keys_and_legacy_aliases(tmp_path, training_key, recognition_key):
    config = get_sem_preset('sem_topology_experimental_v1')
    config_path = tmp_path / 'config.json'
    config_path.write_text(
        json.dumps({
            'work_mode': 'train_only',
            training_key: {'sem_segmentation_config': config.to_dict()},
            recognition_key: {},
        }),
        encoding='utf-8',
    )

    _mode, training, recognition = _load_settings(config_path, None)

    assert training.supervision_targets.basic.skeleton
    assert recognition.sem_config_hash == config.stable_hash()
