import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication

from neuralimage.infrastructure.config.state_store import load_settings_state, save_settings_state
from tests.helpers import make_test_dir
from neuralimage.lib.data_interfaces import build_synthetic_defect_generator_parameters
from neuralimage.view.settings_panel import SettingsPanel
from neuralimage.view.window_dataclasses import SettingsState


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_settings_panel_roundtrips_synthetic_defect_generator_config(qapp):
    panel = SettingsPanel()
    config = {
        'enabled': True,
        'epoch_size_factor': 2.5,
        'image_size_xy': [1024, 768],
        'trace_count_range': [6, 8],
        'segment_count_range': [4, 6],
        'trace_half_width_range': [2, 3],
        'background_noise_sigma_range': [0.02, 0.04],
        'trace_noise_sigma_range': [0.01, 0.03],
        'defects': {
            'enabled': True,
            'defect_probability': 0.85,
            'min_defects': 2,
            'max_defects': 5,
            'defect_probabilities': {
                'break': 1.0,
                'short': 0.0,
                'via': 1.0,
            },
            'defect_severities': {
                'break': 0.72,
                'short': 0.0,
                'via': 0.9,
            },
        },
    }

    panel.set_synthetic_defect_generator_config(config)
    restored = panel.get_synthetic_defect_generator_config()
    resolved = build_synthetic_defect_generator_parameters(restored)

    assert panel.synthetic_defect_generator_check_box.isChecked() is True
    assert panel.synthetic_dataset_factor_spinbox.value() == pytest.approx(2.5)
    assert panel.synthetic_image_width_spinbox.value() == 1024
    assert panel.synthetic_image_height_spinbox.value() == 768
    assert panel.synthetic_trace_count_min_spinbox.value() == 6
    assert panel.synthetic_trace_count_max_spinbox.value() == 8
    assert panel.synthetic_segment_count_min_spinbox.value() == 4
    assert panel.synthetic_segment_count_max_spinbox.value() == 6
    assert panel.synthetic_trace_half_width_min_spinbox.value() == 2
    assert panel.synthetic_trace_half_width_max_spinbox.value() == 3
    assert restored['enabled'] is True
    assert restored['epoch_size_factor'] == pytest.approx(2.5)
    assert restored['image_size_xy'] == [1024, 768]
    assert restored['trace_count_range'] == [6, 8]
    assert restored['segment_count_range'] == [4, 6]
    assert restored['trace_half_width_range'] == [2, 3]
    assert restored['background_noise_sigma_range'] == pytest.approx([0.02, 0.04])
    assert restored['trace_noise_sigma_range'] == pytest.approx([0.01, 0.03])
    assert resolved.defects.defect_probability == pytest.approx(0.85)
    assert resolved.defects.min_defects == 2
    assert resolved.defects.max_defects == 5
    assert resolved.defects.defect_probabilities['break'] == pytest.approx(1.0)
    assert resolved.defects.defect_probabilities['short'] == pytest.approx(0.0)
    assert resolved.defects.defect_probabilities['via'] == pytest.approx(1.0)
    assert resolved.defects.defect_severities['break'] == pytest.approx(0.72)
    assert resolved.defects.defect_severities['short'] == pytest.approx(0.0)
    assert resolved.defects.defect_severities['via'] == pytest.approx(0.9)


def test_state_store_roundtrips_synthetic_defect_generator(monkeypatch):
    settings_dir = make_test_dir('state_store_synthetic_defect_generator')
    monkeypatch.setenv('NEURALIMAGE_SETTINGS_DIR', str(settings_dir))
    state = SettingsState(
        synthetic_defect_generator={
            'enabled': True,
            'epoch_size_factor': 1.75,
            'image_size_xy': [1536, 1024],
            'trace_count_range': [7, 9],
            'segment_count_range': [4, 5],
            'trace_half_width_range': [3, 4],
            'background_noise_sigma_range': [0.03, 0.05],
            'trace_noise_sigma_range': [0.01, 0.02],
            'defects': {
                'enabled': True,
                'defect_probability': 0.9,
                'min_defects': 1,
                'max_defects': 3,
                'defect_probabilities': {
                    'break': 1.0,
                    'short': 0.0,
                },
                'defect_severities': {
                    'break': 0.6,
                    'short': 0.25,
                },
            },
        }
    )

    save_settings_state(state)
    loaded = load_settings_state()

    assert loaded.synthetic_defect_generator['enabled'] is True
    assert loaded.synthetic_defect_generator['epoch_size_factor'] == pytest.approx(1.75)
    assert loaded.synthetic_defect_generator['image_size_xy'] == [1536, 1024]
    assert loaded.synthetic_defect_generator['trace_count_range'] == [7, 9]
    assert loaded.synthetic_defect_generator['segment_count_range'] == [4, 5]
    assert loaded.synthetic_defect_generator['trace_half_width_range'] == [3, 4]
    assert loaded.synthetic_defect_generator['background_noise_sigma_range'] == pytest.approx([0.03, 0.05])
    assert loaded.synthetic_defect_generator['trace_noise_sigma_range'] == pytest.approx([0.01, 0.02])
    assert loaded.synthetic_defect_generator['defects']['defect_probability'] == pytest.approx(0.9)
    assert loaded.synthetic_defect_generator['defects']['defect_probabilities']['short'] == pytest.approx(0.0)
    assert loaded.synthetic_defect_generator['defects']['defect_severities']['short'] == pytest.approx(0.25)


def test_settings_panel_roundtrips_ic_synthetic_generator_config(qapp):
    panel = SettingsPanel()
    config = {
        'enabled': True,
        'topology_domain': 'ic',
        'topology_family': 'ic_cell_array',
        'ic_topology_family': 'ic_cell_array',
        'trace_count_range': [10, 12],
        'segment_count_range': [2, 4],
        'trace_half_width_range': [1, 2],
        'ic_defects': {
            'enabled': True,
            'defect_probability': 0.7,
            'min_defects': 1,
            'max_defects': 2,
            'defect_probabilities': {
                'line_break': 1.0,
                'bridge': 0.0,
                'via_open': 1.0,
            },
            'defect_severities': {
                'line_break': 0.65,
                'bridge': 0.0,
                'via_open': 0.85,
            },
        },
    }

    panel.set_synthetic_defect_generator_config(config)
    restored = panel.get_synthetic_defect_generator_config()
    resolved = build_synthetic_defect_generator_parameters(restored)

    assert panel.get_synthetic_topology_domain_value() == 'ic'
    assert resolved.topology_domain == 'ic'
    assert resolved.topology_family == 'ic_cell_array'
    assert resolved.ic_defects.defect_severities['line_break'] == pytest.approx(0.65)
    assert resolved.ic_defects.defect_severities['via_open'] == pytest.approx(0.85)
    assert restored['ic_topology_family'] == 'ic_cell_array'
