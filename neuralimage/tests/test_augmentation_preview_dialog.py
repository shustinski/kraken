import numpy as np
import pytest

pytest.importorskip('PyQt6')

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from lib.data_interfaces import (
    PCBDefectParameters,
    SampleCutMode,
    SampleGenerationSettings,
    SamplePrepareSettings,
    TrainingParameters,
)
from tests.helpers import make_test_dir
from view.augmentation_preview_dialog import AugmentationPreviewDialog
import view.augmentation_preview_dialog as augmentation_preview_module


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_training_parameters(sample_dir, label_dir) -> TrainingParameters:
    generation = SampleGenerationSettings(
        step=16,
        segment_size=(32, 32),
        vertical_rotation=False,
        horizontal_rotation=False,
        channels=1,
    )
    prepare = SamplePrepareSettings()
    return TrainingParameters(
        image_path=sample_dir,
        label_path=label_dir,
        shuffle=False,
        validation=False,
        validation_percent=0,
        batch_size=1,
        cut_mode=SampleCutMode.online,
        colors=1,
        epochs=1,
        generation=generation,
        prepare=prepare,
    )


def test_augmentation_preview_dialog_middle_button_shows_original_only_while_held(qapp):
    root = make_test_dir('augmentation_preview_dialog_middle_click')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((48, 48), dtype=np.uint8)
    image[6:18, 8:14] = 255
    image[24:36, 20:40] = 180
    label = np.zeros((48, 48), dtype=np.uint8)
    label[10:34, 12:30] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label, mode='L').save(label_dir / 'frame_a.png')

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    assert dialog._show_augmented is True

    QTest.mousePress(dialog.image_preview, Qt.MouseButton.MiddleButton)
    qapp.processEvents()
    assert dialog._show_augmented is False

    QTest.mouseRelease(dialog.image_preview, Qt.MouseButton.MiddleButton)
    qapp.processEvents()

    assert dialog._show_augmented is True
    dialog.close()


def test_augmentation_preview_dialog_updates_preview_when_toggle_changes(qapp):
    root = make_test_dir('augmentation_preview_dialog_toggle')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image_a = np.zeros((48, 48), dtype=np.uint8)
    image_a[4:28, 6:14] = 255
    image_a[30:40, 28:44] = 160
    label_a = np.zeros((48, 48), dtype=np.uint8)
    label_a[8:34, 10:20] = 255
    image_b = np.zeros((48, 48), dtype=np.uint8)
    image_b[12:26, 26:42] = 255
    image_b[30:42, 6:18] = 200
    label_b = np.zeros((48, 48), dtype=np.uint8)
    label_b[16:40, 22:36] = 255
    Image.fromarray(image_a, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label_a, mode='L').save(label_dir / 'frame_a.png')
    Image.fromarray(image_b, mode='L').save(sample_dir / 'frame_b.png')
    Image.fromarray(label_b, mode='L').save(label_dir / 'frame_b.png')

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    before = dialog._augmented_image_array.copy()
    dialog._toggle_boxes['rotate_90'].setChecked(True)
    qapp.processEvents()

    assert dialog._augmented_image_array is not None
    assert not np.array_equal(before, dialog._augmented_image_array)
    dialog.close()


def test_augmentation_preview_dialog_uses_runtime_spinbox_values(qapp):
    root = make_test_dir('augmentation_preview_dialog_spinboxes')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.full((48, 48), 96, dtype=np.uint8)
    image[10:28, 12:36] = 164
    label = np.zeros((48, 48), dtype=np.uint8)
    label[14:34, 16:32] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label, mode='L').save(label_dir / 'frame_a.png')

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    assert dialog.crops_per_image_spinbox.value() == 64
    assert dialog.cutout_holes_spinbox.value() == 1
    assert dialog.pcb_defects_max_count_spinbox.value() == 3

    before = dialog._augmented_image_array.copy()
    dialog._toggle_boxes['brightness'].setChecked(True)
    dialog.augmentation_brightness_spinbox.setValue(1.0)
    qapp.processEvents()

    assert dialog._augmented_image_array is not None
    assert not np.array_equal(before, dialog._augmented_image_array)
    dialog.close()


def test_augmentation_preview_dialog_selects_sample_from_left_list(qapp):
    root = make_test_dir('augmentation_preview_dialog_sample_list')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image_a = np.zeros((48, 48), dtype=np.uint8)
    image_a[4:20, 6:18] = 255
    label_a = np.zeros((48, 48), dtype=np.uint8)
    label_a[8:22, 10:24] = 255
    image_b = np.zeros((48, 48), dtype=np.uint8)
    image_b[24:42, 28:44] = 200
    label_b = np.zeros((48, 48), dtype=np.uint8)
    label_b[20:40, 26:42] = 255
    Image.fromarray(image_a, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label_a, mode='L').save(label_dir / 'frame_a.png')
    Image.fromarray(image_b, mode='L').save(sample_dir / 'frame_b.png')
    Image.fromarray(label_b, mode='L').save(label_dir / 'frame_b.png')

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    assert dialog._current_sample_index == 0
    assert dialog.sample_list_widget.count() == 2

    second_item = dialog.sample_list_widget.item(1)
    second_rect = dialog.sample_list_widget.visualItemRect(second_item)
    QTest.mouseClick(
        dialog.sample_list_widget.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        second_rect.center(),
    )
    qapp.processEvents()

    assert dialog._current_sample_index == 1
    assert dialog.sample_list_widget.currentRow() == 1
    assert 'frame_b.png' in dialog.sample_label.text()
    dialog.close()


def test_augmentation_preview_dialog_generates_synthetic_pair_without_dataset(qapp):
    root = make_test_dir('augmentation_preview_dialog_synthetic_topology')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    dialog._toggle_boxes['synthetic_topology'].setChecked(True)
    qapp.processEvents()

    assert dialog._augmented_image_array is not None
    assert dialog._augmented_label_array is not None
    assert dialog._augmented_image_array.ndim == 3
    assert 3 in dialog._augmented_image_array.shape
    assert int(np.count_nonzero(dialog._augmented_label_array)) > 0
    assert dialog.prev_button.isEnabled() is False
    assert dialog.next_button.isEnabled() is False
    dialog.close()


def test_augmentation_preview_dialog_resamples_synthetic_topology(qapp):
    root = make_test_dir('augmentation_preview_dialog_synthetic_topology_resample')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    dialog._toggle_boxes['synthetic_topology'].setChecked(True)
    qapp.processEvents()
    before_label = dialog._augmented_label_array.copy()

    dialog.resample_button.click()
    qapp.processEvents()

    assert not np.array_equal(before_label, dialog._augmented_label_array)
    dialog.close()


def test_augmentation_preview_dialog_creates_separate_labeled_rows_for_noise_controls(qapp):
    root = make_test_dir('augmentation_preview_dialog_noise_rows')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((32, 32), dtype=np.uint8)
    label = np.zeros((32, 32), dtype=np.uint8)
    label[8:20, 10:24] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label, mode='L').save(label_dir / 'frame_a.png')

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    noise_rows = dialog._value_rows['noise']
    noise_row_texts = []
    for row in noise_rows:
        labels = row.findChildren(type(dialog.sample_label))
        noise_row_texts.extend(label.text() for label in labels if label.text())

    assert len(noise_rows) == 2
    assert any('Noise probability' in text or 'Вероятность шума' in text for text in noise_row_texts)
    assert any('Noise strength' in text or 'Сигма шума' in text or 'Сила шума' in text for text in noise_row_texts)
    dialog.close()


def test_augmentation_preview_dialog_uses_only_matching_pairs_without_missing_file_error(qapp):
    root = make_test_dir('augmentation_preview_dialog_matching_pairs')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image_a = np.zeros((32, 32), dtype=np.uint8)
    image_a[6:18, 8:22] = 180
    image_b = np.zeros((32, 32), dtype=np.uint8)
    image_b[10:24, 12:28] = 220
    label_a = np.zeros((32, 32), dtype=np.uint8)
    label_a[8:20, 10:24] = 255
    label_extra = np.zeros((32, 32), dtype=np.uint8)
    label_extra[4:12, 4:16] = 255

    Image.fromarray(image_a, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(image_b, mode='L').save(sample_dir / 'frame_b.png')
    Image.fromarray(label_a, mode='L').save(label_dir / 'frame_a.png')
    Image.fromarray(label_extra, mode='L').save(label_dir / 'frame_extra.png')

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    assert len(dialog._sample_pairs) == 1
    assert dialog._sample_pairs[0][0].stem == 'frame_a'
    assert dialog._sample_pairs[0][1].stem == 'frame_a'
    assert dialog._load_error is None
    assert 'mismatch' not in dialog.sample_label.text().lower()
    dialog.close()


def test_augmentation_preview_dialog_renders_image_and_label_with_same_display_size(qapp):
    root = make_test_dir('augmentation_preview_dialog_equal_render_size')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((48, 48), dtype=np.uint8)
    image[8:34, 12:30] = 200
    label = np.zeros((48, 48), dtype=np.uint8)
    label[10:28, 14:26] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label, mode='L').save(label_dir / 'frame_a.png')

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    image_pixmap = dialog.image_preview.pixmap()
    label_pixmap = dialog.label_preview.pixmap()

    assert image_pixmap is not None
    assert label_pixmap is not None
    assert image_pixmap.size() == label_pixmap.size()
    dialog.close()


def test_augmentation_preview_dialog_can_switch_to_full_image_preview(qapp):
    root = make_test_dir('augmentation_preview_dialog_full_image')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((48, 64), dtype=np.uint8)
    image[8:34, 12:44] = 200
    label = np.zeros((48, 64), dtype=np.uint8)
    label[10:30, 14:42] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label, mode='L').save(label_dir / 'frame_a.png')

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    assert dialog._augmented_image_array.shape == (32, 32)

    dialog.full_image_check_box.setChecked(True)
    qapp.processEvents()

    assert dialog._augmented_image_array.shape == (48, 64)
    assert dialog._augmented_label_array.shape == (48, 64)
    dialog.close()


def test_augmentation_preview_dialog_builds_apply_payload_from_current_preview_controls(qapp):
    root = make_test_dir('augmentation_preview_dialog_apply_payload')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((48, 48), dtype=np.uint8)
    image[8:34, 12:30] = 200
    label = np.zeros((48, 48), dtype=np.uint8)
    label[10:28, 14:26] = 255
    Image.fromarray(image, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label, mode='L').save(label_dir / 'frame_a.png')

    training_parameters = _build_training_parameters(sample_dir, label_dir)
    training_parameters.pcb_defects = PCBDefectParameters(max_attempts_per_defect=11, use_input_mask=False)
    training_parameters.synthetic_defect_generator = {
        'enabled': True,
        'image_size_xy': [896, 640],
        'trace_count_range': [4, 6],
        'segment_count_range': [3, 5],
        'trace_half_width_range': [2, 3],
        'background_noise_sigma_range': [0.05, 0.07],
        'trace_noise_sigma_range': [0.01, 0.02],
        'defects': {
            'enabled': True,
            'defect_probability': 0.45,
            'min_defects': 2,
            'max_defects': 4,
            'defect_probabilities': {
                'break': 1.25,
                'short': 0.5,
                'via': 0.0,
            },
        },
    }

    dialog = AugmentationPreviewDialog(training_parameters)
    dialog.show()
    qapp.processEvents()

    assert dialog.synthetic_background_noise_sigma_min_spinbox.value() == pytest.approx(0.05)
    assert dialog.synthetic_background_noise_sigma_max_spinbox.value() == pytest.approx(0.07)
    assert dialog.synthetic_image_width_spinbox.value() == 896
    assert dialog.synthetic_image_height_spinbox.value() == 640

    dialog._toggle_boxes['rotate_90'].setChecked(True)
    dialog._toggle_boxes['flip_x'].setChecked(True)
    dialog._toggle_boxes['random_crop'].setChecked(True)
    dialog.crops_per_image_spinbox.setValue(17)
    dialog._toggle_boxes['brightness'].setChecked(True)
    dialog.augmentation_brightness_spinbox.setValue(0.33)
    dialog._toggle_boxes['noise'].setChecked(True)
    dialog.augmentation_noise_probability_spinbox.setValue(0.42)
    dialog.augmentation_noise_sigma_spinbox.setValue(0.015)
    dialog._toggle_boxes['synthetic_topology'].setChecked(True)
    dialog.synthetic_image_width_spinbox.setValue(1024)
    dialog.synthetic_image_height_spinbox.setValue(768)
    dialog.synthetic_trace_count_min_spinbox.setValue(6)
    dialog.synthetic_trace_count_max_spinbox.setValue(7)
    dialog.synthetic_segment_count_min_spinbox.setValue(5)
    dialog.synthetic_segment_count_max_spinbox.setValue(6)
    dialog.synthetic_trace_half_width_min_spinbox.setValue(3)
    dialog.synthetic_trace_half_width_max_spinbox.setValue(4)
    dialog.synthetic_background_noise_sigma_min_spinbox.setValue(0.08)
    dialog.synthetic_background_noise_sigma_max_spinbox.setValue(0.091)
    dialog.synthetic_trace_noise_sigma_min_spinbox.setValue(0.02)
    dialog.synthetic_trace_noise_sigma_max_spinbox.setValue(0.035)
    dialog._toggle_boxes['random_artifacts'].setChecked(True)
    dialog.random_artifacts_probability_spinbox.setValue(0.66)
    dialog._toggle_boxes['artifact_flake'].setChecked(False)
    dialog._toggle_boxes['pcb_defects'].setChecked(True)
    dialog.pcb_defects_probability_spinbox.setValue(0.55)
    dialog._toggle_boxes['pcb_break'].setChecked(True)
    dialog.pcb_defect_type_spinboxes['break'].setValue(80)
    dialog._toggle_boxes['pcb_via'].setChecked(False)
    qapp.processEvents()

    payload = dialog._build_apply_payload()

    assert payload['horizontal_rotation'] is True
    assert payload['flip_x'] is True
    assert payload['flip_y'] is False
    assert payload['random_crop'] is True
    assert payload['crops_per_image'] == 17
    assert payload['additional_augmentation'] is True
    assert payload['augmentation_brightness_strength'] == pytest.approx(0.33)
    assert payload['augmentation_contrast_strength'] == pytest.approx(0.0)
    assert payload['augmentation_noise_probability'] == pytest.approx(0.42)
    assert payload['augmentation_noise_sigma'] == pytest.approx(0.015)
    assert payload['synthetic_defect_generator'].enabled is True
    assert payload['synthetic_defect_generator'].image_size_xy == (1024, 768)
    assert payload['synthetic_defect_generator'].trace_count_range == (6, 7)
    assert payload['synthetic_defect_generator'].segment_count_range == (5, 6)
    assert payload['synthetic_defect_generator'].trace_half_width_range == (3, 4)
    assert payload['synthetic_defect_generator'].background_noise_sigma_range == pytest.approx((0.08, 0.091))
    assert payload['synthetic_defect_generator'].trace_noise_sigma_range == pytest.approx((0.02, 0.035))
    assert payload['synthetic_defect_generator'].defects.enabled is True
    assert payload['synthetic_defect_generator'].defects.defect_probability == pytest.approx(0.55)
    assert payload['synthetic_defect_generator'].defects.defect_probabilities['break'] == pytest.approx(1.0)
    assert payload['synthetic_defect_generator'].defects.defect_probabilities['via'] == pytest.approx(0.0)
    assert payload['synthetic_defect_generator'].defects.defect_severities['break'] == pytest.approx(0.8)
    assert payload['random_artifacts_enabled'] is True
    assert payload['random_artifacts_probability'] == pytest.approx(0.66)
    assert payload['random_artifacts_flake_enabled'] is False
    assert payload['tech_aug'].enabled is False
    assert payload['pcb_defects'].enabled is False
    dialog.close()


def test_augmentation_preview_dialog_builds_ic_domain_payload(qapp):
    root = make_test_dir('augmentation_preview_dialog_ic_payload')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    training_parameters = _build_training_parameters(sample_dir, label_dir)
    training_parameters.synthetic_defect_generator = {
        'enabled': True,
        'topology_domain': 'ic',
        'topology_family': 'ic_cell_array',
        'ic_defects': {
            'enabled': True,
            'defect_probability': 0.6,
            'min_defects': 1,
            'max_defects': 2,
            'defect_probabilities': {
                'line_break': 1.0,
                'bridge': 0.0,
                'via_open': 1.0,
            },
            'defect_severities': {
                'line_break': 0.7,
                'bridge': 0.0,
                'via_open': 0.9,
            },
        },
    }

    dialog = AugmentationPreviewDialog(training_parameters)
    dialog.show()
    qapp.processEvents()

    dialog._toggle_boxes['synthetic_topology'].setChecked(True)
    dialog.synthetic_topology_domain_combo.setCurrentIndex(1)
    dialog._toggle_boxes['pcb_defects'].setChecked(True)
    dialog._toggle_boxes['ic_line_break'].setChecked(True)
    dialog._toggle_boxes['ic_bridge'].setChecked(False)
    dialog._toggle_boxes['ic_via_open'].setChecked(True)
    dialog.ic_defect_type_spinboxes['line_break'].setValue(77)
    dialog.ic_defect_type_spinboxes['via_open'].setValue(88)
    qapp.processEvents()

    payload = dialog._build_apply_payload()

    assert payload['synthetic_defect_generator'].topology_domain == 'ic'
    assert payload['synthetic_defect_generator'].defects.defect_probabilities['line_break'] == pytest.approx(1.0)
    assert payload['synthetic_defect_generator'].defects.defect_probabilities['bridge'] == pytest.approx(0.0)
    assert payload['synthetic_defect_generator'].defects.defect_severities['line_break'] == pytest.approx(0.77)
    assert payload['synthetic_defect_generator'].defects.defect_severities['via_open'] == pytest.approx(0.88)
    dialog.close()


def test_augmentation_preview_dialog_retries_pcb_defects_until_visible_change(qapp, monkeypatch):
    root = make_test_dir('augmentation_preview_dialog_pcb_retry')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((48, 48), dtype=np.uint8)
    image[22:26, 8:40] = 255
    label = image.copy()
    Image.fromarray(image, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label, mode='L').save(label_dir / 'frame_a.png')

    calls: list[int | None] = []

    def _fake_call(self, image_array, mask_array=None, *, seed=None, return_debug=False, return_augmented_mask=False):
        calls.append(seed)
        source_image = np.asarray(image_array).copy()
        source_mask = np.asarray(mask_array).copy() if mask_array is not None else np.zeros_like(source_image)
        if len(calls) == 1:
            empty_mask = np.zeros_like(source_mask)
            if return_debug:
                if return_augmented_mask:
                    return source_image.copy(), source_image.copy(), empty_mask, source_mask.copy()
                return source_image.copy(), source_image.copy(), empty_mask
            if return_augmented_mask:
                return source_image.copy(), empty_mask, source_mask.copy()
            return source_image.copy(), empty_mask
        defect_mask = np.zeros_like(source_mask)
        defect_mask.flat[0] = 1.0
        changed_image = source_image.copy()
        changed_image.flat[0] = 1.0
        augmented_mask = source_mask.copy()
        augmented_mask.flat[1] = 1.0
        if return_debug:
            if return_augmented_mask:
                return source_image.copy(), changed_image, defect_mask, augmented_mask
            return source_image.copy(), changed_image, defect_mask
        if return_augmented_mask:
            return changed_image, defect_mask, augmented_mask
        return changed_image, defect_mask

    monkeypatch.setattr(augmentation_preview_module.PCBDefectAugmentor, '__call__', _fake_call)

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    before = dialog._augmented_image_array.copy()
    dialog._toggle_boxes['synthetic_topology'].setChecked(True)
    dialog._toggle_boxes['pcb_defects'].setChecked(True)
    dialog._toggle_boxes['pcb_break'].setChecked(True)
    dialog.pcb_defects_probability_spinbox.setValue(1.0)
    dialog.pcb_defect_type_spinboxes['break'].setValue(100)
    qapp.processEvents()

    assert len(calls) >= 2
    assert dialog._augmented_image_array is not None
    assert not np.array_equal(before, dialog._augmented_image_array)
    dialog.close()


def test_augmentation_preview_dialog_keeps_original_label_when_synthetic_defects_are_enabled(qapp, monkeypatch):
    root = make_test_dir('augmentation_preview_dialog_pcb_mask')
    sample_dir = root / 'samples'
    label_dir = root / 'labels'
    sample_dir.mkdir()
    label_dir.mkdir()

    image = np.zeros((48, 48), dtype=np.uint8)
    image[22:26, 8:40] = 255
    label = image.copy()
    Image.fromarray(image, mode='L').save(sample_dir / 'frame_a.png')
    Image.fromarray(label, mode='L').save(label_dir / 'frame_a.png')

    def _fake_call(self, image_array, mask_array=None, *, seed=None, return_debug=False, return_augmented_mask=False):
        source_image = np.asarray(image_array).copy()
        source_mask = np.asarray(mask_array).copy()
        defect_mask = np.zeros_like(source_mask)
        defect_mask.flat[0] = 1.0
        augmented_mask = np.zeros_like(source_mask)
        augmented_mask[..., 10:20, 10:20] = 1.0
        changed_image = source_image.copy()
        changed_image.flat[0] = 1.0
        if return_debug:
            if return_augmented_mask:
                return source_image.copy(), changed_image, defect_mask, augmented_mask
            return source_image.copy(), changed_image, defect_mask
        if return_augmented_mask:
            return changed_image, defect_mask, augmented_mask
        return changed_image, defect_mask

    monkeypatch.setattr(augmentation_preview_module.PCBDefectAugmentor, '__call__', _fake_call)

    dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))
    dialog.show()
    qapp.processEvents()

    dialog._toggle_boxes['synthetic_topology'].setChecked(True)
    qapp.processEvents()
    original_label = dialog._augmented_label_array.copy()
    dialog._toggle_boxes['pcb_defects'].setChecked(True)
    dialog._toggle_boxes['pcb_break'].setChecked(True)
    dialog.pcb_defects_probability_spinbox.setValue(1.0)
    dialog.pcb_defect_type_spinboxes['break'].setValue(100)
    qapp.processEvents()

    assert dialog._augmented_label_array is not None
    assert np.array_equal(dialog._augmented_label_array, original_label)
    assert dialog._augmented_image_array is not None
    assert int(dialog._augmented_image_array.flat[0]) > 0
    dialog.close()
