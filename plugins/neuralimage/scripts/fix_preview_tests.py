from __future__ import annotations

import re
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'tests/test_augmentation_preview_dialog.py'
text = path.read_text(encoding='utf-8')

text = text.replace(
    'dialog = AugmentationPreviewDialog(_build_training_parameters(sample_dir, label_dir))',
    'dialog, panel = _build_preview_dialog(_build_training_parameters(sample_dir, label_dir))',
)
text = text.replace(
    'dialog = AugmentationPreviewDialog(training_parameters)',
    'dialog, panel = _build_preview_dialog(training_parameters)',
)

replacements = {
    "dialog._toggle_boxes['rotate_90']": 'panel.horizontal_rotation',
    "dialog._toggle_boxes['flip_x']": 'panel.flip_x',
    "dialog._toggle_boxes['flip_y']": 'panel.flip_y',
    "dialog._toggle_boxes['random_crop']": 'panel.random_crop_check_box',
    "dialog._toggle_boxes['synthetic_topology']": 'panel.synthetic_defect_generator_check_box',
    "dialog._toggle_boxes['random_artifacts']": 'panel.random_artifacts_check_box',
    "dialog._toggle_boxes['pcb_defects']": 'panel.pcb_defects_check_box',
    "dialog._toggle_boxes['artifact_flake']": 'panel.random_artifact_type_checkboxes["flake"]',
    "dialog._toggle_boxes['pcb_break']": 'panel.pcb_defect_type_checkboxes["break"]',
    "dialog._toggle_boxes['pcb_via']": 'panel.pcb_defect_type_checkboxes["via"]',
    "dialog._toggle_boxes['ic_line_break']": 'panel.ic_defect_type_checkboxes["line_break"]',
    "dialog._toggle_boxes['ic_bridge']": 'panel.ic_defect_type_checkboxes["bridge"]',
    "dialog._toggle_boxes['ic_via_open']": 'panel.ic_defect_type_checkboxes["via_open"]',
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Photometric toggles: master + spinbox
text = text.replace(
    "dialog._toggle_boxes['brightness'].setChecked(True)\n    dialog.augmentation_brightness_spinbox.setValue",
    "panel.photometric_groupbox.setChecked(True)\n    panel.augmentation_brightness_spinbox.setValue",
)
text = text.replace(
    "dialog._toggle_boxes['brightness'].setChecked(True)\n    panel.augmentation_brightness_spinbox.setValue",
    "panel.photometric_groupbox.setChecked(True)\n    panel.augmentation_brightness_spinbox.setValue",
)
text = text.replace(
    "dialog._toggle_boxes['brightness'].setChecked(True)\n    dialog.augmentation_brightness_spinbox.setValue(0.33)",
    "panel.photometric_groupbox.setChecked(True)\n    panel.augmentation_brightness_spinbox.setValue(0.33)",
)
text = text.replace(
    "dialog._toggle_boxes['noise'].setChecked(True)\n    dialog.augmentation_noise_probability_spinbox.setValue",
    "panel.photometric_groupbox.setChecked(True)\n    panel.augmentation_noise_probability_spinbox.setValue",
)

panel_attrs = [
    'crops_per_image_spinbox', 'cutout_holes_spinbox', 'pcb_defects_max_count_spinbox',
    'augmentation_brightness_spinbox', 'augmentation_noise_probability_spinbox', 'augmentation_noise_sigma_spinbox',
    'synthetic_background_noise_sigma_min_spinbox', 'synthetic_background_noise_sigma_max_spinbox',
    'synthetic_image_width_spinbox', 'synthetic_image_height_spinbox',
    'synthetic_trace_count_min_spinbox', 'synthetic_trace_count_max_spinbox',
    'synthetic_segment_count_min_spinbox', 'synthetic_segment_count_max_spinbox',
    'synthetic_trace_half_width_min_spinbox', 'synthetic_trace_half_width_max_spinbox',
    'synthetic_background_noise_sigma_min_spinbox', 'synthetic_background_noise_sigma_max_spinbox',
    'synthetic_trace_noise_sigma_min_spinbox', 'synthetic_trace_noise_sigma_max_spinbox',
    'random_artifacts_probability_spinbox', 'pcb_defects_probability_spinbox',
    'shift_spinbox', 'tech_aug_max_changed_pixels_ratio_spinbox', 'synthetic_dataset_factor_spinbox',
    'pcb_defect_type_spinboxes',
]
for attr in panel_attrs:
    text = re.sub(rf'(?<!panel\.)dialog\.{re.escape(attr)}\b', f'panel.{attr}', text)

text = text.replace(
    'assert isinstance(dialog.sem_normalization_editor, SemConfigSectionEditor)',
    'assert isinstance(dialog.sem_normalization_editor, CompactSemSectionEditor)',
)
text = text.replace(
    "assert dialog.sem_normalization_editor.section == 'preprocessing'",
    "assert dialog.sem_normalization_editor.section_key == 'preprocessing'",
)
text = text.replace(
    "assert dialog.sem_augmentation_editor.section == 'augmentation'",
    "assert dialog.sem_augmentation_editor.section_key == 'augmentation'",
)
text = text.replace(
    "assert 'tech_augmentation' in dialog._toggle_boxes",
    'assert panel.tech_augmentation_check_box.isVisible()',
)

path.write_text(text, encoding='utf-8')
print('remaining _toggle_boxes:', text.count('_toggle_boxes'))
print('remaining AugmentationPreviewDialog(', text.count('AugmentationPreviewDialog('))
