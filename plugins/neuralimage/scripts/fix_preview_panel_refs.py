from __future__ import annotations

import re
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'src/neuralimage/view/augmentation_preview_dialog.py'
text = path.read_text(encoding='utf-8')

toggle_map = {
    "'rotate_90'": 'self._panel.horizontal_rotation',
    "'rotate_180'": 'self._panel.vertical_rotation',
    "'flip_x'": 'self._panel.flip_x',
    "'flip_y'": 'self._panel.flip_y',
    "'random_crop'": 'self._panel.random_crop_check_box',
    "'scale'": 'self._panel.scale_augmentation_check_box',
    "'cutout'": 'self._panel.cutout_check_box',
    "'mixup'": 'self._panel.mixup_check_box',
    "'random_artifacts'": 'self._panel.random_artifacts_check_box',
    "'synthetic_topology'": 'self._panel.synthetic_defect_generator_check_box',
    "'pcb_defects'": 'self._panel.pcb_defects_check_box',
    "'tech_augmentation'": 'self._panel.tech_augmentation_check_box',
    "'brightness'": 'self._photometric_effect_enabled(self._panel.augmentation_brightness_spinbox)',
    "'contrast'": 'self._photometric_effect_enabled(self._panel.augmentation_contrast_spinbox)',
    "'gamma'": 'self._photometric_effect_enabled(self._panel.augmentation_gamma_spinbox)',
    "'noise'": 'self._photometric_effect_enabled(self._panel.augmentation_noise_probability_spinbox)',
    "'blur'": 'self._photometric_effect_enabled(self._panel.augmentation_blur_probability_spinbox)',
    "'tech_global_width'": 'self._tech_effect_enabled(self._panel.tech_aug_global_width_probability_spinbox)',
    "'tech_scale_rethreshold'": 'self._tech_effect_enabled(self._panel.tech_aug_scale_rethreshold_probability_spinbox)',
    "'tech_blur_threshold'": 'self._tech_effect_enabled(self._panel.tech_aug_blur_threshold_probability_spinbox)',
    "'tech_boundary_aware'": 'self._tech_effect_enabled(self._panel.tech_aug_boundary_aware_probability_spinbox)',
    "'tech_local_morphology'": 'self._tech_effect_enabled(self._panel.tech_aug_local_morphology_probability_spinbox)',
    "'tech_gap_variation'": 'self._tech_effect_enabled(self._panel.tech_aug_gap_variation_probability_spinbox)',
    "'artifact_dust'": 'self._panel.random_artifact_type_checkboxes["dust"]',
    "'artifact_resist_residue'": 'self._panel.random_artifact_type_checkboxes["resist_residue"]',
    "'artifact_etch_residue'": 'self._panel.random_artifact_type_checkboxes["etch_residue"]',
    "'artifact_particle_cluster'": 'self._panel.random_artifact_type_checkboxes["particle_cluster"]',
    "'artifact_flake'": 'self._panel.random_artifact_type_checkboxes["flake"]',
    "'pcb_break'": 'self._panel.pcb_defect_type_checkboxes["break"]',
    "'pcb_short'": 'self._panel.pcb_defect_type_checkboxes["short"]',
    "'pcb_missing_copper'": 'self._panel.pcb_defect_type_checkboxes["missing_copper"]',
    "'pcb_excess_copper'": 'self._panel.pcb_defect_type_checkboxes["excess_copper"]',
    "'pcb_pinhole'": 'self._panel.pcb_defect_type_checkboxes["pinhole"]',
    "'pcb_spurious_copper'": 'self._panel.pcb_defect_type_checkboxes["spurious_copper"]',
    "'pcb_via'": 'self._panel.pcb_defect_type_checkboxes["via"]',
    "'pcb_misalignment'": 'self._panel.pcb_defect_type_checkboxes["misalignment"]',
    "'ic_line_break'": 'self._panel.ic_defect_type_checkboxes["line_break"]',
    "'ic_bridge'": 'self._panel.ic_defect_type_checkboxes["bridge"]',
    "'ic_necking'": 'self._panel.ic_defect_type_checkboxes["necking"]',
    "'ic_missing_metal'": 'self._panel.ic_defect_type_checkboxes["missing_metal"]',
    "'ic_spur'": 'self._panel.ic_defect_type_checkboxes["spur"]',
    "'ic_pinhole'": 'self._panel.ic_defect_type_checkboxes["pinhole"]',
    "'ic_via_open'": 'self._panel.ic_defect_type_checkboxes["via_open"]',
    "'ic_line_shift'": 'self._panel.ic_defect_type_checkboxes["line_shift"]',
}

for key, repl in toggle_map.items():
    text = text.replace(f'self._toggle_boxes[{key}]', repl)

text = re.sub(
    r'self\._toggle_boxes\.get\(\'synthetic_topology\'\) and self\._panel\.synthetic_defect_generator_check_box\.isChecked\(\)',
    'self._panel.synthetic_defect_generator_check_box.isChecked()',
    text,
)

panel_attrs = [
    'shift_spinbox', 'crops_per_image_spinbox', 'scale_augmentation_strength_spinbox',
    'augmentation_brightness_spinbox', 'augmentation_contrast_spinbox', 'augmentation_gamma_spinbox',
    'augmentation_noise_probability_spinbox', 'augmentation_noise_sigma_spinbox',
    'augmentation_blur_probability_spinbox', 'augmentation_blur_radius_spinbox',
    'tech_aug_min_operations_spinbox', 'tech_aug_max_operations_spinbox',
    'tech_aug_max_changed_pixels_ratio_spinbox', 'tech_aug_max_foreground_ratio_delta_spinbox',
    'tech_aug_global_width_probability_spinbox', 'tech_aug_scale_rethreshold_probability_spinbox',
    'tech_aug_blur_threshold_probability_spinbox', 'tech_aug_boundary_aware_probability_spinbox',
    'tech_aug_local_morphology_probability_spinbox', 'tech_aug_gap_variation_probability_spinbox',
    'cutout_probability_spinbox', 'cutout_holes_spinbox', 'cutout_size_ratio_spinbox',
    'random_artifacts_probability_spinbox', 'random_artifacts_count_spinbox', 'random_artifacts_size_ratio_spinbox',
    'mixup_probability_spinbox', 'mixup_alpha_spinbox',
    'synthetic_dataset_factor_spinbox', 'synthetic_image_width_spinbox', 'synthetic_image_height_spinbox',
    'synthetic_trace_count_min_spinbox', 'synthetic_trace_count_max_spinbox',
    'synthetic_segment_count_min_spinbox', 'synthetic_segment_count_max_spinbox',
    'synthetic_trace_half_width_min_spinbox', 'synthetic_trace_half_width_max_spinbox',
    'synthetic_background_noise_sigma_min_spinbox', 'synthetic_background_noise_sigma_max_spinbox',
    'synthetic_trace_noise_sigma_min_spinbox', 'synthetic_trace_noise_sigma_max_spinbox',
    'pcb_defects_probability_spinbox', 'pcb_defects_min_count_spinbox', 'pcb_defects_max_count_spinbox',
    'synthetic_topology_domain_combo', 'pcb_topology_family_combo', 'ic_topology_family_combo',
    'pcb_defect_type_spinboxes', 'ic_defect_type_spinboxes',
]
for attr in panel_attrs:
    text = re.sub(rf'(?<!self\._panel\.)self\.{re.escape(attr)}\b', f'self._panel.{attr}', text)

text = re.sub(
    r'self\._photometric_effect_enabled\((self\._panel\.[^)]+)\)\.isChecked\(\)',
    r'self._photometric_effect_enabled(\1)',
    text,
)
text = re.sub(
    r'self\._tech_effect_enabled\((self\._panel\.[^)]+)\)\.isChecked\(\)',
    r'self._tech_effect_enabled(\1)',
    text,
)

text = text.replace(
    "'tech_augmentation' in self._toggle_boxes and not self._panel.tech_augmentation_check_box.isChecked()",
    'not self._panel.tech_augmentation_check_box.isChecked()',
)

path.write_text(text, encoding='utf-8')
print('remaining _toggle_boxes:', text.count('_toggle_boxes'))
