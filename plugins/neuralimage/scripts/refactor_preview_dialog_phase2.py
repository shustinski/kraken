from pathlib import Path
import re

path = Path(__file__).resolve().parents[1] / 'src/neuralimage/view/augmentation_preview_dialog.py'
text = path.read_text(encoding='utf-8')

old_connect = """    def _connect_signals(self) -> None:
        self.prev_button.clicked.connect(self._show_previous_sample)
        self.next_button.clicked.connect(self._show_next_sample)
        self.resample_button.clicked.connect(self._resample_current_sample)
        self.apply_to_main_button.clicked.connect(self._emit_apply_to_main)
        self.sample_list_widget.currentRowChanged.connect(self._on_sample_list_row_changed)
        self.image_preview.middle_pressed.connect(self._show_original_preview)
        self.image_preview.middle_released.connect(self._restore_augmented_preview)
        self.label_preview.middle_pressed.connect(self._show_original_preview)
        self.label_preview.middle_released.connect(self._restore_augmented_preview)
        self.full_image_check_box.toggled.connect(self._on_full_image_toggled)
        self.synthetic_topology_domain_combo.currentIndexChanged.connect(self._on_value_changed)
        self.pcb_topology_family_combo.currentIndexChanged.connect(self._on_value_changed)
        self.ic_topology_family_combo.currentIndexChanged.connect(self._on_value_changed)
        for checkbox in self._toggle_boxes.values():
            checkbox.toggled.connect(self._on_toggle_changed)
        connected_widgets: set[int] = set()
        for widgets in self._value_widgets.values():
            for widget in widgets:
                widget_id = id(widget)
                if widget_id in connected_widgets:
                    continue
                connected_widgets.add(widget_id)
                if hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(self._on_value_changed)
                elif hasattr(widget, 'currentIndexChanged'):
                    widget.currentIndexChanged.connect(self._on_value_changed)
        self.tech_aug_min_operations_spinbox.valueChanged.connect(self._on_value_changed)
        self.tech_aug_max_operations_spinbox.valueChanged.connect(self._on_value_changed)
        self.synthetic_image_width_spinbox.valueChanged.connect(self._on_value_changed)
        self.synthetic_image_height_spinbox.valueChanged.connect(self._on_value_changed)
        self.pcb_defects_min_count_spinbox.valueChanged.connect(self._on_pcb_defect_count_changed)
        self.pcb_defects_max_count_spinbox.valueChanged.connect(self._on_pcb_defect_count_changed)"""

new_connect = """    def _connect_signals(self) -> None:
        panel = self._panel
        self.prev_button.clicked.connect(self._show_previous_sample)
        self.next_button.clicked.connect(self._show_next_sample)
        self.resample_button.clicked.connect(self._resample_current_sample)
        self.apply_to_main_button.clicked.connect(self._emit_apply_to_main)
        self.sample_list_widget.currentRowChanged.connect(self._on_sample_list_row_changed)
        self.image_preview.middle_pressed.connect(self._show_original_preview)
        self.image_preview.middle_released.connect(self._restore_augmented_preview)
        self.label_preview.middle_pressed.connect(self._show_original_preview)
        self.label_preview.middle_released.connect(self._restore_augmented_preview)
        self.full_image_check_box.toggled.connect(self._on_full_image_toggled)
        self.sem_normalization_editor.changed.connect(self._on_sem_pipeline_changed)
        self.sem_augmentation_editor.changed.connect(self._on_sem_pipeline_changed)
        for checkbox in (
            panel.horizontal_rotation,
            panel.vertical_rotation,
            panel.flip_x,
            panel.flip_y,
            panel.random_crop_check_box,
            panel.scale_augmentation_check_box,
            panel.photometric_groupbox,
            panel.tech_augmentation_check_box,
            panel.cutout_check_box,
            panel.random_artifacts_check_box,
            panel.mixup_check_box,
            panel.synthetic_defect_generator_check_box,
            *panel.random_artifact_type_checkboxes.values(),
            *panel.pcb_defect_type_checkboxes.values(),
            *panel.ic_defect_type_checkboxes.values(),
        ):
            checkbox.toggled.connect(self._on_settings_changed)
        for widget in (
            panel.shift_spinbox,
            panel.crops_per_image_spinbox,
            panel.scale_augmentation_strength_spinbox,
            panel.augmentation_brightness_spinbox,
            panel.augmentation_contrast_spinbox,
            panel.augmentation_gamma_spinbox,
            panel.augmentation_noise_probability_spinbox,
            panel.augmentation_noise_sigma_spinbox,
            panel.augmentation_blur_probability_spinbox,
            panel.augmentation_blur_radius_spinbox,
            panel.tech_aug_min_operations_spinbox,
            panel.tech_aug_max_operations_spinbox,
            panel.tech_aug_max_changed_pixels_ratio_spinbox,
            panel.tech_aug_max_foreground_ratio_delta_spinbox,
            panel.tech_aug_global_width_probability_spinbox,
            panel.tech_aug_scale_rethreshold_probability_spinbox,
            panel.tech_aug_blur_threshold_probability_spinbox,
            panel.tech_aug_boundary_aware_probability_spinbox,
            panel.tech_aug_local_morphology_probability_spinbox,
            panel.tech_aug_gap_variation_probability_spinbox,
            panel.cutout_probability_spinbox,
            panel.cutout_holes_spinbox,
            panel.cutout_size_ratio_spinbox,
            panel.random_artifacts_probability_spinbox,
            panel.random_artifacts_count_spinbox,
            panel.random_artifacts_size_ratio_spinbox,
            panel.mixup_probability_spinbox,
            panel.mixup_alpha_spinbox,
            panel.synthetic_dataset_factor_spinbox,
            panel.synthetic_image_width_spinbox,
            panel.synthetic_image_height_spinbox,
            panel.synthetic_trace_count_min_spinbox,
            panel.synthetic_trace_count_max_spinbox,
            panel.synthetic_segment_count_min_spinbox,
            panel.synthetic_segment_count_max_spinbox,
            panel.synthetic_trace_half_width_min_spinbox,
            panel.synthetic_trace_half_width_max_spinbox,
            panel.synthetic_background_noise_sigma_min_spinbox,
            panel.synthetic_background_noise_sigma_max_spinbox,
            panel.synthetic_trace_noise_sigma_min_spinbox,
            panel.synthetic_trace_noise_sigma_max_spinbox,
            panel.pcb_defects_probability_spinbox,
            panel.pcb_defects_min_count_spinbox,
            panel.pcb_defects_max_count_spinbox,
            panel.synthetic_topology_domain_combo,
            panel.pcb_topology_family_combo,
            panel.ic_topology_family_combo,
            *panel.pcb_defect_type_spinboxes.values(),
            *panel.ic_defect_type_spinboxes.values(),
        ):
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._on_settings_changed)
            elif hasattr(widget, 'currentIndexChanged'):
                widget.currentIndexChanged.connect(self._on_settings_changed)"""

if old_connect not in text:
    raise SystemExit('connect block not found')
text = text.replace(old_connect, new_connect)

sync_pat = r'    def _sync_group_boxes\(self\) -> None:.*?(?=    def _on_toggle_changed)'
new_sync = """    def _sync_group_boxes(self) -> None:
        panel = self._panel
        panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
        panel._sync_training_augmentation_controls()
        panel._sync_synthetic_defect_generator_controls(panel.synthetic_defect_generator_check_box.isChecked())
        panel._sync_sem_segmentation_controls()

    def _on_settings_changed(self, *_args: object) -> None:
        self._prepared_arrays_cache.clear()
        self._sync_group_boxes()
        self._refresh_preview()

"""
text, n = re.subn(sync_pat, new_sync, text, count=1, flags=re.DOTALL)
if n != 1:
    raise SystemExit(f'sync replace failed: {n}')

text = text.replace(
    """    def _on_toggle_changed(self, _checked: bool) -> None:
        self._sync_group_boxes()
        self._refresh_preview()

    def _on_value_changed(self, _value: object) -> None:
        self._prepared_arrays_cache.clear()
        self._sync_group_boxes()
        self._refresh_preview()

""",
    '',
)

replacements = {
    "self._toggle_boxes['rotate_90']": 'self._panel.horizontal_rotation',
    "self._toggle_boxes['rotate_180']": 'self._panel.vertical_rotation',
    "self._toggle_boxes['flip_x']": 'self._panel.flip_x',
    "self._toggle_boxes['flip_y']": 'self._panel.flip_y',
    "self._toggle_boxes['random_crop']": 'self._panel.random_crop_check_box',
    "self._toggle_boxes['scale']": 'self._panel.scale_augmentation_check_box',
    "self._toggle_boxes['tech_augmentation']": 'self._panel.tech_augmentation_check_box',
    "self._toggle_boxes['cutout']": 'self._panel.cutout_check_box',
    "self._toggle_boxes['random_artifacts']": 'self._panel.random_artifacts_check_box',
    "self._toggle_boxes['mixup']": 'self._panel.mixup_check_box',
    "self._toggle_boxes['synthetic_topology']": 'self._panel.synthetic_defect_generator_check_box',
    "self._toggle_boxes['pcb_defects']": 'self._panel.pcb_defects_check_box',
    "self._toggle_boxes.get('synthetic_topology') and self._toggle_boxes['synthetic_topology']": 'self._panel.synthetic_defect_generator_check_box',
}
for old, new in replacements.items():
    text = text.replace(old, new)

for name in ('dust', 'resist_residue', 'etch_residue', 'particle_cluster', 'flake'):
    text = text.replace(
        f"self._toggle_boxes['artifact_{name}']",
        f"self._panel.random_artifact_type_checkboxes[{name!r}]",
    )

pcb_map = {
    'pcb_break': 'break', 'pcb_short': 'short', 'pcb_missing_copper': 'missing_copper',
    'pcb_excess_copper': 'excess_copper', 'pcb_pinhole': 'pinhole',
    'pcb_spurious_copper': 'spurious_copper', 'pcb_via': 'via', 'pcb_misalignment': 'misalignment',
}
for key, name in pcb_map.items():
    text = text.replace(f"self._toggle_boxes['{key}']", f"self._panel.pcb_defect_type_checkboxes[{name!r}]")

ic_map = {
    'ic_line_break': 'line_break', 'ic_bridge': 'bridge', 'ic_necking': 'necking',
    'ic_missing_metal': 'missing_metal', 'ic_spur': 'spur', 'ic_pinhole': 'pinhole',
    'ic_via_open': 'via_open', 'ic_line_shift': 'line_shift',
}
for key, name in ic_map.items():
    text = text.replace(f"self._toggle_boxes['{key}']", f"self._panel.ic_defect_type_checkboxes[{name!r}]")

spinbox_attrs = [
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
]
for attr in spinbox_attrs:
    text = re.sub(rf'(?<!_panel\.)self\.{re.escape(attr)}\b', f'self._panel.{attr}', text)

text = text.replace('self.pcb_defect_type_spinboxes', 'self._panel.pcb_defect_type_spinboxes')
text = text.replace('self.ic_defect_type_spinboxes', 'self._panel.ic_defect_type_spinboxes')

apply_pat = r'    def _build_apply_payload\(self\) -> dict\[str, object\]:.*?(?=    def _build_apply_synthetic_defect_generator_config|    def _show_previous_sample)'
apply_match = re.search(apply_pat, text, flags=re.DOTALL)
if not apply_match:
    raise SystemExit('apply payload not found')

new_apply = """    def _build_apply_payload(self) -> dict[str, object]:
        panel = self._panel
        sem_config = self._current_sem_config()
        return {
            'horizontal_rotation': panel.horizontal_rotation.isChecked(),
            'vertical_rotation': panel.vertical_rotation.isChecked(),
            'flip_x': panel.flip_x.isChecked(),
            'flip_y': panel.flip_y.isChecked(),
            'random_crop': panel.random_crop_check_box.isChecked(),
            'step': int(panel.shift_spinbox.value()),
            'crops_per_image': int(panel.crops_per_image_spinbox.value()),
            'scale_augmentation': panel.scale_augmentation_check_box.isChecked(),
            'scale_augmentation_strength': float(panel.scale_augmentation_strength_spinbox.value()),
            'additional_augmentation': panel.photometric_groupbox.isChecked(),
            'augmentation_brightness_strength': float(panel.augmentation_brightness_spinbox.value()),
            'augmentation_contrast_strength': float(panel.augmentation_contrast_spinbox.value()),
            'augmentation_gamma_strength': float(panel.augmentation_gamma_spinbox.value()),
            'augmentation_noise_probability': float(panel.augmentation_noise_probability_spinbox.value()),
            'augmentation_noise_sigma': float(panel.augmentation_noise_sigma_spinbox.value()),
            'augmentation_blur_probability': float(panel.augmentation_blur_probability_spinbox.value()),
            'augmentation_blur_radius': float(panel.augmentation_blur_radius_spinbox.value()),
            'synthetic_defect_generator': panel.get_synthetic_defect_generator_config(),
            'tech_aug': panel.get_tech_aug_config(),
            'preprocessing': asdict(sem_config.preprocessing),
            'sem_augmentation': asdict(sem_config.augmentation),
            'cutout_enabled': panel.cutout_check_box.isChecked(),
            'cutout_probability': float(panel.cutout_probability_spinbox.value()),
            'cutout_holes': int(panel.cutout_holes_spinbox.value()),
            'cutout_size_ratio': float(panel.cutout_size_ratio_spinbox.value()),
            'random_artifacts_enabled': panel.random_artifacts_check_box.isChecked(),
            'random_artifacts_probability': float(panel.random_artifacts_probability_spinbox.value()),
            'random_artifacts_count': int(panel.random_artifacts_count_spinbox.value()),
            'random_artifacts_size_ratio': float(panel.random_artifacts_size_ratio_spinbox.value()),
            **{
                f'random_artifacts_{artifact_name}_enabled': checkbox.isChecked()
                for artifact_name, checkbox in panel.random_artifact_type_checkboxes.items()
            },
            'mixup_enabled': panel.mixup_check_box.isChecked(),
            'mixup_probability': float(panel.mixup_probability_spinbox.value()),
            'mixup_alpha': float(panel.mixup_alpha_spinbox.value()),
            'pcb_defects': build_pcb_defect_parameters(None),
        }

"""
text = text[:apply_match.start()] + new_apply + text[apply_match.end():]

for method in (
    '_build_apply_synthetic_defect_generator_config',
    '_build_apply_tech_aug_config',
    '_build_apply_pcb_defects_config',
    '_build_apply_ic_defects_config',
):
    pat = rf'    def {method}\(.*?\n(?:(?!    def ).*\n)*?(?=    def )'
    text, _ = re.subn(pat, '', text, count=1, flags=re.DOTALL)

text = text.replace(
    'config = self._build_apply_synthetic_defect_generator_config()',
    'config = build_synthetic_defect_generator_parameters(self._panel.get_synthetic_defect_generator_config())',
)
text = text.replace(
    'config = self._build_apply_tech_aug_config()',
    'config = build_tech_augmentation_config(self._panel.get_tech_aug_config())',
)

old_photo = """    def _apply_photometric_augmentations(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        image = image_patch.astype(np.float32, copy=True)
        with _seeded_random(self._seed_for(sample_index, 'photometric')):
            if self._toggle_boxes['blur'].isChecked() and self._passes_probability("""
if old_photo in text:
    text = re.sub(
        r'    def _apply_photometric_augmentations\(self, sample_index: int, image_patch: np.ndarray\) -> np.ndarray:.*?return image\.astype\(np\.float32, copy=False\)',
        """    def _apply_photometric_augmentations(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        panel = self._panel
        if not panel.photometric_groupbox.isChecked():
            return image_patch
        image = image_patch.astype(np.float32, copy=True)
        with _seeded_random(self._seed_for(sample_index, 'photometric')):
            blur_probability = float(panel.augmentation_blur_probability_spinbox.value())
            if blur_probability > 0.0 and self._passes_probability(sample_index, 'blur_probability', blur_probability):
                blur_radius = max(0.0, float(panel.augmentation_blur_radius_spinbox.value()))
                if blur_radius > 0.0:
                    image = SampleFastCutter._apply_gaussian_blur(image, blur_radius)
            brightness_strength = max(0.0, float(panel.augmentation_brightness_spinbox.value()))
            if brightness_strength > 0.0:
                image *= float(1.0 + brightness_strength)
            contrast_strength = max(0.0, float(panel.augmentation_contrast_spinbox.value()))
            if contrast_strength > 0.0:
                contrast = 1.0 + contrast_strength
                mean = image.mean(axis=(1, 2), keepdims=True)
                image = (image - mean) * float(contrast) + mean
            gamma_strength = max(0.0, float(panel.augmentation_gamma_spinbox.value()))
            if gamma_strength > 0.0:
                gamma = max(0.1, 1.0 - min(gamma_strength, 0.9))
                image = np.power(np.clip(image, 0.0, 1.0), float(gamma)).astype(np.float32, copy=False)
            noise_probability = float(panel.augmentation_noise_probability_spinbox.value())
            if noise_probability > 0.0 and self._passes_probability(sample_index, 'noise_probability', noise_probability):
                sigma = max(0.0, float(panel.augmentation_noise_sigma_spinbox.value()))
                if sigma > 0.0:
                    image += np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        np.clip(image, 0.0, 1.0, out=image)
        return image.astype(np.float32, copy=False)""",
        text,
        count=1,
        flags=re.DOTALL,
    )

text = re.sub(
    r'    def _has_selected_tech_variations\(self\) -> bool:.*?^\s{4}\)',
    """    def _has_selected_tech_variations(self) -> bool:
        config = build_tech_augmentation_config(self._panel.get_tech_aug_config())
        if not config.enabled:
            return False
        return any(
            float(probability) > 0.0
            for probability in (
                config.global_width.probability,
                config.scale_rethreshold.probability,
                config.blur_threshold.probability,
                config.boundary_aware.probability,
                config.local_morphology.probability,
                config.gap_variation.probability,
            )
        )""",
    text,
    count=1,
    flags=re.DOTALL | re.MULTILINE,
)

text = text.replace(
    """    def _get_synthetic_topology_domain(self) -> str:
        return str(
            self._panel.synthetic_topology_domain_combo.currentData()
            or self._panel.synthetic_topology_domain_combo.currentText()
            or 'pcb'
        ).strip().lower()""",
    """    def _get_synthetic_topology_domain(self) -> str:
        return str(self._panel.get_synthetic_topology_domain_value()).strip().lower()""",
)

text = text.replace(
    'from neuralimage.view.sem_compact_section_editor import CompactSemSectionEditor, SemAugmentationSectionEditor\n',
    '',
)
text = text.replace(
    'from neuralimage.view.settings_panel_widgets import NoWheelComboBox, create_double_spinbox, create_size_widget, create_slider, create_spinbox\n',
    'from neuralimage.view.settings_panel_widgets import NoWheelComboBox\n',
)

for method in ('_set_value_widgets_enabled', '_set_value_widgets_visible', '_on_pcb_defect_count_changed'):
    pat = rf'    def {method}\(.*?\n(?:(?!    def ).*\n)*?(?=    def )'
    text, _ = re.subn(pat, '', text, count=1, flags=re.DOTALL)

# cleanup remaining _toggle_boxes references
if '_toggle_boxes' in text:
    print('WARNING: remaining _toggle_boxes references')

path.write_text(text, encoding='utf-8')
print('phase 2 done, lines:', len(text.splitlines()))
