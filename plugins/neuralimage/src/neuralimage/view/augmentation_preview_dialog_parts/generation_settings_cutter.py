"""augmentation_preview_dialog: generation settings cutter."""
from __future__ import annotations
import copy
from dataclasses import replace
from PyQt6.QtWidgets import (
    QWidget,
)
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)
from neuralimage.lib.data_interfaces import (
    build_ic_defect_parameters,
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
)
from neuralimage.lib.images import SampleFastCutter

from neuralimage.view.augmentation_preview_dialog_parts.module_state import (
    MAX_AUGMENTATION_MULTIPLIER,
    MIN_AUGMENTATION_MULTIPLIER,
)


class GenerationSettingsCutterMixin:
    def _build_apply_pcb_defects_config(self):
        if not self._panel.synthetic_defect_generator_check_box.isChecked():
            config = build_pcb_defect_parameters(None)
            config.enabled = False
            return config
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self._training_parameters, 'synthetic_defect_generator', None)
        )
        config = copy.deepcopy(synthetic_generator.pcb_defects)
        min_defects = int(self._panel.pcb_defects_min_count_spinbox.value())
        max_defects = int(self._panel.pcb_defects_max_count_spinbox.value())
        if min_defects > max_defects:
            min_defects, max_defects = max_defects, min_defects
        config.enabled = self._panel.pcb_defects_check_box.isChecked()
        config.defect_probability = float(self._panel.pcb_defects_probability_spinbox.value())
        config.min_defects = min_defects
        config.max_defects = max_defects
        for defect_name in self._panel.pcb_defect_type_checkboxes:
            config.defect_probabilities[defect_name] = (
                1.0 if self._panel.pcb_defect_type_checkboxes[defect_name].isChecked() else 0.0
            )
            config.defect_severities[defect_name] = float(self._panel.pcb_defect_type_spinboxes[defect_name].value()) / 100.0
        return config

    def _build_apply_ic_defects_config(self):
        if not self._panel.synthetic_defect_generator_check_box.isChecked():
            config = build_ic_defect_parameters(None)
            config.enabled = False
            return config
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self._training_parameters, 'synthetic_defect_generator', None)
        )
        config = copy.deepcopy(synthetic_generator.ic_defects)
        min_defects = int(self._panel.pcb_defects_min_count_spinbox.value())
        max_defects = int(self._panel.pcb_defects_max_count_spinbox.value())
        if min_defects > max_defects:
            min_defects, max_defects = max_defects, min_defects
        config.enabled = self._panel.pcb_defects_check_box.isChecked()
        config.defect_probability = float(self._panel.pcb_defects_probability_spinbox.value())
        config.min_defects = min_defects
        config.max_defects = max_defects
        for defect_name in self._panel.ic_defect_type_checkboxes:
            config.defect_probabilities[defect_name] = (
                1.0 if self._panel.ic_defect_type_checkboxes[defect_name].isChecked() else 0.0
            )
            config.defect_severities[defect_name] = float(self._panel.ic_defect_type_spinboxes[defect_name].value()) / 100.0
        return config

    def _show_previous_sample(self) -> None:
        self._navigate_frame(-1)

    def _show_next_sample(self) -> None:
        self._navigate_frame(1)

    def _resample_current_sample(self) -> None:
        self._resample_salt += 1
        self._refresh_preview()

    def _augmentation_multiplier_value(self) -> float:
        return max(
            MIN_AUGMENTATION_MULTIPLIER,
            min(MAX_AUGMENTATION_MULTIPLIER, float(self._panel.augmentation_multiplier_spinbox.value())),
        )

    def _augmentation_slots(self) -> int:
        multiplier = self._augmentation_multiplier_value()
        if multiplier <= 0.0:
            return 1
        return int(round(multiplier)) + 1

    def _generation_settings_for_cutter(self) -> object:
        generation = self._training_parameters.generation
        operations = self._panel.get_training_augmentation_config()
        return replace(
            generation,
            horizontal_rotation=False,
            vertical_rotation=False,
            flip_x=False,
            flip_y=False,
            additional_augmentation=False,
            augmentation_multiplier=0.0,
            augmentation_brightness_strength=float(self._panel.augmentation_brightness_spinbox.value()),
            augmentation_brightness_enabled=bool(operations['brightness']['enabled']),
            augmentation_brightness_probability=float(operations['brightness']['probability']),
            augmentation_contrast_strength=float(self._panel.augmentation_contrast_spinbox.value()),
            augmentation_contrast_enabled=bool(operations['contrast']['enabled']),
            augmentation_contrast_probability=float(operations['contrast']['probability']),
            augmentation_gamma_strength=float(self._panel.augmentation_gamma_spinbox.value()),
            augmentation_gamma_enabled=bool(operations['gamma']['enabled']),
            augmentation_gamma_probability=float(operations['gamma']['probability']),
            augmentation_noise_enabled=bool(operations['noise']['enabled']),
            augmentation_noise_probability=float(self._panel.augmentation_noise_probability_spinbox.value()),
            augmentation_noise_sigma=float(self._panel.augmentation_noise_sigma_spinbox.value()),
            augmentation_blur_probability=float(self._panel.augmentation_blur_probability_spinbox.value()),
            augmentation_blur_enabled=bool(operations['blur']['enabled']),
            augmentation_blur_radius=float(self._panel.augmentation_blur_radius_spinbox.value()),
            random_crop=bool(self._panel.random_crop_check_box.isChecked()),
            crops_per_image=int(self._panel.crops_per_image_spinbox.value()),
            scale_augmentation=bool(self._panel.scale_augmentation_check_box.isChecked()),
            scale_augmentation_probability=float(operations['scale']['probability']),
            scale_augmentation_strength=float(self._panel.scale_augmentation_strength_spinbox.value()),
        )

    def _cutter_length_for_source(self, sample_index: int) -> int:
        cached = self._cutter_length_cache.get(sample_index)
        if cached is not None:
            return cached
        if self.full_image_check_box.isChecked():
            length = 1
        else:
            raw_image, raw_label = self._load_prepared_arrays(sample_index)
            cutter = SampleFastCutter(
                (raw_image, raw_label),
                self._generation_settings_for_cutter(),
                shuffle=False,
            )
            length = max(1, len(cutter))
        self._cutter_length_cache[sample_index] = length
        return length

    def _rebuild_frame_plan(self) -> None:
        previous_coords = self._frame_plan[self._current_frame_index] if self._frame_plan else None
        self._frame_plan = []
        if not self._sample_pairs:
            self._current_frame_index = 0
            return
        aug_slots = self._augmentation_slots()
        for source_index in range(len(self._sample_pairs)):
            cutter_length = self._cutter_length_for_source(source_index)
            for cutter_item in range(cutter_length):
                for aug_variant in range(aug_slots):
                    self._frame_plan.append((source_index, cutter_item, aug_variant))
        if not self._frame_plan:
            self._current_frame_index = 0
            return
        if previous_coords is not None and previous_coords in self._frame_plan:
            self._current_frame_index = self._frame_plan.index(previous_coords)
        else:
            self._current_frame_index = min(self._current_frame_index, len(self._frame_plan) - 1)

    def _apply_frame_coords(self, source_index: int, cutter_item: int, aug_variant: int) -> None:
        self._current_sample_index = int(source_index)
        self._variant_serial = int(aug_variant)
        self._cutter_item_index = int(cutter_item)

    def _current_frame_coords(self) -> tuple[int, int, int]:
        if not self._frame_plan:
            return self._current_sample_index, 0, self._variant_serial
        source_index, cutter_item, aug_variant = self._frame_plan[self._current_frame_index]
        self._apply_frame_coords(source_index, cutter_item, aug_variant)
        return source_index, cutter_item, aug_variant

    def _navigate_frame(self, delta: int) -> bool:
        if self._panel.synthetic_defect_generator_check_box.isChecked():
            return False
        self._rebuild_frame_plan()
        if not self._frame_plan:
            return False
        self._current_frame_index = (self._current_frame_index + int(delta)) % len(self._frame_plan)
        source_index, _cutter_item, _aug_variant = self._current_frame_coords()
        self._sync_sample_list_selection()
        self._refresh_preview()
        return True

    def _photometric_effect_enabled(self, spinbox: QWidget) -> bool:
        controls = {
            self._panel.augmentation_brightness_spinbox: self._panel.brightness_augmentation_check_box,
            self._panel.augmentation_contrast_spinbox: self._panel.contrast_augmentation_check_box,
            self._panel.augmentation_gamma_spinbox: self._panel.gamma_augmentation_check_box,
            self._panel.augmentation_noise_probability_spinbox: self._panel.noise_augmentation_check_box,
            self._panel.augmentation_blur_probability_spinbox: self._panel.blur_augmentation_check_box,
        }
        return controls[spinbox].isChecked() and float(spinbox.value()) > 0.0

    def _tech_effect_enabled(self, spinbox: QWidget) -> bool:
        if not self._panel.tech_augmentation_check_box.isChecked():
            return False
        return float(spinbox.value()) > 0.0

    def _show_original_preview(self) -> None:
        self._show_augmented = False
        self._update_preview_mode_label()
        self._update_visible_preview()

    def _restore_augmented_preview(self) -> None:
        self._show_augmented = True
        self._update_preview_mode_label()
        self._update_visible_preview()
