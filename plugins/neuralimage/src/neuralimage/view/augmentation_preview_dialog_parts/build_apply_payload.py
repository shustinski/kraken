"""augmentation_preview_dialog: build apply payload."""
from __future__ import annotations
import copy
from dataclasses import asdict
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)
from neuralimage.lib.data_interfaces import (
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
)


class BuildApplyPayloadMixin:
    def _build_apply_payload(self) -> dict[str, object]:
        brightness_enabled = self._photometric_effect_enabled(self._panel.augmentation_brightness_spinbox)
        contrast_enabled = self._photometric_effect_enabled(self._panel.augmentation_contrast_spinbox)
        gamma_enabled = self._photometric_effect_enabled(self._panel.augmentation_gamma_spinbox)
        noise_enabled = self._photometric_effect_enabled(self._panel.augmentation_noise_probability_spinbox)
        blur_enabled = self._photometric_effect_enabled(self._panel.augmentation_blur_probability_spinbox)

        sem_config = self._current_sem_config()
        return {
            'horizontal_rotation': self._panel.horizontal_rotation.isChecked(),
            'vertical_rotation': self._panel.vertical_rotation.isChecked(),
            'flip_x': self._panel.flip_x.isChecked(),
            'flip_y': self._panel.flip_y.isChecked(),
            'random_crop': self._panel.random_crop_check_box.isChecked(),
            'step': int(self._panel.shift_spinbox.value()),
            'crops_per_image': int(self._panel.crops_per_image_spinbox.value()),
            'augmentation_multiplier': float(self._augmentation_multiplier_value()),
            'scale_augmentation': self._panel.scale_augmentation_check_box.isChecked(),
            'scale_augmentation_strength': float(self._panel.scale_augmentation_strength_spinbox.value()),
            'additional_augmentation': any(
                (
                    brightness_enabled,
                    contrast_enabled,
                    gamma_enabled,
                    noise_enabled,
                    blur_enabled,
                )
            ),
            'augmentation_brightness_strength': (
                float(self._panel.augmentation_brightness_spinbox.value()) if brightness_enabled else 0.0
            ),
            'augmentation_contrast_strength': (
                float(self._panel.augmentation_contrast_spinbox.value()) if contrast_enabled else 0.0
            ),
            'augmentation_gamma_strength': (
                float(self._panel.augmentation_gamma_spinbox.value()) if gamma_enabled else 0.0
            ),
            'augmentation_noise_probability': (
                float(self._panel.augmentation_noise_probability_spinbox.value()) if noise_enabled else 0.0
            ),
            'augmentation_noise_sigma': (
                float(self._panel.augmentation_noise_sigma_spinbox.value()) if noise_enabled else 0.0
            ),
            'augmentation_blur_probability': (
                float(self._panel.augmentation_blur_probability_spinbox.value()) if blur_enabled else 0.0
            ),
            'augmentation_blur_radius': (
                float(self._panel.augmentation_blur_radius_spinbox.value()) if blur_enabled else 0.0
            ),
            'synthetic_defect_generator': self._build_apply_synthetic_defect_generator_config(),
            'tech_aug': self._build_apply_tech_aug_config(),
            'preprocessing': asdict(sem_config.preprocessing),
            'sem_augmentation': asdict(sem_config.augmentation),
            'training_augmentation': self._panel.get_training_augmentation_config(),
            'cutout_enabled': self._panel.cutout_check_box.isChecked(),
            'cutout_probability': float(self._panel.cutout_probability_spinbox.value()),
            'cutout_holes': int(self._panel.cutout_holes_spinbox.value()),
            'cutout_size_ratio': float(self._panel.cutout_size_ratio_spinbox.value()),
            'random_artifacts_enabled': self._panel.random_artifacts_check_box.isChecked(),
            'random_artifacts_probability': float(self._panel.random_artifacts_probability_spinbox.value()),
            'random_artifacts_count': int(self._panel.random_artifacts_count_spinbox.value()),
            'random_artifacts_size_ratio': float(self._panel.random_artifacts_size_ratio_spinbox.value()),
            'random_artifacts_dust_enabled': self._panel.random_artifact_type_checkboxes["dust"].isChecked(),
            'random_artifacts_resist_residue_enabled': self._panel.random_artifact_type_checkboxes["resist_residue"].isChecked(),
            'random_artifacts_etch_residue_enabled': self._panel.random_artifact_type_checkboxes["etch_residue"].isChecked(),
            'random_artifacts_particle_cluster_enabled': self._panel.random_artifact_type_checkboxes["particle_cluster"].isChecked(),
            'random_artifacts_flake_enabled': self._panel.random_artifact_type_checkboxes["flake"].isChecked(),
            'mixup_enabled': self._panel.mixup_check_box.isChecked(),
            'mixup_probability': float(self._panel.mixup_probability_spinbox.value()),
            'mixup_alpha': float(self._panel.mixup_alpha_spinbox.value()),
            'pcb_defects': build_pcb_defect_parameters(None),
        }

    def _build_apply_synthetic_defect_generator_config(self):
        config = copy.deepcopy(
            build_synthetic_defect_generator_parameters(
                getattr(self._training_parameters, 'synthetic_defect_generator', None)
            )
        )
        config.enabled = self._panel.synthetic_defect_generator_check_box.isChecked()
        config.epoch_size_factor = float(self._panel.synthetic_dataset_factor_spinbox.value())
        config.topology_domain = self._get_synthetic_topology_domain()
        config.topology_family = self._get_synthetic_topology_family()
        config.image_size_xy = (
            int(self._panel.synthetic_image_width_spinbox.value()),
            int(self._panel.synthetic_image_height_spinbox.value()),
        )
        config.trace_count_range = tuple(
            sorted(
                (
                    int(self._panel.synthetic_trace_count_min_spinbox.value()),
                    int(self._panel.synthetic_trace_count_max_spinbox.value()),
                )
            )
        )
        config.segment_count_range = tuple(
            sorted(
                (
                    int(self._panel.synthetic_segment_count_min_spinbox.value()),
                    int(self._panel.synthetic_segment_count_max_spinbox.value()),
                )
            )
        )
        config.trace_half_width_range = tuple(
            sorted(
                (
                    int(self._panel.synthetic_trace_half_width_min_spinbox.value()),
                    int(self._panel.synthetic_trace_half_width_max_spinbox.value()),
                )
            )
        )
        config.background_noise_sigma_range = tuple(
            sorted(
                (
                    float(self._panel.synthetic_background_noise_sigma_min_spinbox.value()),
                    float(self._panel.synthetic_background_noise_sigma_max_spinbox.value()),
                )
            )
        )
        config.trace_noise_sigma_range = tuple(
            sorted(
                (
                    float(self._panel.synthetic_trace_noise_sigma_min_spinbox.value()),
                    float(self._panel.synthetic_trace_noise_sigma_max_spinbox.value()),
                )
            )
        )
        config.pcb_defects = self._build_apply_pcb_defects_config()
        config.ic_defects = self._build_apply_ic_defects_config()
        config.defects = config.ic_defects if config.topology_domain == 'ic' else config.pcb_defects
        return config

    def _build_apply_tech_aug_config(self, *, preview: bool = False):
        config = copy.deepcopy(build_tech_augmentation_config(getattr(self._training_parameters.generation, 'tech_aug', None)))
        min_operations = int(self._panel.tech_aug_min_operations_spinbox.value())
        max_operations = int(self._panel.tech_aug_max_operations_spinbox.value())
        if min_operations > max_operations:
            min_operations, max_operations = max_operations, min_operations
        config.enabled = self._has_selected_tech_variations()
        config.min_operations = min_operations
        config.max_operations = max_operations
        config.max_changed_pixels_ratio = float(self._panel.tech_aug_max_changed_pixels_ratio_spinbox.value())
        config.max_foreground_ratio_delta = float(self._panel.tech_aug_max_foreground_ratio_delta_spinbox.value())
        config.global_width.probability = (
            (1.0 if preview else float(self._panel.tech_aug_global_width_probability_spinbox.value()))
            if self._tech_effect_enabled(self._panel.tech_aug_global_width_probability_spinbox)
            else 0.0
        )
        config.scale_rethreshold.probability = (
            (1.0 if preview else float(self._panel.tech_aug_scale_rethreshold_probability_spinbox.value()))
            if self._tech_effect_enabled(self._panel.tech_aug_scale_rethreshold_probability_spinbox)
            else 0.0
        )
        config.blur_threshold.probability = (
            (1.0 if preview else float(self._panel.tech_aug_blur_threshold_probability_spinbox.value()))
            if self._tech_effect_enabled(self._panel.tech_aug_blur_threshold_probability_spinbox)
            else 0.0
        )
        config.boundary_aware.probability = (
            (1.0 if preview else float(self._panel.tech_aug_boundary_aware_probability_spinbox.value()))
            if self._tech_effect_enabled(self._panel.tech_aug_boundary_aware_probability_spinbox)
            else 0.0
        )
        config.local_morphology.probability = (
            (1.0 if preview else float(self._panel.tech_aug_local_morphology_probability_spinbox.value()))
            if self._tech_effect_enabled(self._panel.tech_aug_local_morphology_probability_spinbox)
            else 0.0
        )
        config.gap_variation.probability = (
            (1.0 if preview else float(self._panel.tech_aug_gap_variation_probability_spinbox.value()))
            if self._tech_effect_enabled(self._panel.tech_aug_gap_variation_probability_spinbox)
            else 0.0
        )
        return config
