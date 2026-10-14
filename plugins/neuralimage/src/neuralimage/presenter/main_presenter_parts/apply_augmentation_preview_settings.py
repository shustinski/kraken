"""main_presenter: apply augmentation preview settings."""
from __future__ import annotations


class ApplyAugmentationPreviewSettingsMixin:
    def _apply_augmentation_preview_settings(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        panel = self.__dict__.get('settings_panel')
        if panel is None:
            return

        panel.horizontal_rotation.setChecked(bool(payload.get('horizontal_rotation', False)))
        panel.vertical_rotation.setChecked(bool(payload.get('vertical_rotation', False)))
        panel.flip_x.setChecked(bool(payload.get('flip_x', False)))
        panel.flip_y.setChecked(bool(payload.get('flip_y', False)))
        panel.random_crop_check_box.setChecked(bool(payload.get('random_crop', False)))
        panel.shift_spinbox.setValue(int(payload.get('step', panel.shift_spinbox.value())))
        panel.crops_per_image_spinbox.setValue(int(payload.get('crops_per_image', 64)))
        panel.augmentation_multiplier_spinbox.setValue(float(payload.get('augmentation_multiplier', 0.0)))
        panel.scale_augmentation_check_box.setChecked(bool(payload.get('scale_augmentation', False)))
        panel.scale_augmentation_strength_spinbox.setValue(float(payload.get('scale_augmentation_strength', 0.2)))
        panel.additional_augmentation_check_box.setChecked(bool(payload.get('additional_augmentation', False)))
        panel.augmentation_brightness_spinbox.setValue(float(payload.get('augmentation_brightness_strength', 0.0)))
        panel.augmentation_contrast_spinbox.setValue(float(payload.get('augmentation_contrast_strength', 0.0)))
        panel.augmentation_gamma_spinbox.setValue(float(payload.get('augmentation_gamma_strength', 0.0)))
        panel.augmentation_noise_probability_spinbox.setValue(float(payload.get('augmentation_noise_probability', 0.0)))
        panel.augmentation_noise_sigma_spinbox.setValue(float(payload.get('augmentation_noise_sigma', 0.0)))
        panel.augmentation_blur_probability_spinbox.setValue(float(payload.get('augmentation_blur_probability', 0.0)))
        panel.augmentation_blur_radius_spinbox.setValue(float(payload.get('augmentation_blur_radius', 0.0)))
        panel.set_training_augmentation_config(payload.get('training_augmentation', {}))
        if hasattr(panel, 'set_synthetic_defect_generator_config'):
            panel.set_synthetic_defect_generator_config(payload.get('synthetic_defect_generator', {}))
        if hasattr(panel, 'set_tech_aug_config'):
            panel.set_tech_aug_config(payload.get('tech_aug', {}))
        if hasattr(panel, 'get_sem_segmentation_config') and hasattr(panel, 'set_sem_segmentation_config'):
            sem_config = panel.get_sem_segmentation_config()
            sem_config['preprocessing'] = dict(payload.get('preprocessing', {}))
            sem_config['augmentation'] = dict(payload.get('sem_augmentation', {}))
            panel.set_sem_segmentation_config(sem_config)

        panel.cutout_check_box.setChecked(bool(payload.get('cutout_enabled', False)))
        panel.cutout_probability_spinbox.setValue(float(payload.get('cutout_probability', 1.0)))
        panel.cutout_holes_spinbox.setValue(int(payload.get('cutout_holes', 1)))
        panel.cutout_size_ratio_spinbox.setValue(float(payload.get('cutout_size_ratio', 0.25)))

        panel.random_artifacts_check_box.setChecked(bool(payload.get('random_artifacts_enabled', False)))
        panel.random_artifacts_probability_spinbox.setValue(float(payload.get('random_artifacts_probability', 1.0)))
        panel.random_artifacts_count_spinbox.setValue(int(payload.get('random_artifacts_count', 1)))
        panel.random_artifacts_size_ratio_spinbox.setValue(float(payload.get('random_artifacts_size_ratio', 0.25)))
        for artifact_name, checkbox in getattr(panel, 'random_artifact_type_checkboxes', {}).items():
            checkbox.setChecked(bool(payload.get(f'random_artifacts_{artifact_name}_enabled', True)))

        panel.mixup_check_box.setChecked(bool(payload.get('mixup_enabled', False)))
        panel.mixup_probability_spinbox.setValue(float(payload.get('mixup_probability', 1.0)))
        panel.mixup_alpha_spinbox.setValue(float(payload.get('mixup_alpha', 0.2)))
        if hasattr(panel, '_sync_augmentation_controls'):
            panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
        if hasattr(panel, '_sync_training_augmentation_controls'):
            panel._sync_training_augmentation_controls()

        update_settings = self.__dict__.get('_update_settings_window_state')
        if callable(update_settings):
            update_settings()
            return
        type(self)._update_settings_window_state(self)
