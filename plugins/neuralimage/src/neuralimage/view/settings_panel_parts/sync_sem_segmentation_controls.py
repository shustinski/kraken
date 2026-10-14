"""settings_panel: sync sem segmentation controls."""
from __future__ import annotations
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )


class SyncSemSegmentationControlsMixin:
    def _sync_sem_segmentation_controls(self) -> None:
        controls = self.sem_segmentation_controls

        def _set_sem_visible(key: str, visible: bool) -> None:
            control = controls.get(key)
            label = self.sem_segmentation_field_labels.get(key)
            if control is not None:
                container = control.parentWidget()
                (container or control).setVisible(visible)
            if label is not None:
                label.setVisible(visible)

        for target_name in (
            'boundary', 'skeleton', 'sdf', 'distance_transform', 'thickness',
            'vertex', 'corner', 'endpoint', 'junction', 'orientation', 'tangent',
            'curvature', 'topology',
        ):
            enabled_control = controls.get(f'target_{target_name}')
            weight_control = controls.get(f'weight_{target_name}')
            if enabled_control is not None and weight_control is not None:
                weight_control.setEnabled(bool(enabled_control.isChecked()))

        sdf_control = controls.get('target_sdf')
        distance_boundary_weight = controls.get('target_distance_boundary_weight')
        if sdf_control is not None and distance_boundary_weight is not None:
            distance_boundary_weight.setEnabled(bool(sdf_control.isChecked()))

        dependency_groups = (
            ('aug_enabled', tuple(key for key in controls if key.startswith('aug_') and key != 'aug_enabled')),
            ('context_enabled', tuple(key for key in controls if key.startswith('context_') and key != 'context_enabled')),
            ('al_enabled', tuple(key for key in controls if key.startswith('al_') and key != 'al_enabled')),
            ('validation_enabled', tuple(key for key in controls if key.startswith('validation_') and key != 'validation_enabled')),
        )
        for parent_key, child_keys in dependency_groups:
            parent = controls.get(parent_key)
            if parent is None:
                continue
            enabled = (
                bool(self.validation_check_box.isChecked())
                if parent_key == 'validation_enabled'
                else bool(parent.isChecked())
            )
            if parent_key == 'aug_enabled':
                training_editor = getattr(self, 'training_augmentation_editor', None)
                if training_editor is not None:
                    training_editor.sync_from_controls()
                    continue
                augmentation_editor = self.sem_segmentation_section_editors.get('augmentation')
                if augmentation_editor is not None and hasattr(augmentation_editor, '_sync_effect_rows'):
                    if hasattr(augmentation_editor, '_plan_row'):
                        augmentation_editor._plan_row.setEnabled(enabled)
                    if enabled:
                        augmentation_editor._sync_effect_rows()
                    else:
                        for child_key in child_keys:
                            controls[child_key].setEnabled(False)
                            _set_sem_visible(child_key, False)
                    continue
            for child_key in child_keys:
                controls[child_key].setEnabled(enabled)
                _set_sem_visible(child_key, enabled)

        confidence_enabled = bool(controls['confidence_training_enabled'].isChecked())
        controls['confidence_training_loss_weight'].setEnabled(confidence_enabled)
        _set_sem_visible('confidence_training_loss_weight', confidence_enabled)

        uncertainty_enabled = bool(controls['uncertainty_enabled'].isChecked())
        method = str(controls['uncertainty_method'].currentData() or 'confidence_head')
        for key in ('uncertainty_method', 'uncertainty_export'):
            controls[key].setEnabled(uncertainty_enabled)
            _set_sem_visible(key, uncertainty_enabled)
        for key in ('uncertainty_samples', 'uncertainty_rate'):
            visible = uncertainty_enabled and method in {'mc_dropout', 'combined', 'auto'}
            controls[key].setEnabled(visible)
            _set_sem_visible(key, visible)
        for key in ('uncertainty_tta_flips', 'uncertainty_tta_rotations'):
            visible = uncertainty_enabled and method in {'tta_variance', 'combined', 'auto'}
            controls[key].setEnabled(visible)
            _set_sem_visible(key, visible)

        hard_mode = controls.get('hard_mode')
        if hard_mode is not None:
            enabled = str(hard_mode.currentData()) != 'off'
            for key, control in controls.items():
                if key.startswith('hard_') and key != 'hard_mode':
                    control.setEnabled(enabled)
                    _set_sem_visible(key, enabled)
        self._sync_legacy_uncertainty_adapters_from_sem()
