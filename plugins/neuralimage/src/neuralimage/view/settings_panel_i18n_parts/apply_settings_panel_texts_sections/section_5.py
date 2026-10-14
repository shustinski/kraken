"""apply_settings_panel_texts section 5."""
from __future__ import annotations
from typing import Any
from neuralimage.configuration import (
    SEM_UI_SECTIONS,
    sem_ui_section_help,
    sem_ui_section_label,
)
from neuralimage.lib.ui_texts import get_ui_language
from neuralimage.view.settings_panel_i18n_parts.module_state import (
    TEXT_FIELD_LOG_UPDATE_FREQUENCY,
    TEXT_FIELD_RECOGNITION_JPEG_QUALITY,
)
from neuralimage.view.settings_panel_i18n_parts.apply_description_tooltips import (
    _copy_text_dict,
    _read_text_from_mappings,
    apply_description_tooltips,
)

def _apply_settings_panel_texts_section_5(panel, descriptions, labels_map, t):
    panel.recognition_tta_check_box.setToolTip(
        str(
            t.get(
                'recognition_tta_tip',
                'Apply test-time augmentation to the main recognition mask during inference.',
            )
        )
    )
    panel.confidence_save_mode_combo.setToolTip(
        str(
            t.get(
                'confidence_save_mode_tip',
                'Choose whether to skip confidence output or save it from the model output or TTA result.',
            )
        )
    )
    panel.confidence_save_mode_combo.setItemText(
        0,
        str(t.get('confidence_save_mode_off', 'Do not save')),
    )
    panel.confidence_save_mode_combo.setItemText(
        1,
        str(t.get('confidence_save_mode_model_output', 'From model output')),
    )
    panel.confidence_save_mode_combo.setItemText(
        2,
        str(t.get('confidence_save_mode_tta', 'From TTA')),
    )
    panel.recognition_postprocess_check_box.setText(
        str(t.get('recognition_postprocess', 'Postprocess binary mask'))
    )
    panel.recognition_postprocess_check_box.setToolTip(
        str(t.get('recognition_postprocess_tip', 'Apply simple morphology to clean up the binary mask.'))
    )
    panel.recognition_postprocess_kernel_size_spinbox.setToolTip(
        str(
            t.get(
                'recognition_postprocess_kernel_size_tip',
                'Odd kernel size used for binary mask cleanup after thresholding.',
            )
        )
    )
    panel.log_update_frequency_spinbox.setToolTip(
        str(t.get('log_update_frequency_tip', 'Log update frequency in batches (0 = auto).'))
    )
    panel.mixed_precision_type.setToolTip(str(t.get('mixed_precision_tip', '')))
    panel.deep_supervision_check_box.setText(str(t.get('deep_supervision', 'Enable deep supervision')))
    panel.deep_supervision_check_box.setToolTip(
        str(
            t.get(
                'deep_supervision_tip',
                'Train convolutional segmentation models with auxiliary decoder heads.',
            )
        )
    )
    panel.topograph_term_label.setText(str(t.get('topograph_loss_weight', 'Topograph')))
    panel.topograph_term_label.setToolTip(
        str(
            t.get(
                'topograph_enabled_tip',
                'Add an optional topology-preserving loss term on top of the selected mask losses. Binary segmentation only.',
            )
        )
    )
    panel.topograph_enabled_check_box.setToolTip(panel.topograph_term_label.toolTip())
    panel.topograph_loss_weight_spinbox.setToolTip(
        str(
            t.get(
                'topograph_loss_weight_tip',
                'Additive weight for the Topograph loss term. It is added on top of the configured BCE/Dice terms.',
            )
        )
    )
    panel.topograph_debug_viz_label.setText(
        str(t.get('topograph_debug_viz', 'Show Topograph critical regions'))
    )
    panel.topograph_debug_viz_label.setToolTip(
        str(
            t.get(
                'topograph_debug_viz_tip',
                'Overlay topologically critical regions on the training batch preview output.',
            )
        )
    )
    panel.topograph_debug_viz_check_box.setToolTip(panel.topograph_debug_viz_label.toolTip())
    loss_tip = str(t.get('loss_function_tip', ''))
    loss_group_title = str(labels_map.get('loss_function', t.get('loss_function', 'Loss terms')))
    panel.loss_terms_groupbox.setTitle(loss_group_title)
    panel.loss_terms_groupbox.setToolTip(loss_tip)
    panel.loss_terms_widget.setToolTip(loss_tip)
    panel.loss_formula_label.setToolTip(loss_tip)
    if 'conductors' in getattr(panel, 'loss_preset_buttons', {}):
        panel.loss_preset_buttons['conductors'].setText(str(t.get('loss_preset_conductors', 'Conductors')))
        panel.loss_preset_buttons['conductors'].setToolTip(
            str(
                t.get(
                    'loss_preset_conductors_tip',
                    'Preset: 0.5 BCE + 0.5 Dice.',
                )
            )
        )
    if 'contacts' in getattr(panel, 'loss_preset_buttons', {}):
        panel.loss_preset_buttons['contacts'].setText(str(t.get('loss_preset_contacts', 'Contacts')))
        panel.loss_preset_buttons['contacts'].setToolTip(
            str(
                t.get(
                    'loss_preset_contacts_tip',
                    'Preset: 0.5 BCE + 0.5 Focal Tversky.',
                )
            )
        )
    if hasattr(panel, 'loss_presets_widget'):
        panel.loss_presets_widget.setToolTip(loss_tip)
    for loss_name in getattr(panel, 'loss_term_checkboxes', {}):
        panel.loss_term_checkboxes[loss_name].setToolTip(loss_tip)
        panel.loss_term_spinboxes[loss_name].setToolTip(loss_tip)
        panel.loss_term_labels[loss_name].setToolTip(loss_tip)
    panel.multi_gpu_mode_combo.setToolTip(str(t.get('multi_gpu_tip', '')))
    panel.torch_compile_check_box.setText(str(t.get('torch_compile_enable', 'Enable torch.compile')))
    panel.torch_compile_check_box.setToolTip(
        str(t.get('torch_compile_tip', 'Compile PyTorch graph for speedup when supported.'))
    )
    panel.skip_uniform_labels_check_box.setText(
        str(t.get('skip_uniform_labels', 'Skip samples where label is all 0 or all 1'))
    )
    panel.skip_uniform_labels_check_box.setToolTip(str(t.get('skip_uniform_labels_tip', '')))
    panel.rare_patch_oversampling_check_box.setText(
        str(
            t.get(
                'rare_patch_oversampling_enable',
                'Oversample patches with selected rare regions',
            )
        )
    )
    panel.rare_patch_oversampling_check_box.setToolTip(
        str(
            t.get(
                'rare_patch_oversampling_tip',
                'In online mode, repeats patches that intersect manually selected rare regions.',
            )
        )
    )
    panel.rare_patch_oversampling_factor_spinbox.setToolTip(
        str(
            t.get(
                'rare_patch_oversampling_factor_tip',
                'How many times to repeat patches that intersect selected rare regions.',
            )
        )
    )
    panel.edit_rare_regions_button.setText(str(t.get('edit_rare_regions', 'Edit rare regions')))
    panel.edit_rare_regions_button.setToolTip(
        str(
            t.get(
                'edit_rare_regions_tip',
                'Open the editor for manual selection of rare and difficult regions.',
            )
        )
    )
    panel.warmup_check_box.setText(str(t.get('warmup_enable', 'Enable warmup')))
    panel.warmup_check_box.setToolTip(str(t.get('warmup_tip', '')))
    panel.hard_mining_check_box.setText(str(t.get('hard_mining_enable', 'Train more on hard samples (high loss)')))
    panel.hard_mining_check_box.setToolTip(str(t.get('hard_mining_tip', '')))
    panel.hard_pixel_mining_check_box.setText(
        str(t.get('hard_pixel_mining_enable', 'Focus loss on hardest pixels'))
    )
    panel.hard_pixel_mining_check_box.setToolTip(str(t.get('hard_pixel_mining_tip', '')))
    panel.early_stopping_check_box.setText(str(t.get('early_stopping_enable', 'Enable early stopping')))
    panel.early_stopping_check_box.setToolTip(str(t.get('early_stopping_tip', '')))
    panel.reset_defaults_button.setText(str(t.get('reset_defaults', 'Reset defaults')))
    panel.reset_defaults_button.setToolTip(str(t.get('reset_defaults_tip', 'Reset all parameters to initial values.')))
    color_modes = t.get('color_modes', {'RGB': 'RGB', 'ЧБ': 'ЧБ'})
    if isinstance(color_modes, dict) and color_modes:
        panel.set_color_mode_items([(str(value), str(label)) for value, label in color_modes.items()])
    elif isinstance(color_modes, list) and color_modes:
        panel.set_color_mode_items([(str(v), str(v)) for v in color_modes])
    apply_description_tooltips(panel, descriptions, labels_map)
    panel._sync_validation_path_labels()
    if hasattr(panel, '_rebuild_settings_registry') and hasattr(panel, 'settings_search_model'):
        panel._rebuild_settings_registry()
