"""augmentation_preview_dialog: resolve setting text."""
from __future__ import annotations
import numpy as np
from PIL import Image
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)
from neuralimage.lib.images import ImagePreparator, SampleFastCutter
from neuralimage.lib.rare_patch_masks import (
    prepare_cif_label_raster,
)
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.view.settings_panel_widgets import NoWheelComboBox

from neuralimage.view.augmentation_preview_dialog_parts.module_state import (
    PREVIEW_VALUE_LABELS_EN,
    PREVIEW_VALUE_LABELS_RU,
)


class ResolveSettingTextMixin:
    def _update_preview_mode_label(self) -> None:
        mode_key = 'mode_augmented' if self._show_augmented else 'mode_original'
        self.mode_label.setText(
            str(
                self._texts.get(
                    'mode_hold_template',
                    'Mode: {mode}. Preview: {preview_mode}. Hold the middle mouse button to inspect the original image and label.',
                )
            ).format(
                mode=str(self._texts.get(mode_key, 'Augmented')),
                preview_mode=self._preview_mode_text(),
            )
        )

    def _update_visible_preview(self) -> None:
        if self._show_augmented:
            image_array = self._augmented_image_array
            label_array = self._augmented_label_array
        else:
            image_array = self._original_image_array
            label_array = self._original_label_array
        target_width = max(1, min(self.image_preview.width(), self.label_preview.width()))
        target_height = max(1, min(self.image_preview.height(), self.label_preview.height()))
        self._set_preview_image(self.image_preview, image_array, target_width=target_width, target_height=target_height)
        self._set_preview_image(self.label_preview, label_array, target_width=target_width, target_height=target_height)

    def _build_preview_arrays(self, sample_index):
        from neuralimage.remote.preview_qt import snapshot
        from neuralimage.remote.preview import compute_preview, decode_array
        payload = snapshot(self)
        payload["index"] = sample_index
        return tuple(decode_array(array) for array in compute_preview(payload))

    def _load_prepared_arrays(self, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._prepared_arrays_cache.get(sample_index)
        if cached is not None:
            self._prepared_arrays_cache.move_to_end(sample_index)
            return cached

        sample_path, label_path = self._sample_pairs[sample_index]
        prepared_image = ImagePreparator(sample_path, self._training_parameters.prepare).image
        if label_path.suffix.lower() == '.cif':
            raster_path, error_message = prepare_cif_label_raster(label_path)
            if raster_path is None:
                raise ValueError(error_message or f'Unable to rasterize {label_path.name}.')
            label_path = raster_path
        prepared_label = ImagePreparator(label_path, self._training_parameters.prepare).image
        prepared_label = prepared_label.convert('L')
        if prepared_label.size != prepared_image.size:
            prepared_label = prepared_label.resize(prepared_image.size, resample=Image.Resampling.NEAREST)
        prepared = (
            image_to_channel_first_float01(prepared_image, self._training_parameters.colors),
            SampleFastCutter.get_matrix_from_image(prepared_label, 1),
        )
        self._prepared_arrays_cache[sample_index] = prepared
        self._prepared_arrays_cache.move_to_end(sample_index)
        while len(self._prepared_arrays_cache) > 2:
            self._prepared_arrays_cache.popitem(last=False)
        return prepared

    def _has_selected_tech_variations(self) -> bool:
        if not self._panel.tech_augmentation_check_box.isChecked():
            return False
        return any(
            self._tech_effect_enabled(spinbox)
            for spinbox in (
                self._panel.tech_aug_global_width_probability_spinbox,
                self._panel.tech_aug_scale_rethreshold_probability_spinbox,
                self._panel.tech_aug_blur_threshold_probability_spinbox,
                self._panel.tech_aug_boundary_aware_probability_spinbox,
                self._panel.tech_aug_local_morphology_probability_spinbox,
                self._panel.tech_aug_gap_variation_probability_spinbox,
            )
        )

    def _preview_mode_text(self) -> str:
        if self.full_image_check_box.isChecked():
            return str(
                self._texts.get(
                    'preview_mode_full',
                    'Полный кадр' if self._is_russian_ui else 'Full image',
                )
            )
        return str(
            self._texts.get(
                'preview_mode_patch',
                'Патч' if self._is_russian_ui else 'Patch',
            )
        )

    @staticmethod
    def _set_combo_value(combo: NoWheelComboBox, value: str) -> None:
        normalized = str(value or '').strip().lower()
        index = combo.findData(normalized)
        if index < 0:
            index = combo.findText(normalized)
        if index < 0:
            index = 0
        combo.setCurrentIndex(index)

    def _get_synthetic_topology_domain(self) -> str:
        return str(
            self._panel.synthetic_topology_domain_combo.currentData()
            or self._panel.synthetic_topology_domain_combo.currentText()
            or 'pcb'
        ).strip().lower()

    def _get_synthetic_topology_family(self) -> str:
        combo = self._panel.ic_topology_family_combo if self._get_synthetic_topology_domain() == 'ic' else self._panel.pcb_topology_family_combo
        return str(combo.currentData() or combo.currentText() or '').strip().lower()

    def _resolve_text(self, key: str) -> str:
        if key == 'synthetic_topology':
            return str(
                self._texts.get(
                    key,
                    'Генерировать синтетическую топологию' if self._is_russian_ui else 'Generate synthetic topology',
                )
            )
        ic_fallbacks = {
            'ic_line_break': 'Обрыв линии' if self._is_russian_ui else 'Line break',
            'ic_bridge': 'Мостик' if self._is_russian_ui else 'Bridge',
            'ic_necking': 'Пережатие' if self._is_russian_ui else 'Necking',
            'ic_missing_metal': 'Потеря металла' if self._is_russian_ui else 'Missing metal',
            'ic_spur': 'Шпора' if self._is_russian_ui else 'Spur',
            'ic_pinhole': 'Pinhole' if self._is_russian_ui else 'Pinhole',
            'ic_via_open': 'Via open' if self._is_russian_ui else 'Via open',
            'ic_line_shift': 'Сдвиг линии' if self._is_russian_ui else 'Line shift',
        }
        if key in ic_fallbacks:
            return ic_fallbacks[key]
        return str(self._texts.get(key, self._settings_texts.get(key, key)))

    def _resolve_setting_text(self, key: str) -> str:
        fallback_labels = (
            PREVIEW_VALUE_LABELS_EN | PREVIEW_VALUE_LABELS_RU
            if self._is_russian_ui
            else PREVIEW_VALUE_LABELS_EN
        )
        if key == 'synthetic_background_noise_sigma':
            return str(fallback_labels.get(key, 'Background noise sigma'))
        if key == 'synthetic_topology_domain':
            return 'Домен синтетики' if self._is_russian_ui else 'Synthetic domain'
        if key == 'pcb_topology_family':
            return 'Семейство PCB-топологии' if self._is_russian_ui else 'PCB topology family'
        if key == 'ic_topology_family':
            return 'Семейство IC-топологии' if self._is_russian_ui else 'IC topology family'
        if key == 'synthetic_trace_count':
            return 'Количество трасс' if self._is_russian_ui else 'Trace count'
        if key == 'synthetic_segment_count':
            return 'Сегментов на трассу' if self._is_russian_ui else 'Segments per trace'
        if key == 'synthetic_trace_half_width':
            return 'Полуширина трассы' if self._is_russian_ui else 'Trace half-width'
        if key == 'ic_line_break_severity':
            return 'Сила обрыва линии' if self._is_russian_ui else 'Line break severity'
        if key == 'ic_bridge_severity':
            return 'Сила мостика' if self._is_russian_ui else 'Bridge severity'
        if key == 'ic_necking_severity':
            return 'Сила пережатия' if self._is_russian_ui else 'Necking severity'
        if key == 'ic_missing_metal_severity':
            return 'Сила потери металла' if self._is_russian_ui else 'Missing metal severity'
        if key == 'ic_spur_severity':
            return 'Сила шпоры' if self._is_russian_ui else 'Spur severity'
        if key == 'ic_pinhole_severity':
            return 'Сила pinhole' if self._is_russian_ui else 'Pinhole severity'
        if key == 'ic_via_open_severity':
            return 'Сила via open' if self._is_russian_ui else 'Via open severity'
        if key == 'ic_line_shift_severity':
            return 'Сила сдвига линии' if self._is_russian_ui else 'Line shift severity'
        return str(
            self._settings_texts.get(
                key,
                self._settings_form_labels.get(
                    key,
                    fallback_labels.get(
                        key,
                        self._texts.get(key, key),
                    ),
                ),
            )
        )
