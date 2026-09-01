from __future__ import annotations

import copy
import random
import zlib
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import replace

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QEvent, QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from neuralimage.augmentations import (
    ICDefectAugmentor,
    PCBDefectAugmentor,
    SyntheticTopologyGenerator,
    SyntheticTopologyParameters,
    TechVariationAugmentor,
)
from neuralimage.lib.data_interfaces import (
    IC_TOPOLOGY_FAMILIES,
    PCB_TOPOLOGY_FAMILIES,
    TrainingParameters,
    build_ic_defect_parameters,
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
)
from neuralimage.lib.images import ImagePreparator, SampleFastCutter, resolved_augmentation_variant_count
from neuralimage.lib.random_artifacts import generate_random_artifact_patch
from neuralimage.lib.rare_patch_masks import (
    collect_matching_sample_label_pairs,
    prepare_cif_label_raster,
)
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.model.NeuralNetwork.dataset import _apply_binary_tech_augmentation_to_pair
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.targets.dataset_hooks import (
    apply_dataset_preprocessing,
    apply_dataset_sem_augmentation_preview,
)
from neuralimage.configuration import (
    build_sem_segmentation_config,
    sem_config_from_form_values,
    sem_config_to_form_values,
)
from neuralimage.view.sem_compact_section_editor import CompactSemSectionEditor
from neuralimage.view.settings_panel import SettingsPanel
from neuralimage.view.settings_panel_widgets import NoWheelComboBox

MIN_AUG_STRENGTH = 0.0
MAX_AUG_STRENGTH = 1.0
MIN_AUGMENTATION_PROBABILITY = 0.0
MAX_AUGMENTATION_PROBABILITY = 1.0
MIN_AUG_NOISE_SIGMA = 0.0
MAX_AUG_NOISE_SIGMA = 0.2
MIN_AUG_BLUR_RADIUS = 0.0
MAX_AUG_BLUR_RADIUS = 5.0
MIN_CROPS_PER_IMAGE = 1
MAX_CROPS_PER_IMAGE = 5000
MIN_TECH_AUG_OPERATIONS = 1
MAX_TECH_AUG_OPERATIONS = 6
SHIFT_RANGE_MIN = 4
SHIFT_RANGE_MAX = 2000
MIN_SYNTHETIC_DATASET_FACTOR = 0.0
MAX_SYNTHETIC_DATASET_FACTOR = 10.0
MIN_CUTOUT_HOLES = 1
MAX_CUTOUT_HOLES = 32
MIN_RANDOM_ARTIFACTS_COUNT = 1
MAX_RANDOM_ARTIFACTS_COUNT = 16
MIN_MIXUP_ALPHA = 0.0
MAX_MIXUP_ALPHA = 10.0
MIN_PCB_DEFECT_COUNT = 1
MAX_PCB_DEFECT_COUNT = 8
MIN_PCB_DEFECT_WEIGHT = 0.0
MAX_PCB_DEFECT_WEIGHT = 5.0
MIN_SYNTHETIC_TRACE_COUNT = 1
MAX_SYNTHETIC_TRACE_COUNT = 200
MIN_SYNTHETIC_IMAGE_SIZE = 64
MAX_SYNTHETIC_IMAGE_SIZE = 8192
MIN_SYNTHETIC_SEGMENT_COUNT = 1
MAX_SYNTHETIC_SEGMENT_COUNT = 24
MIN_SYNTHETIC_TRACE_HALF_WIDTH = 1
MAX_SYNTHETIC_TRACE_HALF_WIDTH = 50
MIN_SYNTHETIC_BACKGROUND_NOISE_SIGMA = 0.0
MAX_SYNTHETIC_BACKGROUND_NOISE_SIGMA = 0.2
MIN_SYNTHETIC_TRACE_NOISE_SIGMA = 0.0
MAX_SYNTHETIC_TRACE_NOISE_SIGMA = 0.2
MIN_AUGMENTATION_MULTIPLIER = 0.0
MAX_AUGMENTATION_MULTIPLIER = 20.0
IC_DEFECT_WEIGHT_FIELDS: tuple[tuple[str, str], ...] = (
    ('line_break', 'ic_line_break_severity'),
    ('bridge', 'ic_bridge_severity'),
    ('necking', 'ic_necking_severity'),
    ('missing_metal', 'ic_missing_metal_severity'),
    ('spur', 'ic_spur_severity'),
    ('pinhole', 'ic_pinhole_severity'),
    ('via_open', 'ic_via_open_severity'),
    ('line_shift', 'ic_line_shift_severity'),
)

PREVIEW_VALUE_LABELS_EN = {
    'shift': 'Patch shift',
    'crops_per_image': 'Crops per image',
    'scale_augmentation_strength': 'Scale strength',
    'augmentation_brightness_strength': 'Brightness change',
    'augmentation_contrast_strength': 'Contrast change',
    'augmentation_gamma_strength': 'Gamma change',
    'augmentation_noise_probability': 'Noise probability',
    'augmentation_noise_sigma': 'Noise strength',
    'augmentation_blur_probability': 'Blur probability',
    'augmentation_blur_radius': 'Blur radius',
    'tech_aug_min_operations': 'Min operations',
    'tech_aug_max_operations': 'Max operations',
    'tech_aug_max_changed_pixels_ratio': 'Changed pixels limit',
    'tech_aug_max_foreground_ratio_delta': 'Foreground ratio limit',
    'tech_aug_global_width_probability': 'Width variation probability',
    'tech_aug_scale_rethreshold_probability': 'Scale + rethreshold probability',
    'tech_aug_blur_threshold_probability': 'Blur + threshold probability',
    'tech_aug_boundary_aware_probability': 'Boundary-aware probability',
    'tech_aug_local_morphology_probability': 'Local morphology probability',
    'tech_aug_gap_variation_probability': 'Gap variation probability',
    'cutout_probability': 'Cutout probability',
    'cutout_holes': 'Cutout holes',
    'cutout_size_ratio': 'Cutout size ratio',
    'random_artifacts_probability': 'Artifacts probability',
    'random_artifacts_count': 'Artifacts count',
    'random_artifacts_size_ratio': 'Artifacts size ratio',
    'mixup_probability': 'Mixup probability',
    'mixup_alpha': 'Mixup alpha',
    'pcb_defects_probability': 'Defect probability',
    'pcb_defects_min_count': 'Min defects',
    'pcb_defects_max_count': 'Max defects',
    'synthetic_image_size': 'Synthetic image size',
    'synthetic_dataset_factor': 'Synthetic epoch factor',
    'synthetic_trace_count': 'Trace count',
    'synthetic_segment_count': 'Segments per trace',
    'synthetic_trace_half_width': 'Trace half-width',
    'synthetic_background_noise_sigma': 'Background noise sigma',
    'synthetic_trace_noise_sigma': 'Trace noise sigma',
    'pcb_break_severity': 'Break severity',
    'pcb_short_severity': 'Short severity',
    'pcb_missing_copper_severity': 'Missing copper severity',
    'pcb_excess_copper_severity': 'Excess copper severity',
    'pcb_pinhole_severity': 'Pinhole severity',
    'pcb_spurious_copper_severity': 'Spurious copper severity',
    'pcb_via_severity': 'Via defect severity',
    'pcb_misalignment_severity': 'Misalignment severity',
}
PREVIEW_VALUE_LABELS_RU = {
    'shift': 'Шаг нарезки',
    'crops_per_image': 'Фрагментов на изображение',
    'scale_augmentation_strength': 'Сила масштабирования',
    'augmentation_brightness_strength': 'Изменение яркости',
    'augmentation_contrast_strength': 'Изменение контраста',
    'augmentation_gamma_strength': 'Сила гаммы',
    'augmentation_noise_probability': 'Вероятность шума',
    'augmentation_noise_sigma': 'Сила шума',
    'augmentation_blur_probability': 'Вероятность размытия',
    'augmentation_blur_radius': 'Радиус размытия',
    'tech_aug_min_operations': 'Минимум операций',
    'tech_aug_max_operations': 'Максимум операций',
    'tech_aug_max_changed_pixels_ratio': 'Лимит изменённых пикселей',
    'tech_aug_max_foreground_ratio_delta': 'Лимит изменения foreground',
    'tech_aug_global_width_probability': 'Вероятность вариации ширины',
    'tech_aug_scale_rethreshold_probability': 'Вероятность scale + threshold',
    'tech_aug_blur_threshold_probability': 'Вероятность blur + threshold',
    'tech_aug_boundary_aware_probability': 'Вероятность пограничной вариации',
    'tech_aug_local_morphology_probability': 'Вероятность локальной морфологии',
    'tech_aug_gap_variation_probability': 'Вероятность вариации зазоров',
    'cutout_probability': 'Вероятность cutout',
    'cutout_holes': 'Количество областей cutout',
    'cutout_size_ratio': 'Размер области cutout',
    'random_artifacts_probability': 'Вероятность артефактов',
    'random_artifacts_count': 'Количество артефактов',
    'random_artifacts_size_ratio': 'Размер артефактов',
    'mixup_probability': 'Вероятность mixup',
    'mixup_alpha': 'Параметр alpha',
    'pcb_defects_probability': 'Вероятность дефектов',
    'pcb_defects_min_count': 'Минимум дефектов',
    'pcb_defects_max_count': 'Максимум дефектов',
    'synthetic_dataset_factor': 'Коэффициент synthetic-эпохи',
}


@contextmanager
def _seeded_random(seed: int):
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        torch.random.set_rng_state(torch_random_state)


class _PreviewLabel(QLabel):
    middle_pressed = pyqtSignal()
    middle_released = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.middle_pressed.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self.middle_released.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AugmentationPreviewDialog(QDialog):
    apply_to_main_requested = pyqtSignal(object)

    def __init__(
        self,
        training_parameters: TrainingParameters,
        settings_panel: SettingsPanel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._parent_window = parent
        self._panel = settings_panel
        self._training_parameters = training_parameters
        self._texts = get_ui_section('augmentation_preview_dialog')
        self._settings_texts = get_ui_section('settings_panel')
        self._is_russian_ui = any('\u0400' <= char <= '\u04FF' for char in str(self._texts.get('window_title', '')))
        settings_form = dict(self._settings_texts.get('settings_form', {}))
        self._settings_form_labels = dict(settings_form.get('labels', {}))
        self._settings_form_tooltips = dict(settings_form.get('tooltips', {}))
        self._sample_pairs, self._load_error = collect_matching_sample_label_pairs(
            training_parameters.image_path,
            training_parameters.label_path,
            strict=False,
            allow_cif_labels=True,
            recursive=bool(getattr(training_parameters, 'recursive_file_search', False)),
        )
        self._current_sample_index = 0
        self._current_frame_index = 0
        self._frame_plan: list[tuple[int, int, int]] = []
        self._cutter_length_cache: dict[int, int] = {}
        self._variant_serial = 0
        self._cutter_item_index = 0
        self._resample_salt = 0
        self._show_augmented = True
        self._sample_list_updating = False
        self._sidebar_restore: list[tuple[QWidget, QWidget | None, object, int | None]] = []
        self._original_image_array: np.ndarray | None = None
        self._augmented_image_array: np.ndarray | None = None
        self._original_label_array: np.ndarray | None = None
        self._augmented_label_array: np.ndarray | None = None
        self._prepared_arrays_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self.sem_normalization_editor = self._panel.sem_segmentation_section_editors['preprocessing']
        self.sem_augmentation_editor = self._panel.training_augmentation_editor
        self._build_ui()
        self.sample_list_widget.installEventFilter(self)
        self._connect_signals()
        self._sync_group_boxes()
        self._show_loading_state()
        # Paint the window before reading and augmenting a potentially large
        # SEM frame. The work still starts immediately on the next event turn.
        QTimer.singleShot(0, self._refresh_preview)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_visible_preview()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            if self._navigate_frame(-1 if key == Qt.Key.Key_Up else 1):
                event.accept()
                return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.sample_list_widget and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                if self._navigate_frame(-1 if key == Qt.Key.Key_Up else 1):
                    return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                return True
        return super().eventFilter(watched, event)

    def _build_ui(self) -> None:
        self.setWindowTitle(str(self._texts.get('window_title', 'Augmentation preview')))
        self.resize(1420, 860)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(6)
        self.prev_button = QPushButton(str(self._texts.get('prev_button', 'Previous')))
        self.next_button = QPushButton(str(self._texts.get('next_button', 'Next')))
        self.resample_button = QPushButton(str(self._texts.get('resample_button', 'Resample')))
        self.apply_to_main_button = QPushButton(
            str(
                self._texts.get(
                    'apply_to_main_button',
                    'Перенести в основное окно' if self._is_russian_ui else 'Apply to main window',
                )
            )
        )
        self.full_image_check_box = QCheckBox(
            str(
                self._texts.get(
                    'full_image_toggle',
                    'Показывать целиком' if self._is_russian_ui else 'Show full image',
                )
            )
        )
        self.full_image_check_box.setChecked(True)
        self.sample_label = QLabel('')
        self.sample_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        nav_row.addWidget(self.prev_button)
        nav_row.addWidget(self.next_button)
        nav_row.addWidget(self.resample_button)
        nav_row.addWidget(self.apply_to_main_button)
        nav_row.addWidget(self.full_image_check_box)
        nav_row.addWidget(self.sample_label, 1)
        left_layout.addLayout(nav_row)

        self.mode_label = QLabel('')
        self.mode_label.setWordWrap(True)
        left_layout.addWidget(self.mode_label)

        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(10)

        self.sample_list_group = QGroupBox(
            str(
                self._texts.get(
                    'sample_list_group',
                    'Изображения' if self._is_russian_ui else 'Images',
                )
            )
        )
        sample_list_layout = QVBoxLayout(self.sample_list_group)
        sample_list_layout.setContentsMargins(6, 6, 6, 6)
        self.sample_list_widget = QListWidget(self.sample_list_group)
        self.sample_list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.sample_list_widget.setMinimumWidth(220)
        sample_list_layout.addWidget(self.sample_list_widget)

        preview_widget = QWidget(self)
        preview_row = QHBoxLayout(preview_widget)
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(10)

        self.image_group = QGroupBox(str(self._texts.get('image_group', 'Image')))
        image_layout = QVBoxLayout(self.image_group)
        self.image_preview = _PreviewLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(420, 420)
        self.image_preview.setStyleSheet('border: 1px solid #666; background: #111;')
        image_layout.addWidget(self.image_preview)

        self.label_group = QGroupBox(str(self._texts.get('label_group', 'Label')))
        label_layout = QVBoxLayout(self.label_group)
        self.label_preview = _PreviewLabel()
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview.setMinimumSize(420, 420)
        self.label_preview.setStyleSheet('border: 1px solid #666; background: #111;')
        label_layout.addWidget(self.label_preview)

        preview_row.addWidget(self.image_group, 1)
        preview_row.addWidget(self.label_group, 1)
        content_row.addWidget(self.sample_list_group, 0)
        content_row.addWidget(preview_widget, 1)
        left_layout.addLayout(content_row, 1)

        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        right_scroll = QScrollArea(self)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(420)
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        normalization_group, augmentation_group = self._build_shared_training_transform_groups()
        self._attach_sidebar_widget(normalization_group, right_layout)
        self._attach_sidebar_widget(augmentation_group, right_layout)
        right_layout.addStretch(1)
        right_scroll.setWidget(right_content)

        self._populate_sample_list()
        root_layout.addWidget(left_widget, 1)
        root_layout.addWidget(right_scroll, 0)

    def _build_shared_training_transform_groups(self) -> tuple[QGroupBox, QGroupBox]:
        return self._panel.sem_normalization_editor, self._panel.training_augmentation_editor

    def _attach_sidebar_widget(self, widget: QGroupBox, layout: QVBoxLayout) -> None:
        old_parent = widget.parentWidget()
        old_layout = old_parent.layout() if old_parent is not None else None
        index: int | None = None
        if old_layout is not None:
            index = old_layout.indexOf(widget)
            if index >= 0:
                old_layout.removeWidget(widget)
        layout.addWidget(widget)
        self._sidebar_restore.append((widget, old_parent, old_layout, index))

    def closeEvent(self, event) -> None:
        for widget, old_parent, old_layout, index in reversed(self._sidebar_restore):
            current_parent = widget.parentWidget()
            if current_parent is not None:
                current_layout = current_parent.layout()
                if current_layout is not None:
                    current_layout.removeWidget(widget)
            if old_layout is not None and index is not None and index >= 0:
                widget.setParent(old_parent)
                if isinstance(old_layout, QVBoxLayout):
                    old_layout.insertWidget(index, widget)
                else:
                    old_layout.addWidget(widget)
            elif old_parent is not None:
                widget.setParent(old_parent)
        self._sidebar_restore.clear()
        super().closeEvent(event)

    def _current_sem_config(self):
        return build_sem_segmentation_config(self._panel.get_sem_segmentation_config())

    def _on_sem_pipeline_changed(self) -> None:
        self._prepared_arrays_cache.clear()
        self._cutter_length_cache.clear()
        self._sync_sem_pipeline_editor_visibility()
        self._rebuild_frame_plan()
        self._refresh_preview()

    def _sync_sem_pipeline_editor_visibility(self) -> None:
        self._panel._sync_sem_segmentation_controls()

    def _connect_signals(self) -> None:
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
            panel.pcb_defects_check_box,
            *panel.random_artifact_type_checkboxes.values(),
            *panel.pcb_defect_type_checkboxes.values(),
            *panel.ic_defect_type_checkboxes.values(),
        ):
            checkbox.toggled.connect(self._on_settings_changed)
        for widget in (
            panel.shift_spinbox,
            panel.crops_per_image_spinbox,
            panel.augmentation_multiplier_spinbox,
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
                widget.currentIndexChanged.connect(self._on_settings_changed)

    def _sync_group_boxes(self) -> None:
        panel = self._panel
        panel._sync_augmentation_controls(panel.additional_augmentation_check_box.isChecked())
        panel._sync_training_augmentation_controls()
        panel._sync_synthetic_defect_generator_controls(panel.synthetic_defect_generator_check_box.isChecked())
        panel._sync_sem_segmentation_controls()

    def _on_settings_changed(self, *_args: object) -> None:
        self._prepared_arrays_cache.clear()
        self._cutter_length_cache.clear()
        self._sync_group_boxes()
        self._rebuild_frame_plan()
        self._refresh_preview()

    def _on_full_image_toggled(self, _checked: bool) -> None:
        self._refresh_preview()

    def _emit_apply_to_main(self) -> None:
        self.apply_to_main_requested.emit(self._build_apply_payload())

    def _populate_sample_list(self) -> None:
        self._sample_list_updating = True
        self.sample_list_widget.setUpdatesEnabled(False)
        self.sample_list_widget.clear()
        self.sample_list_widget.addItems(
            [sample_path.name for sample_path, _label_path in self._sample_pairs]
        )
        self.sample_list_widget.setToolTip(str(self._training_parameters.image_path))
        synthetic_enabled = bool(self._panel.synthetic_defect_generator_check_box.isChecked())
        self.sample_list_widget.setEnabled(bool(self._sample_pairs) and not synthetic_enabled)
        if self._sample_pairs:
            self.sample_list_widget.setCurrentRow(self._current_sample_index)
        self.sample_list_widget.setUpdatesEnabled(True)
        self._sample_list_updating = False

    def _show_loading_state(self) -> None:
        loading_text = str(self._texts.get('loading', 'Loading preview...'))
        self.image_preview.setText(loading_text)
        self.label_preview.setText(loading_text)
        self.status_label.setText(loading_text)

    def _sync_sample_list_selection(self) -> None:
        if self.sample_list_widget.count() != len(self._sample_pairs):
            self._populate_sample_list()
            return
        self._sample_list_updating = True
        self.sample_list_widget.setEnabled(bool(self._sample_pairs) and not self._panel.synthetic_defect_generator_check_box.isChecked())
        if self._sample_pairs and self.sample_list_widget.currentRow() != self._current_sample_index:
            self.sample_list_widget.setCurrentRow(self._current_sample_index)
        elif not self._sample_pairs and self.sample_list_widget.currentRow() != -1:
            self.sample_list_widget.setCurrentRow(-1)
        self._sample_list_updating = False

    def _on_sample_list_row_changed(self, row: int) -> None:
        if self._sample_list_updating or not self._sample_pairs:
            return
        if row < 0 or row >= len(self._sample_pairs) or row == self._current_sample_index:
            return
        self._current_sample_index = row
        self._rebuild_frame_plan()
        for frame_index, (source_index, _cutter_item, _aug_variant) in enumerate(self._frame_plan):
            if source_index == row:
                self._current_frame_index = frame_index
                break
        self._refresh_preview()

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
        if not self._panel.photometric_groupbox.isChecked():
            return False
        return float(spinbox.value()) > 0.0

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

    def _refresh_preview(self) -> None:
        synthetic_enabled = self._panel.synthetic_defect_generator_check_box.isChecked()
        if not synthetic_enabled:
            self._rebuild_frame_plan()
            if self._frame_plan:
                self._current_frame_coords()
        if not self._sample_pairs and not synthetic_enabled:
            self._sync_sample_list_selection()
            error_text = str(
                self._load_error
                or self._texts.get('empty_error', 'No matched sample/label pairs were found.')
            )
            self.sample_label.setText(error_text)
            self.status_label.setText(error_text)
            self.image_preview.setText(error_text)
            self.label_preview.setText(error_text)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.resample_button.setEnabled(False)
            self._update_preview_mode_label()
            return

        self._sync_sample_list_selection()
        if synthetic_enabled:
            self.sample_label.setText(
                str(
                    self._texts.get(
                        'synthetic_label',
                        'Синтетическая топология' if self._is_russian_ui else 'Synthetic topology',
                    )
                )
            )
        else:
            sample_path, _label_path = self._sample_pairs[self._current_sample_index]
            self.sample_label.setText(
                str(
                    self._texts.get('sample_label_template', '{index}/{total}: {name}')
                ).format(
                    index=self._current_frame_index + 1,
                    total=max(1, len(self._frame_plan)),
                    name=sample_path.name,
                )
            )
        self.status_label.setText(
            str(
                self._texts.get(
                    'status_hold_template',
                    'Variant #{variant}. Preview: {preview_mode}. Hold the middle mouse button to inspect the original image and label.',
                )
            ).format(
                variant=self._variant_serial + 1,
                preview_mode=self._preview_mode_text(),
            )
        )
        try:
            original_image, original_label, augmented_image, augmented_label = self._build_preview_arrays(
                self._current_sample_index
            )
        except Exception as exc:
            error_text = str(
                self._texts.get('preview_error', 'Unable to build preview: {error}')
            ).format(error=exc)
            self.status_label.setText(error_text)
            self.image_preview.setText(error_text)
            self.label_preview.setText(error_text)
            return
        self._original_image_array = original_image
        self._original_label_array = original_label
        self._augmented_image_array = augmented_image
        self._augmented_label_array = augmented_label
        self.prev_button.setEnabled((not synthetic_enabled) and len(self._frame_plan) > 1)
        self.next_button.setEnabled((not synthetic_enabled) and len(self._frame_plan) > 1)
        self.resample_button.setEnabled(True)
        self._update_preview_mode_label()
        self._update_visible_preview()

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

    def _build_preview_arrays(
        self,
        sample_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        synthetic = self._panel.synthetic_defect_generator_check_box.isChecked()
        raw_image, raw_label = (
            self._build_synthetic_base_arrays()
            if synthetic
            else self._load_prepared_arrays(sample_index)
        )
        sem_config = self._current_sem_config()
        original_base = apply_dataset_preprocessing(raw_image, sem_config.preprocessing)
        with _seeded_random(self._seed_for(sample_index, f'sem:{self._variant_serial}')):
            augmented_base, augmented_base_label = apply_dataset_sem_augmentation_preview(
                raw_image,
                raw_label,
                sem_config.augmentation,
            )
        augmented_base = apply_dataset_preprocessing(
            augmented_base,
            sem_config.preprocessing,
        )

        full_image = self.full_image_check_box.isChecked()
        if full_image:
            original_image, original_label = self._build_original_full_image(
                original_base,
                raw_label,
            )
            augmented_image, augmented_label = self._build_augmented_full_image(
                sample_index,
                augmented_base,
                augmented_base_label,
                include_mixup=not synthetic,
            )
        else:
            original_image, original_label = self._build_original_patch(
                sample_index,
                original_base,
                raw_label,
            )
            augmented_image, augmented_label = self._build_augmented_patch(
                sample_index,
                augmented_base,
                augmented_base_label,
                include_mixup=not synthetic,
            )
        return (
            self._to_display_array(original_image),
            self._to_display_array(original_label),
            self._to_display_array(augmented_image),
            self._to_display_array(augmented_label),
        )

    def _build_synthetic_base_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        patch_height, patch_width = tuple(getattr(self._training_parameters.generation, 'segment_size', (256, 256)))
        size_hw = (
            max(int(patch_height), int(self._panel.synthetic_image_height_spinbox.value())),
            max(int(patch_width), int(self._panel.synthetic_image_width_spinbox.value())),
        )
        trace_count = self._sample_preview_int_range(
            self._panel.synthetic_trace_count_min_spinbox,
            self._panel.synthetic_trace_count_max_spinbox,
            salt='synthetic_trace_count',
        )
        background_noise_sigma = self._sample_preview_float_range(
            self._panel.synthetic_background_noise_sigma_min_spinbox,
            self._panel.synthetic_background_noise_sigma_max_spinbox,
            salt='synthetic_background_noise_sigma',
        )
        trace_noise_sigma = self._sample_preview_float_range(
            self._panel.synthetic_trace_noise_sigma_min_spinbox,
            self._panel.synthetic_trace_noise_sigma_max_spinbox,
            salt='synthetic_trace_noise_sigma',
        )
        params = SyntheticTopologyParameters(
            trace_count=trace_count,
            segment_count_range=tuple(
                sorted(
                    (
                        int(self._panel.synthetic_segment_count_min_spinbox.value()),
                        int(self._panel.synthetic_segment_count_max_spinbox.value()),
                    )
                )
            ),
            trace_half_width_range=tuple(
                sorted(
                    (
                        int(self._panel.synthetic_trace_half_width_min_spinbox.value()),
                        int(self._panel.synthetic_trace_half_width_max_spinbox.value()),
                    )
                )
            ),
            topology_domain=self._get_synthetic_topology_domain(),
            topology_family=self._get_synthetic_topology_family(),
            via_count_range=(1, max(1, min(6, int(round(trace_count / 3.0))))),
            background_noise_sigma=background_noise_sigma,
            trace_noise_sigma=trace_noise_sigma,
        )
        generator = SyntheticTopologyGenerator(params)
        synthetic_channels = 3 if self._get_synthetic_topology_domain() == 'pcb' else int(self._training_parameters.colors)
        image_array, label_array = generator.generate(
            size_hw=size_hw,
            channels=synthetic_channels,
            seed=self._seed_for(self._current_sample_index, f'synthetic:{self._variant_serial}'),
        )
        return image_array.astype(np.float32, copy=False), label_array.astype(np.float32, copy=False)

    def _build_augmented_synthetic_pair(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
        *,
        full_image: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if full_image:
            return self._build_augmented_full_image(
                sample_index,
                image_matrix,
                label_matrix,
                include_mixup=False,
            )
        return self._build_augmented_patch(
            sample_index,
            image_matrix,
            label_matrix,
            include_mixup=False,
        )

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

    def _build_original_patch(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._extract_patch(
            sample_index,
            image_matrix,
            label_matrix,
            random_crop=False,
            scale=False,
        )

    def _build_original_full_image(
        self,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            image_matrix.astype(np.float32, copy=True),
            label_matrix.astype(np.float32, copy=True),
        )

    def _build_augmented_patch(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
        *,
        include_mixup: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_patch, label_patch = self._build_pre_batch_patch(
            sample_index,
            image_matrix,
            label_matrix,
        )
        if include_mixup and self._panel.mixup_check_box.isChecked():
            image_patch, label_patch = self._apply_mixup(sample_index, image_patch, label_patch)
        image_patch = self._apply_cutout(sample_index, image_patch)
        image_patch = self._apply_random_artifacts(sample_index, image_patch)
        return image_patch, label_patch

    def _build_augmented_full_image(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
        *,
        include_mixup: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_full, label_full = self._build_pre_batch_full_image(
            sample_index,
            image_matrix,
            label_matrix,
        )
        if include_mixup and self._panel.mixup_check_box.isChecked():
            image_full, label_full = self._apply_mixup(
                sample_index,
                image_full,
                label_full,
                full_image=True,
            )
        image_full = self._apply_cutout(sample_index, image_full)
        image_full = self._apply_random_artifacts(sample_index, image_full)
        return image_full, label_full

    def _build_pre_batch_patch(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        augmented_image = image_matrix.astype(np.float32, copy=True)
        augmented_label = label_matrix.astype(np.float32, copy=True)
        image_patch, label_patch = self._extract_patch(
            sample_index,
            augmented_image,
            augmented_label,
            random_crop=self._panel.random_crop_check_box.isChecked(),
            scale=self._panel.scale_augmentation_check_box.isChecked(),
        )
        image_patch, label_patch = self._apply_rotations(image_patch, label_patch)
        image_patch = self._apply_photometric_augmentations(sample_index, image_patch)
        image_patch, label_patch = self._apply_pcb_defects(sample_index, image_patch, label_patch)
        return image_patch, label_patch

    def _build_pre_batch_full_image(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_full = image_matrix.astype(np.float32, copy=True)
        label_full = label_matrix.astype(np.float32, copy=True)
        image_full, label_full = self._apply_rotations(image_full, label_full)
        image_full = self._apply_photometric_augmentations(sample_index, image_full)
        image_full, label_full = self._apply_pcb_defects(sample_index, image_full, label_full)
        return image_full, label_full

    def _extract_patch(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
        *,
        random_crop: bool,
        scale: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        generation = self._generation_settings_for_cutter()
        with _seeded_random(self._seed_for(sample_index, f'extract:{int(random_crop)}:{int(scale)}')):
            cutter = SampleFastCutter((image_matrix, label_matrix), generation, shuffle=False)
        if len(cutter) <= 0:
            return image_matrix.copy(), label_matrix.copy()
        item_index = min(max(0, int(getattr(self, '_cutter_item_index', 0))), len(cutter) - 1)
        image_patch, label_patch = cutter[item_index]
        return (
            np.asarray(image_patch, dtype=np.float32).copy(),
            np.asarray(label_patch, dtype=np.float32).copy(),
        )

    def _apply_rotations(
        self,
        image_patch: np.ndarray,
        label_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = image_patch.copy()
        label = label_patch.copy()
        operations = self._panel.get_training_augmentation_config()

        def selected(key: str) -> bool:
            item = operations[key]
            return bool(item['enabled']) and random.random() < float(item['probability'])

        if selected('rotate_90') and image.shape[1] == image.shape[2]:
            image = np.rot90(image, k=-1, axes=(1, 2)).copy()
            label = np.rot90(label, k=-1, axes=(1, 2)).copy()
        if selected('rotate_180'):
            image = image[:, ::-1, ::-1].copy()
            label = label[:, ::-1, ::-1].copy()
        if selected('flip_x'):
            image = image[:, ::-1, :].copy()
            label = label[:, ::-1, :].copy()
        if selected('flip_y'):
            image = image[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()
        return image, label

    def _apply_photometric_augmentations(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        multiplier = self._augmentation_multiplier_value()
        if multiplier > 0.0 and self._variant_serial <= 0:
            return image_patch.astype(np.float32, copy=False)
        image = image_patch.astype(np.float32, copy=True)
        with _seeded_random(self._seed_for(sample_index, 'photometric')):
            operations = self._panel.get_training_augmentation_config()

            def selected(key: str) -> bool:
                item = operations[key]
                return bool(item['enabled']) and random.random() < float(item['probability'])

            if selected('blur'):
                blur_radius = max(0.0, float(self._panel.augmentation_blur_radius_spinbox.value()))
                if blur_radius > 0.0:
                    image = SampleFastCutter._apply_gaussian_blur(image, blur_radius)
            if selected('brightness'):
                strength = max(0.0, float(self._panel.augmentation_brightness_spinbox.value()))
                brightness = 1.0 + strength
                image *= float(brightness)
            if selected('contrast'):
                strength = max(0.0, float(self._panel.augmentation_contrast_spinbox.value()))
                contrast = 1.0 + strength
                mean = image.mean(axis=(1, 2), keepdims=True)
                image = (image - mean) * float(contrast) + mean
            if selected('gamma'):
                gamma_strength = max(0.0, float(self._panel.augmentation_gamma_spinbox.value()))
                gamma = max(0.1, 1.0 - min(gamma_strength, 0.9))
                image = np.power(np.clip(image, 0.0, 1.0), float(gamma)).astype(np.float32, copy=False)
            if selected('noise'):
                sigma = max(0.0, float(self._panel.augmentation_noise_sigma_spinbox.value()))
                if sigma > 0.0:
                    image += np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        np.clip(image, 0.0, 1.0, out=image)
        return image.astype(np.float32, copy=False)

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

    def _apply_tech_variations(
        self,
        sample_index: int,
        image_matrix: np.ndarray,
        label_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        config = self._build_apply_tech_aug_config(preview=True)
        if not config.enabled:
            return image_matrix, label_matrix
        selected_count = sum(
            1
            for spinbox in (
                self._panel.tech_aug_global_width_probability_spinbox,
                self._panel.tech_aug_scale_rethreshold_probability_spinbox,
                self._panel.tech_aug_blur_threshold_probability_spinbox,
                self._panel.tech_aug_boundary_aware_probability_spinbox,
                self._panel.tech_aug_local_morphology_probability_spinbox,
                self._panel.tech_aug_gap_variation_probability_spinbox,
            )
            if self._tech_effect_enabled(spinbox)
        )
        if selected_count <= 0:
            return image_matrix, label_matrix
        augmentor = TechVariationAugmentor(config)
        source_image = image_matrix.astype(np.float32, copy=False)
        source_label = label_matrix.astype(np.float32, copy=False)
        preview_attempts = max(1, min(12, max(selected_count * 2, int(getattr(config, 'max_operations', 1)))))
        for attempt_index in range(preview_attempts):
            with _seeded_random(self._seed_for(sample_index, f'tech:{attempt_index}')):
                augmented_image, augmented_label = _apply_binary_tech_augmentation_to_pair(
                    source_image,
                    source_label,
                    augmentor,
                    binary_tolerance=float(getattr(config, 'binary_tolerance', 0.15)),
                )
            if np.array_equal(augmented_image, source_image) and np.array_equal(augmented_label, source_label):
                continue
            return augmented_image, augmented_label
        return source_image, source_label

    def _apply_pcb_defects(
        self,
        sample_index: int,
        image_patch: np.ndarray,
        label_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._panel.synthetic_defect_generator_check_box.isChecked():
            return image_patch, label_patch
        if not self._panel.pcb_defects_check_box.isChecked():
            return image_patch, label_patch
        current_domain = self._get_synthetic_topology_domain()
        if current_domain == 'ic':
            selected_probabilities = {
                'line_break': 1.0 if self._panel.ic_defect_type_checkboxes["line_break"].isChecked() else 0.0,
                'bridge': 1.0 if self._panel.ic_defect_type_checkboxes["bridge"].isChecked() else 0.0,
                'necking': 1.0 if self._panel.ic_defect_type_checkboxes["necking"].isChecked() else 0.0,
                'missing_metal': 1.0 if self._panel.ic_defect_type_checkboxes["missing_metal"].isChecked() else 0.0,
                'spur': 1.0 if self._panel.ic_defect_type_checkboxes["spur"].isChecked() else 0.0,
                'pinhole': 1.0 if self._panel.ic_defect_type_checkboxes["pinhole"].isChecked() else 0.0,
                'via_open': 1.0 if self._panel.ic_defect_type_checkboxes["via_open"].isChecked() else 0.0,
                'line_shift': 1.0 if self._panel.ic_defect_type_checkboxes["line_shift"].isChecked() else 0.0,
            }
            selected_severities = {
                defect_name: float(self._panel.ic_defect_type_spinboxes[defect_name].value()) / 100.0
                for defect_name in selected_probabilities
            }
            config = self._build_apply_ic_defects_config()
            augmentor_cls = ICDefectAugmentor
        else:
            selected_probabilities = {
                'break': 1.0 if self._panel.pcb_defect_type_checkboxes["break"].isChecked() else 0.0,
                'short': 1.0 if self._panel.pcb_defect_type_checkboxes["short"].isChecked() else 0.0,
                'missing_copper': 1.0 if self._panel.pcb_defect_type_checkboxes["missing_copper"].isChecked() else 0.0,
                'excess_copper': 1.0 if self._panel.pcb_defect_type_checkboxes["excess_copper"].isChecked() else 0.0,
                'pinhole': 1.0 if self._panel.pcb_defect_type_checkboxes["pinhole"].isChecked() else 0.0,
                'spurious_copper': 1.0 if self._panel.pcb_defect_type_checkboxes["spurious_copper"].isChecked() else 0.0,
                'via': 1.0 if self._panel.pcb_defect_type_checkboxes["via"].isChecked() else 0.0,
                'misalignment': 1.0 if self._panel.pcb_defect_type_checkboxes["misalignment"].isChecked() else 0.0,
            }
            selected_severities = {
                defect_name: float(self._panel.pcb_defect_type_spinboxes[defect_name].value()) / 100.0
                for defect_name in selected_probabilities
            }
            config = self._build_apply_pcb_defects_config()
            augmentor_cls = PCBDefectAugmentor
        active_count = sum(1 for probability in selected_probabilities.values() if probability > 0.0)
        if active_count <= 0:
            return image_patch, label_patch
        config.defect_probability = 1.0
        config.min_defects = min(active_count, max(1, int(self._panel.pcb_defects_min_count_spinbox.value())))
        config.max_defects = min(active_count, max(config.min_defects, int(self._panel.pcb_defects_max_count_spinbox.value())))
        for defect_name in tuple(config.defect_probabilities.keys()):
            config.defect_probabilities[defect_name] = float(selected_probabilities.get(defect_name, 0.0))
            config.defect_severities[defect_name] = float(selected_severities.get(defect_name, 0.5))
        augmentor = augmentor_cls(config)
        source_image = image_patch.astype(np.float32, copy=False)
        source_label = label_patch.astype(np.float32, copy=False)
        preview_attempts = max(1, min(16, int(getattr(config, 'max_attempts_per_defect', 8))))
        for attempt_index in range(preview_attempts):
            augmented_image, defect_mask, _augmented_mask = augmentor(
                source_image,
                source_label,
                seed=self._seed_for(sample_index, f'pcb_defects_{attempt_index}'),
                return_augmented_mask=True,
            )
            defect_mask_array = np.asarray(defect_mask)
            if np.count_nonzero(defect_mask_array) <= 0 and np.array_equal(augmented_image, source_image):
                continue
            return (
                augmented_image,
                source_label,
            )
        return source_image, source_label

    def _apply_mixup(
        self,
        sample_index: int,
        image_patch: np.ndarray,
        label_patch: np.ndarray,
        *,
        full_image: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self._sample_pairs) <= 1:
            return image_patch, label_patch
        alpha = max(0.0, float(self._panel.mixup_alpha_spinbox.value()))
        if alpha <= 0.0:
            return image_patch, label_patch
        partner_index = (sample_index + 1 + self._variant_serial) % len(self._sample_pairs)
        if partner_index == sample_index:
            partner_index = (partner_index + 1) % len(self._sample_pairs)
        partner_base_image, partner_base_label = self._load_prepared_arrays(partner_index)
        if full_image:
            partner_image, partner_label = self._build_pre_batch_full_image(
                partner_index,
                partner_base_image,
                partner_base_label,
            )
        else:
            partner_image, partner_label = self._build_pre_batch_patch(
                partner_index,
                partner_base_image,
                partner_base_label,
            )
        if partner_image.shape != image_patch.shape or partner_label.shape != label_patch.shape:
            return image_patch, label_patch
        with _seeded_random(self._seed_for(sample_index, 'mixup')):
            lambda_value = float(np.random.beta(alpha, alpha))
        lambda_value = float(min(max(lambda_value, 0.0), 1.0))
        mixed_image = (lambda_value * image_patch) + ((1.0 - lambda_value) * partner_image)
        mixed_label = (lambda_value * label_patch) + ((1.0 - lambda_value) * partner_label)
        return (
            mixed_image.astype(np.float32, copy=False),
            mixed_label.astype(np.float32, copy=False),
        )

    def _apply_cutout(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        if not self._panel.cutout_check_box.isChecked():
            return image_patch
        holes = max(1, int(self._panel.cutout_holes_spinbox.value()))
        size_ratio = float(self._panel.cutout_size_ratio_spinbox.value())
        if size_ratio <= 0.0:
            return image_patch
        image = torch.from_numpy(np.ascontiguousarray(image_patch[None, ...])).float()
        with _seeded_random(self._seed_for(sample_index, 'cutout')):
            _batch, channels, height, width = image.shape
            max_cutout_height = max(1, min(int(height), int(round(int(height) * size_ratio))))
            max_cutout_width = max(1, min(int(width), int(round(int(width) * size_ratio))))
            if max_cutout_height <= 0 or max_cutout_width <= 0:
                return image_patch
            for _ in range(holes):
                cutout_height = (
                    1 if max_cutout_height == 1 else int(torch.randint(1, max_cutout_height + 1, (1,)).item())
                )
                cutout_width = (
                    1 if max_cutout_width == 1 else int(torch.randint(1, max_cutout_width + 1, (1,)).item())
                )
                max_top = max(0, height - cutout_height)
                max_left = max(0, width - cutout_width)
                top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,)).item())
                left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,)).item())
                fill_color = torch.rand((channels, 1, 1), dtype=image.dtype)
                image[0, :, top:top + cutout_height, left:left + cutout_width] = fill_color
        return image[0].numpy().astype(np.float32, copy=False)

    def _apply_random_artifacts(self, sample_index: int, image_patch: np.ndarray) -> np.ndarray:
        if not self._panel.random_artifacts_check_box.isChecked():
            return image_patch
        artifact_types = self._selected_artifact_types()
        if not artifact_types:
            return image_patch
        count = max(1, int(self._panel.random_artifacts_count_spinbox.value()))
        size_ratio = float(self._panel.random_artifacts_size_ratio_spinbox.value())
        if size_ratio <= 0.0:
            return image_patch
        image = torch.from_numpy(np.ascontiguousarray(image_patch[None, ...])).float()
        _, channels, height, width = image.shape
        min_h, max_h, min_w, max_w = self._artifact_size_bounds(height, width, size_ratio)
        with _seeded_random(self._seed_for(sample_index, 'random_artifacts')):
            for _ in range(count):
                artifact_height = int(min_h if max_h == min_h else np.random.randint(min_h, max_h + 1))
                artifact_width = int(min_w if max_w == min_w else np.random.randint(min_w, max_w + 1))
                max_top = max(0, height - artifact_height)
                max_left = max(0, width - artifact_width)
                top = 0 if max_top == 0 else int(np.random.randint(0, max_top + 1))
                left = 0 if max_left == 0 else int(np.random.randint(0, max_left + 1))
                overlay, alpha = generate_random_artifact_patch(
                    int(channels),
                    int(artifact_height),
                    int(artifact_width),
                    device=torch.device('cpu'),
                    dtype=torch.float32,
                    artifact_types=artifact_types,
                )
                patch = image[0, :, top:top + artifact_height, left:left + artifact_width]
                image[0, :, top:top + artifact_height, left:left + artifact_width] = torch.clamp(
                    (patch * (1.0 - alpha)) + (overlay * alpha),
                    min=0.0,
                    max=1.0,
                )
        return image[0].numpy().astype(np.float32, copy=False)

    def _selected_artifact_types(self) -> tuple[str, ...]:
        return tuple(
            artifact_name
            for artifact_name, checkbox in self._panel.random_artifact_type_checkboxes.items()
            if checkbox.isChecked()
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

    def _sample_preview_int_range(self, min_widget: QWidget, max_widget: QWidget, *, salt: str) -> int:
        lower = int(min(getattr(min_widget, 'value')(), getattr(max_widget, 'value')()))
        upper = int(max(getattr(min_widget, 'value')(), getattr(max_widget, 'value')()))
        with _seeded_random(self._seed_for(self._current_sample_index, salt)):
            return int(np.random.randint(lower, upper + 1))

    def _sample_preview_float_range(self, min_widget: QWidget, max_widget: QWidget, *, salt: str) -> float:
        lower = float(min(getattr(min_widget, 'value')(), getattr(max_widget, 'value')()))
        upper = float(max(getattr(min_widget, 'value')(), getattr(max_widget, 'value')()))
        if upper <= lower:
            return lower
        with _seeded_random(self._seed_for(self._current_sample_index, salt)):
            return float(np.random.uniform(lower, upper))

    def _passes_probability(self, sample_index: int, salt: str, probability: float) -> bool:
        return float(probability) > 0.0

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

    def _resolve_tip(self, key: str) -> str:
        return str(
            self._texts.get(
                f'{key}_tip',
                self._settings_texts.get(
                    f'{key}_tip',
                    self._settings_form_tooltips.get(key, ''),
                ),
            )
        )

    @staticmethod
    def _bounded_int(value: object, lower: int, upper: int) -> int:
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            resolved = int(lower)
        return max(int(lower), min(int(upper), resolved))

    @staticmethod
    def _bounded_float(value: object, lower: float, upper: float) -> float:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            resolved = float(lower)
        return max(float(lower), min(float(upper), resolved))

    @staticmethod
    def _artifact_size_bounds(
        image_height: int,
        image_width: int,
        size_ratio: float,
    ) -> tuple[int, int, int, int]:
        max_artifact_height = max(1, min(int(image_height), int(round(int(image_height) * float(size_ratio)))))
        max_artifact_width = max(1, min(int(image_width), int(round(int(image_width) * float(size_ratio)))))
        min_artifact_height = 1 if max_artifact_height <= 2 else max(2, int(round(max_artifact_height * 0.35)))
        min_artifact_width = 1 if max_artifact_width <= 2 else max(2, int(round(max_artifact_width * 0.35)))
        min_artifact_height = min(max_artifact_height, min_artifact_height)
        min_artifact_width = min(max_artifact_width, min_artifact_width)
        return (
            int(min_artifact_height),
            int(max_artifact_height),
            int(min_artifact_width),
            int(max_artifact_width),
        )

    def _seed_for(self, sample_index: int, salt: str) -> int:
        if 0 <= int(sample_index) < len(self._sample_pairs):
            sample_key = self._sample_pairs[sample_index][0].as_posix()
        else:
            sample_key = 'synthetic'
        payload = (
            f'{sample_key}|{sample_index}|{getattr(self, "_cutter_item_index", 0)}|'
            f'{self._variant_serial}|{self._resample_salt}|{salt}'
        )
        return int(zlib.crc32(payload.encode('utf-8')) & 0xFFFFFFFF)

    @staticmethod
    def _to_display_array(image_array: np.ndarray) -> np.ndarray:
        array = np.asarray(image_array, dtype=np.float32)
        finite = array[np.isfinite(array)]
        if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
            low, high = np.percentile(finite, (1.0, 99.0))
            if float(high - low) > 1e-6:
                array = (array - float(low)) / float(high - low)
            else:
                array = np.zeros_like(array, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.ndim == 2:
            return np.clip(np.round(array * 255.0), 0.0, 255.0).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] == 1:
            return np.clip(np.round(array[0] * 255.0), 0.0, 255.0).astype(np.uint8)
        if array.ndim == 3 and array.shape[0] >= 3:
            rgb = np.transpose(array[:3], (1, 2, 0))
            return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
        return np.zeros((16, 16), dtype=np.uint8)

    @staticmethod
    def _set_preview_image(
        widget: QLabel,
        image_data: np.ndarray | None,
        *,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> None:
        if not isinstance(image_data, np.ndarray) or image_data.size == 0:
            widget.clear()
            return
        if image_data.ndim == 2:
            contiguous = np.ascontiguousarray(image_data)
            qimg = QImage(
                contiguous.tobytes(),
                contiguous.shape[1],
                contiguous.shape[0],
                contiguous.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()
        elif image_data.ndim == 3 and image_data.shape[2] == 3:
            contiguous = np.ascontiguousarray(image_data)
            qimg = QImage(
                contiguous.tobytes(),
                contiguous.shape[1],
                contiguous.shape[0],
                contiguous.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
        else:
            widget.clear()
            return
        scaled_width = max(1, int(target_width or widget.width()))
        scaled_height = max(1, int(target_height or widget.height()))
        pixmap = QPixmap.fromImage(qimg).scaled(
            scaled_width,
            scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        widget.setPixmap(pixmap)
