from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import (
    QWidget,
    QDockWidget,
    QCheckBox,
    QFormLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QApplication,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QSizePolicy,
    QGroupBox,
    QRadioButton,
    QVBoxLayout,
    QHBoxLayout,
)

from lib.data_interfaces import SampleCutMode, WorkMode
from lib.ui_texts import get_ui_section

SHIFT_RANGE_MIN = 4
SHIFT_RANGE_MAX = 256

SAMPLE_SIZE_MIN = 8
SAMPLE_SIZE_MAX = 1024

VALIDATION_MIN = 0
VALIDATION_MAX = 50

MIN_BATCH = 4
MAX_BATCH = 64

MIN_OVERLAP = 0
MAX_OVERLAP = 32
MIN_LOG_UPDATE_FREQUENCY = 0
MAX_LOG_UPDATE_FREQUENCY = 5000

MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1.0
MIN_WEIGHT_DECAY = 0.0
MAX_WEIGHT_DECAY = 1.0
MIN_EARLY_STOPPING_PATIENCE = 0
MAX_EARLY_STOPPING_PATIENCE = 1000
MIN_EARLY_STOPPING_MIN_DELTA = 0.0
MAX_EARLY_STOPPING_MIN_DELTA = 1.0
MIN_WARMUP_EPOCHS = 1
MAX_WARMUP_EPOCHS = 2000
MIN_WARMUP_START_FACTOR = 0.0
MAX_WARMUP_START_FACTOR = 1.0
MIN_HARD_MINING_STRENGTH = 0.0
MAX_HARD_MINING_STRENGTH = 10.0
MIN_HARD_MINING_EMA_ALPHA = 0.0
MAX_HARD_MINING_EMA_ALPHA = 1.0
MIN_AUG_STRENGTH = 0.0
MAX_AUG_STRENGTH = 1.0
MIN_AUG_NOISE_SIGMA = 0.0
MAX_AUG_NOISE_SIGMA = 0.2
OPTIMIZERS = ('adam', 'adamw', 'adamw_muon')
MIXED_PRECISION_MODES = ('off', 'fp16', 'bf16')
LOSS_FUNCTIONS = ('bce', 'dice', 'bce_dice', 'iou', 'bce_iou')
OPTIMIZER_PRESETS = (
    ('Adam', 'adam', 1e-3, 0.0),
    ('AdamW', 'adamw', 5e-4, 1e-2),
    ('AdamW + Muon', 'adamw_muon', 3e-4, 2e-2),
)

class SlidingPanel(QScrollArea):
    """Animated sliding panel wrapper."""

    def __init__(self, widget, width=450, duration=350, parent=None):
        super().__init__(parent)
        self._content = widget
        self._width = width
        self._duration = duration

        self.setWidget(self._content)
        self.setWidgetResizable(True)

        self.hide()
        self._animation = QPropertyAnimation(self, b'geometry')
        self._animation.setDuration(self._duration)
        self._animation.setEasingCurve(QEasingCurve.Type.BezierSpline)
        self._animation.finished.connect(self._on_animation_finished)

    def toggle(self):
        if self.isVisible():
            self._slide_out()
        else:
            self._slide_in()

    def set_width(self, width):
        self._width = width
        self.setMinimumWidth(self._width)

    def _slide_in(self):
        self.show()
        parent = self.parent()
        if not isinstance(parent, QWidget):
            return

        parent_rect = parent.rect()
        start = QRect(parent_rect.right(), 0, self._width, parent_rect.height())
        end = QRect(parent_rect.right() - self._width, 0, self._width, parent_rect.height())
        self._animating_in = True
        self._animation.stop()
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def _slide_out(self):
        parent = self.parent()
        if not isinstance(parent, QWidget):
            return

        parent_rect = parent.rect()
        start = QRect(parent_rect.right() - self._width, 0, self._width, parent_rect.height())
        end = QRect(parent_rect.right(), 0, self._width, parent_rect.height())
        self._animating_in = False
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def _on_animation_finished(self):
        if not self._animating_in:
            self.hide()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


def get_text_index_in_qcombobox(combobox: QComboBox, text: str):
    item_texts = [combobox.itemText(i) for i in range(combobox.count())]
    try:
        text_location = item_texts.index(text)
    except ValueError:
        text_location = -1
    return text_location


def create_spinbox(spin_range: tuple[int, int], step: int, default_value: int, policy=None) -> QSpinBox:
    spinbox = NoWheelSpinBox()
    spinbox.setRange(spin_range[0], spin_range[1])
    spinbox.setValue(default_value)
    spinbox.setSingleStep(step)
    if isinstance(policy, QSizePolicy):
        spinbox.setSizePolicy(policy)
    return spinbox


def create_double_spinbox(
    spin_range: tuple[float, float], step: float, default_value: float, decimals: int = 6, policy=None
) -> QDoubleSpinBox:
    spinbox = NoWheelDoubleSpinBox()
    spinbox.setRange(spin_range[0], spin_range[1])
    spinbox.setValue(default_value)
    spinbox.setSingleStep(step)
    spinbox.setDecimals(decimals)
    spinbox.setKeyboardTracking(False)
    if isinstance(policy, QSizePolicy):
        spinbox.setSizePolicy(policy)
    return spinbox


def create_size_widget(x_size: QWidget, y_size: QWidget) -> QWidget:
    size_widget = QWidget()
    row_layout = QHBoxLayout(size_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(5)
    row_layout.addWidget(x_size)
    row_layout.addWidget(QLabel('X'))
    row_layout.addWidget(y_size)
    return size_widget


class SettingsPanel(QDockWidget):
    cut_slider_shifted: pyqtSignal = pyqtSignal()
    horisontal_rotate_clicked: pyqtSignal = pyqtSignal()
    vertical_rotate_clicked: pyqtSignal = pyqtSignal()
    model_changed: pyqtSignal = pyqtSignal()
    segment_size_changed: pyqtSignal = pyqtSignal()
    sample_size_changed: pyqtSignal = pyqtSignal()
    optimizer_settings_changed: pyqtSignal = pyqtSignal()
    validation_settings_changed: pyqtSignal = pyqtSignal()
    reset_defaults_requested: pyqtSignal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._texts = get_ui_section('settings_panel')
        self._content_widget = QWidget()
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._content_widget)
        self.setWidget(self._scroll_area)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.models: list[str] = []
        self.size_policy = QSizePolicy()
        self._desc_labels: dict[str, QLabel] = {}
        self._field_rows: dict[QWidget, QWidget] = {}
        self._desc_fields: dict[str, QWidget] = {}
        self._training_controls_applicable = True
        self._recognition_controls_applicable = True
        self._model_selector_applicable = True

        self.horizontal_rotation = QCheckBox('')
        self.vertical_rotation = QCheckBox('')
        self.additional_augmentation_check_box = QCheckBox('')
        self.validation_check_box = QCheckBox('')
        self.samples_number = QLabel('')

        self.nn_model_type = NoWheelComboBox()
        self.nn_model_type.setSizePolicy(self.size_policy)

        self.shift_spinbox = create_spinbox((SHIFT_RANGE_MIN, SHIFT_RANGE_MAX), default_value=100, step=10)
        self.augmentation_brightness_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=0.1,
            decimals=2,
        )
        self.augmentation_contrast_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=0.1,
            decimals=2,
        )
        self.augmentation_noise_probability_spinbox = create_double_spinbox(
            (MIN_AUG_STRENGTH, MAX_AUG_STRENGTH),
            step=0.05,
            default_value=0.5,
            decimals=2,
        )
        self.augmentation_noise_sigma_spinbox = create_double_spinbox(
            (MIN_AUG_NOISE_SIGMA, MAX_AUG_NOISE_SIGMA),
            step=0.005,
            default_value=0.01,
            decimals=3,
        )

        self.sample_x_size = create_spinbox((SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX), default_value=256, step=10)
        self.sample_y_size = create_spinbox((SAMPLE_SIZE_MIN, SAMPLE_SIZE_MAX), default_value=256, step=10)

        self.validation_spinbox = create_spinbox((VALIDATION_MIN, VALIDATION_MAX), default_value=20, step=5)
        self.validation_check_box.toggled.connect(self._sync_validation_controls)
        self._sync_validation_controls(self.validation_check_box.isChecked())

        self._init_color_type_combobox()
        self._init_sample_type()
        self._init_nn_auxilary_settings()
        self._init_preprocess_groupbox()
        self._init_layout()
        self._apply_russian_texts()
        self.sync_business_logic_controls(None)

    def _get_desc_label(self, key: str) -> QLabel:
        label = self._desc_labels.get(key)
        if label is not None:
            return label
        label = QLabel('')
        label.setWordWrap(False)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setStyleSheet('')
        self._desc_labels[key] = label
        return label

    def _field_with_description(self, field: QWidget, key: str) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(self._get_desc_label(key), 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        row_layout.addWidget(field, 1, Qt.AlignmentFlag.AlignVCenter)
        self._field_rows[field] = row_widget
        self._desc_fields[key] = field
        return row_widget

    @staticmethod
    def _apply_tooltip_to_widget_and_children(widget: QWidget, text: str) -> None:
        widget.setToolTip(text)
        for child in widget.findChildren(QWidget):
            child.setToolTip(text)

    def _set_field_enabled(self, field: QWidget, enabled: bool) -> None:
        row_widget = self._field_rows.get(field)
        if row_widget is not None:
            row_widget.setEnabled(enabled)
        field.setEnabled(enabled)

    @staticmethod
    def _resolve_work_mode_applicability(work_mode: str | None) -> tuple[bool, bool, bool]:
        mode = str(work_mode or '')
        training_modes = {
            WorkMode.train_only.value,
            WorkMode.train_and_recognition.value,
            WorkMode.further_training.value,
        }
        recognition_modes = {
            WorkMode.train_and_recognition.value,
            WorkMode.recognition_only.value,
            WorkMode.further_training.value,
        }
        model_selector_modes = {
            WorkMode.train_only.value,
            WorkMode.train_and_recognition.value,
        }
        known_modes = training_modes | recognition_modes
        if mode not in known_modes:
            return True, True, True
        return mode in training_modes, mode in recognition_modes, mode in model_selector_modes

    def sync_business_logic_controls(self, work_mode: str | None) -> None:
        training_applicable, recognition_applicable, model_selector_applicable = (
            self._resolve_work_mode_applicability(work_mode)
        )
        self._training_controls_applicable = training_applicable
        self._recognition_controls_applicable = recognition_applicable
        self._model_selector_applicable = model_selector_applicable

        batch_related = training_applicable or recognition_applicable

        self.samples_number.setEnabled(training_applicable)
        self._set_field_enabled(self.nn_model_type, model_selector_applicable)
        self._set_field_enabled(self.color_type, training_applicable)
        self._set_field_enabled(self.sample_size_widget, batch_related)

        self.augmentation_groupbox.setEnabled(training_applicable)
        self.horizontal_rotation.setEnabled(training_applicable)
        self.vertical_rotation.setEnabled(training_applicable)
        self.additional_augmentation_check_box.setEnabled(training_applicable)
        self._set_field_enabled(self.shift_spinbox, training_applicable)
        self._sync_augmentation_controls(self.additional_augmentation_check_box.isChecked())

        self.validation_groupbox.setEnabled(training_applicable)
        self.validation_check_box.setEnabled(training_applicable)
        self._sync_validation_controls(self.validation_check_box.isChecked())

        self.sample_type_groupbox.setEnabled(training_applicable)
        self.cut_dataset_type.setEnabled(training_applicable)
        self.no_cut_dataset_type.setEnabled(training_applicable)

        self.prepare_samples_groupbox.setEnabled(training_applicable)
        self.enable_crop_processing.setEnabled(training_applicable)
        self.enable_resize_processing.setEnabled(training_applicable)
        self._sync_preprocess_controls()

        self._set_field_enabled(self.optimizer_presets_widget, training_applicable)
        self._set_field_enabled(self.optimizer_type, training_applicable)
        self._set_field_enabled(self.learning_rate_spinbox, training_applicable)
        self._set_field_enabled(self.weight_decay_spinbox, training_applicable)
        self._set_field_enabled(self.log_update_frequency_spinbox, training_applicable)
        self._set_field_enabled(self.batch_spinbox, batch_related)
        self._set_field_enabled(self.overlap_spinbox, recognition_applicable)

        self._set_field_enabled(self.mixed_precision_type, training_applicable)
        self._set_field_enabled(self.loss_function_type, training_applicable)
        self._sync_loss_controls(self.loss_function_type.currentIndex())

        self.multi_gpu_check_box.setEnabled(training_applicable)
        self.skip_uniform_labels_check_box.setEnabled(training_applicable)
        self.torch_compile_check_box.setEnabled(batch_related)

        self.warmup_groupbox.setEnabled(training_applicable)
        self.warmup_check_box.setEnabled(training_applicable)
        self._sync_warmup_controls(self.warmup_check_box.isChecked())

        self.hard_mining_groupbox.setEnabled(training_applicable)
        self.hard_mining_check_box.setEnabled(training_applicable)
        self._sync_hard_mining_controls(self.hard_mining_check_box.isChecked())

        self.early_stopping_groupbox.setEnabled(training_applicable)
        self.early_stopping_check_box.setEnabled(training_applicable)
        self._sync_early_stopping_controls(self.early_stopping_check_box.isChecked())

    @staticmethod
    def _create_form_layout(margins: tuple[int, int, int, int] = (8, 6, 8, 8)) -> QFormLayout:
        layout = QFormLayout()
        layout.setContentsMargins(*margins)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        return layout

    def _init_color_type_combobox(self):
        self.color_type = NoWheelComboBox()
        self.color_type.addItem('')
        self.color_type.addItem('RGB')
        self.color_type.setSizePolicy(self.size_policy)

    def _init_sample_type(self):
        self.sample_type_groupbox = QGroupBox('')
        vbox_layout = QVBoxLayout()
        vbox_layout.setContentsMargins(8, 6, 8, 8)
        vbox_layout.setSpacing(6)
        self.sample_type_groupbox.setLayout(vbox_layout)
        self.cut_dataset_type = QRadioButton('')
        self.no_cut_dataset_type = QRadioButton('')
        vbox_layout.addWidget(self.cut_dataset_type)
        vbox_layout.addWidget(self.no_cut_dataset_type)

    def _init_preprocess_groupbox(self):
        self.prepare_samples_groupbox = QGroupBox('')
        self.enable_crop_processing = QCheckBox('')
        self.enable_resize_processing = QCheckBox('')
        self.cut_corner_spinbox = create_spinbox((0, 500), step=10, default_value=0)
        self.target_x_size = create_spinbox((0, 4000), step=100, default_value=2000)
        self.target_y_size = create_spinbox((0, 4000), step=100, default_value=2000)
        size_widget = create_size_widget(self.target_x_size, self.target_y_size)
        self.target_size_widget = size_widget

        form = self._create_form_layout()
        self.prepare_samples_groupbox.setLayout(form)
        self.prepare_samples_form = form
        form.addRow(self.enable_crop_processing)
        form.addRow(self.enable_resize_processing)
        form.addRow('', self._field_with_description(self.cut_corner_spinbox, 'edge_cut'))
        form.addRow('', self._field_with_description(self.target_size_widget, 'target_size'))
        self.enable_crop_processing.toggled.connect(self._sync_preprocess_controls)
        self.enable_resize_processing.toggled.connect(self._sync_preprocess_controls)
        self._sync_preprocess_controls()

    def _init_nn_auxilary_settings(self):
        self.nn_auxilary_settings_groupbox = QGroupBox('')
        nn_sections_layout = QVBoxLayout()
        nn_sections_layout.setContentsMargins(8, 6, 8, 8)
        nn_sections_layout.setSpacing(10)
        self.nn_auxilary_settings_groupbox.setLayout(nn_sections_layout)

        def _build_subgroup() -> tuple[QGroupBox, QFormLayout]:
            group = QGroupBox('')
            form = self._create_form_layout()
            group.setLayout(form)
            return group, form

        self.optimizer_preset_buttons: list[QPushButton] = []
        self.optimizer_presets_widget = QWidget()
        preset_layout = QHBoxLayout(self.optimizer_presets_widget)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        for title, optimizer_name, learning_rate, weight_decay in OPTIMIZER_PRESETS:
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.clicked.connect(
                lambda _checked=False, n=optimizer_name, lr=learning_rate, wd=weight_decay: self._apply_optimizer_preset(
                    n, lr, wd
                )
            )
            self.optimizer_preset_buttons.append(btn)
            preset_layout.addWidget(btn)

        self.optimizer_type = NoWheelComboBox()
        self.optimizer_type.addItems(list(OPTIMIZERS))
        self.mixed_precision_type = NoWheelComboBox()
        self.mixed_precision_type.addItems(list(MIXED_PRECISION_MODES))
        self.loss_function_type = NoWheelComboBox()
        self.loss_function_type.addItems(list(LOSS_FUNCTIONS))
        self.dice_loss_weight_spinbox = create_double_spinbox((0.0, 1.0), step=0.05, default_value=0.5, decimals=2)
        self.iou_loss_weight_spinbox = create_double_spinbox((0.0, 1.0), step=0.05, default_value=0.5, decimals=2)

        self.learning_rate_spinbox = create_double_spinbox(
            (MIN_LEARNING_RATE, MAX_LEARNING_RATE), step=1e-4, default_value=1e-3, decimals=6
        )
        self.weight_decay_spinbox = create_double_spinbox(
            (MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY), step=1e-4, default_value=0.0, decimals=6
        )

        self.batch_spinbox = create_spinbox((MIN_BATCH, MAX_BATCH), step=1, default_value=16)
        self.overlap_spinbox = create_spinbox((MIN_OVERLAP, MAX_OVERLAP), step=4, default_value=8)
        self.log_update_frequency_spinbox = create_spinbox(
            (MIN_LOG_UPDATE_FREQUENCY, MAX_LOG_UPDATE_FREQUENCY),
            step=1,
            default_value=0,
        )

        self.multi_gpu_check_box = QCheckBox('')
        self.torch_compile_check_box = QCheckBox('')
        self.warmup_check_box = QCheckBox('')
        self.warmup_epochs_spinbox = create_spinbox((MIN_WARMUP_EPOCHS, MAX_WARMUP_EPOCHS), step=1, default_value=3)
        self.warmup_start_factor_spinbox = create_double_spinbox(
            (MIN_WARMUP_START_FACTOR, MAX_WARMUP_START_FACTOR),
            step=0.01,
            default_value=0.1,
            decimals=3,
        )
        self.hard_mining_check_box = QCheckBox('')
        self.hard_mining_strength_spinbox = create_double_spinbox(
            (MIN_HARD_MINING_STRENGTH, MAX_HARD_MINING_STRENGTH),
            step=0.1,
            default_value=2.0,
            decimals=2,
        )
        self.hard_mining_ema_alpha_spinbox = create_double_spinbox(
            (MIN_HARD_MINING_EMA_ALPHA, MAX_HARD_MINING_EMA_ALPHA),
            step=0.05,
            default_value=0.2,
            decimals=2,
        )
        self.skip_uniform_labels_check_box = QCheckBox('')
        self.early_stopping_check_box = QCheckBox('')
        self.early_stopping_patience_spinbox = create_spinbox(
            (MIN_EARLY_STOPPING_PATIENCE, MAX_EARLY_STOPPING_PATIENCE),
            step=1,
            default_value=10,
        )
        self.early_stopping_min_delta_spinbox = create_double_spinbox(
            (MIN_EARLY_STOPPING_MIN_DELTA, MAX_EARLY_STOPPING_MIN_DELTA),
            step=1e-4,
            default_value=0.0,
            decimals=6,
        )
        self.restore_best_weights_check_box = QCheckBox('')
        self.restore_best_weights_check_box.setChecked(True)

        self.optimizer_groupbox, self.optimizer_form = _build_subgroup()
        self.precision_loss_groupbox, self.precision_loss_form = _build_subgroup()
        self.runtime_groupbox, self.runtime_form = _build_subgroup()
        self.warmup_groupbox, self.warmup_form = _build_subgroup()
        self.hard_mining_groupbox, self.hard_mining_form = _build_subgroup()
        self.early_stopping_groupbox, self.early_stopping_form = _build_subgroup()

        # Keep backward compatibility for code that may inspect this attr.
        self.nn_aux_form = self.optimizer_form

        self.optimizer_form.addRow('', self._field_with_description(self.optimizer_presets_widget, 'optimizer_presets'))
        self.optimizer_form.addRow('', self._field_with_description(self.optimizer_type, 'optimizer'))
        self.optimizer_form.addRow('', self._field_with_description(self.learning_rate_spinbox, 'learning_rate'))
        self.optimizer_form.addRow('', self._field_with_description(self.weight_decay_spinbox, 'weight_decay'))
        self.optimizer_form.addRow('', self._field_with_description(self.batch_spinbox, 'batch_size'))
        self.optimizer_form.addRow('', self._field_with_description(self.overlap_spinbox, 'overlap'))
        self.optimizer_form.addRow('', self._field_with_description(self.log_update_frequency_spinbox, 'log_update_frequency'))

        self.precision_loss_form.addRow('', self._field_with_description(self.mixed_precision_type, 'mixed_precision'))
        self.precision_loss_form.addRow('', self._field_with_description(self.loss_function_type, 'loss_function'))
        self.precision_loss_form.addRow('', self._field_with_description(self.dice_loss_weight_spinbox, 'dice_loss_weight'))
        self.precision_loss_form.addRow('', self._field_with_description(self.iou_loss_weight_spinbox, 'iou_loss_weight'))

        self.runtime_form.addRow(self.multi_gpu_check_box)
        self.runtime_form.addRow(self.torch_compile_check_box)
        self.runtime_form.addRow(self.skip_uniform_labels_check_box)

        self.warmup_form.addRow(self.warmup_check_box)
        self.warmup_form.addRow('', self._field_with_description(self.warmup_epochs_spinbox, 'warmup_epochs'))
        self.warmup_form.addRow('', self._field_with_description(self.warmup_start_factor_spinbox, 'warmup_start_factor'))

        self.hard_mining_form.addRow(self.hard_mining_check_box)
        self.hard_mining_form.addRow('', self._field_with_description(self.hard_mining_strength_spinbox, 'hard_mining_strength'))
        self.hard_mining_form.addRow('', self._field_with_description(self.hard_mining_ema_alpha_spinbox, 'hard_mining_ema_alpha'))

        self.early_stopping_form.addRow(self.early_stopping_check_box)
        self.early_stopping_form.addRow('', self._field_with_description(self.early_stopping_patience_spinbox, 'early_stop_patience'))
        self.early_stopping_form.addRow('', self._field_with_description(self.early_stopping_min_delta_spinbox, 'early_stop_min_delta'))
        self.early_stopping_form.addRow(self.restore_best_weights_check_box)

        nn_sections_layout.addWidget(self.optimizer_groupbox)
        nn_sections_layout.addWidget(self.precision_loss_groupbox)
        nn_sections_layout.addWidget(self.runtime_groupbox)
        nn_sections_layout.addWidget(self.warmup_groupbox)
        nn_sections_layout.addWidget(self.hard_mining_groupbox)
        nn_sections_layout.addWidget(self.early_stopping_groupbox)
        nn_sections_layout.addStretch(1)

        self._sync_active_optimizer_preset()
        self.warmup_check_box.toggled.connect(self._sync_warmup_controls)
        self.loss_function_type.currentIndexChanged.connect(self._sync_loss_controls)
        self.hard_mining_check_box.toggled.connect(self._sync_hard_mining_controls)
        self.early_stopping_check_box.toggled.connect(self._sync_early_stopping_controls)
        self._sync_warmup_controls(self.warmup_check_box.isChecked())
        self._sync_loss_controls(self.loss_function_type.currentIndex())
        self._sync_hard_mining_controls(self.hard_mining_check_box.isChecked())
        self._sync_early_stopping_controls(self.early_stopping_check_box.isChecked())

    def _init_layout(self):
        sample_size_widget = create_size_widget(self.sample_x_size, self.sample_y_size)
        self.sample_size_widget = sample_size_widget

        self.general_groupbox = QGroupBox('')
        self.general_form = self._create_form_layout()
        self.general_groupbox.setLayout(self.general_form)
        self.general_form.addRow(self.samples_number)
        self.general_form.addRow('', self._field_with_description(self.nn_model_type, 'model'))
        self.general_form.addRow('', self._field_with_description(self.color_type, 'image_format'))
        self.general_form.addRow('', self._field_with_description(self.sample_size_widget, 'sample_size'))

        self.augmentation_groupbox = QGroupBox('')
        self.augmentation_form = self._create_form_layout()
        self.augmentation_groupbox.setLayout(self.augmentation_form)
        self.augmentation_form.addRow(self.vertical_rotation)
        self.augmentation_form.addRow(self.horizontal_rotation)
        self.augmentation_form.addRow(self.additional_augmentation_check_box)
        self.augmentation_form.addRow(
            '',
            self._field_with_description(self.augmentation_brightness_spinbox, 'augmentation_brightness_strength'),
        )
        self.augmentation_form.addRow(
            '',
            self._field_with_description(self.augmentation_contrast_spinbox, 'augmentation_contrast_strength'),
        )
        self.augmentation_form.addRow(
            '',
            self._field_with_description(self.augmentation_noise_probability_spinbox, 'augmentation_noise_probability'),
        )
        self.augmentation_form.addRow(
            '',
            self._field_with_description(self.augmentation_noise_sigma_spinbox, 'augmentation_noise_sigma'),
        )
        self.augmentation_form.addRow('', self._field_with_description(self.shift_spinbox, 'shift'))
        self.additional_augmentation_check_box.toggled.connect(self._sync_augmentation_controls)
        self._sync_augmentation_controls(self.additional_augmentation_check_box.isChecked())

        self.validation_groupbox = QGroupBox('')
        self.validation_form = self._create_form_layout()
        self.validation_groupbox.setLayout(self.validation_form)
        self.validation_form.addRow(self.validation_check_box)
        self.validation_form.addRow('', self._field_with_description(self.validation_spinbox, 'validation_percent'))

        self.main_form = self.general_form
        self.reset_defaults_button = QPushButton('')

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(self.general_groupbox)
        layout.addWidget(self.augmentation_groupbox)
        layout.addWidget(self.validation_groupbox)
        layout.addWidget(self.sample_type_groupbox)
        layout.addWidget(self.prepare_samples_groupbox)
        layout.addWidget(self.nn_auxilary_settings_groupbox)

        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        reset_row.addStretch(1)
        reset_row.addWidget(self.reset_defaults_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(reset_row)
        layout.addStretch(1)

        self.general_form.setAlignment(self.nn_model_type, Qt.AlignmentFlag.AlignRight)
        self.general_form.setAlignment(self.color_type, Qt.AlignmentFlag.AlignRight)
        self.augmentation_form.setAlignment(self.shift_spinbox, Qt.AlignmentFlag.AlignRight)
        self.validation_form.setAlignment(self.validation_spinbox, Qt.AlignmentFlag.AlignRight)

        self._content_widget.setLayout(layout)

    def _apply_russian_texts(self):
        t = self._texts
        labels_map = t.get('labels', {}) if isinstance(t.get('labels', {}), dict) else {}
        descriptions = t.get('field_descriptions', {}) if isinstance(t.get('field_descriptions', {}), dict) else {}
        short_descriptions = (
            t.get('field_short_descriptions', {})
            if isinstance(t.get('field_short_descriptions', {}), dict)
            else {}
        )
        log_update_key = 'log_update_frequency'
        if log_update_key not in short_descriptions:
            short_descriptions[log_update_key] = str(labels_map.get(log_update_key, 'Частота логов (батчей)'))
        if log_update_key not in descriptions:
            descriptions[log_update_key] = str(
                t.get('log_update_frequency_tip', 'Частота обновления логов в батчах (0 = авто).')
            )

        self.horizontal_rotation.setText(str(t.get('rotate_90', 'Поворот кадра на 90 градусов')))
        self.horizontal_rotation.setToolTip(str(t.get('rotate_90_tip', '')))

        self.vertical_rotation.setText(str(t.get('rotate_180', 'Поворот кадра на 180 градусов')))
        self.vertical_rotation.setToolTip(str(t.get('rotate_180_tip', '')))
        self.additional_augmentation_check_box.setText(
            str(t.get('extra_aug', 'Дополнительная аугментация образцов'))
        )
        self.additional_augmentation_check_box.setToolTip(str(t.get('extra_aug_tip', '')))
        self.augmentation_brightness_spinbox.setToolTip(str(t.get('extra_aug_brightness_tip', '')))
        self.augmentation_contrast_spinbox.setToolTip(str(t.get('extra_aug_contrast_tip', '')))
        self.augmentation_noise_probability_spinbox.setToolTip(str(t.get('extra_aug_noise_prob_tip', '')))
        self.augmentation_noise_sigma_spinbox.setToolTip(str(t.get('extra_aug_noise_sigma_tip', '')))

        self.validation_check_box.setText(str(t.get('validation', 'Использовать валидацию при обучении')))
        self.validation_check_box.setToolTip(str(t.get('validation_tip', '')))

        self.samples_number.setText(str(t.get('samples_count', 'Кадров в выборке: 0')))
        self.shift_spinbox.setToolTip(str(t.get('shift_tip', '')))

        self.general_groupbox.setTitle(str(t.get('general_group', 'Данные и модель')))
        self.augmentation_groupbox.setTitle(str(t.get('augmentation_group', 'Аугментации и сдвиг')))
        self.validation_groupbox.setTitle(str(t.get('validation_group', 'Валидация')))
        self.sample_type_groupbox.setTitle(str(t.get('sample_group', 'Метод работы с выборкой')))
        self.sample_type_groupbox.setToolTip(str(t.get('sample_group_tip', '')))
        self.cut_dataset_type.setText(str(t.get('cut_to_disk', 'Нарезка в файл')))
        self.no_cut_dataset_type.setText(str(t.get('cut_online', 'Нарезка на лету')))

        self.prepare_samples_groupbox.setTitle(str(t.get('preprocess_group', 'Предварительная обработка образцов')))
        self.enable_crop_processing.setText(str(t.get('preprocess_crop_enable', 'Включить обрезку края')))
        self.enable_resize_processing.setText(str(t.get('preprocess_resize_enable', 'Включить изменение размера')))

        self.nn_auxilary_settings_groupbox.setTitle(str(t.get('aux_group', 'Дополнительные настройки')))
        self.optimizer_groupbox.setTitle(str(t.get('optimizer_group', 'Оптимизатор и батч')))
        self.precision_loss_groupbox.setTitle(str(t.get('loss_precision_group', 'Loss и mixed precision')))
        self.runtime_groupbox.setTitle(str(t.get('runtime_group', 'Ускорение и фильтрация')))
        self.warmup_groupbox.setTitle(str(t.get('warmup_group', 'Warmup')))
        self.hard_mining_groupbox.setTitle(str(t.get('hard_mining_group', 'Hard mining')))
        self.early_stopping_groupbox.setTitle(str(t.get('early_stopping_group', 'Ранняя остановка')))
        self.optimizer_type.setToolTip(str(t.get('optimizer_tip', '')))
        self.mixed_precision_type.setToolTip(str(t.get('mixed_precision_tip', '')))
        self.loss_function_type.setToolTip(str(t.get('loss_function_tip', '')))
        self.dice_loss_weight_spinbox.setToolTip(str(t.get('dice_loss_weight_tip', '')))
        self.iou_loss_weight_spinbox.setToolTip(str(t.get('iou_loss_weight_tip', '')))
        self.learning_rate_spinbox.setToolTip(str(t.get('lr_tip', '')))
        self.weight_decay_spinbox.setToolTip(str(t.get('wd_tip', '')))
        self.batch_spinbox.setToolTip(str(t.get('batch_tip', '')))
        self.overlap_spinbox.setToolTip(str(t.get('overlap_tip', '')))
        self.log_update_frequency_spinbox.setToolTip(
            str(t.get('log_update_frequency_tip', 'Частота обновления логов в батчах (0 = авто).'))
        )

        self.multi_gpu_check_box.setText(str(t.get('multi_gpu', 'Использовать multi-GPU (если доступно)')))
        self.multi_gpu_check_box.setToolTip(str(t.get('multi_gpu_tip', '')))
        self.torch_compile_check_box.setText(str(t.get('torch_compile_enable', 'Включить torch.compile')))
        self.torch_compile_check_box.setToolTip(
            str(t.get('torch_compile_tip', 'Компиляция графа PyTorch для ускорения, если поддерживается.'))
        )
        self.warmup_check_box.setText(str(t.get('warmup_enable', 'Включить warmup')))
        self.warmup_check_box.setToolTip(str(t.get('warmup_tip', '')))
        self.hard_mining_check_box.setText(str(t.get('hard_mining_enable', 'Чаще обучать на сложных примерах (high loss)')))
        self.hard_mining_check_box.setToolTip(str(t.get('hard_mining_tip', '')))
        self.skip_uniform_labels_check_box.setText(
            str(t.get('skip_uniform_labels', 'Пропускать сэмплы, где label полностью 0 или 1'))
        )
        self.skip_uniform_labels_check_box.setToolTip(str(t.get('skip_uniform_labels_tip', '')))
        self.early_stopping_check_box.setText(str(t.get('early_stopping_enable', 'Включить раннюю остановку')))
        self.early_stopping_check_box.setToolTip(str(t.get('early_stopping_tip', '')))
        self.restore_best_weights_check_box.setText(str(t.get('restore_best', 'Восстановить лучшие веса')))
        self.restore_best_weights_check_box.setToolTip(str(t.get('restore_best_tip', '')))
        self.reset_defaults_button.setText(str(t.get('reset_defaults', 'По умолчанию')))
        self.reset_defaults_button.setToolTip(str(t.get('reset_defaults_tip', 'Сбросить все параметры к исходным значениям.')))

        color_modes = t.get('color_modes', ['ЧБ', 'RGB'])
        if isinstance(color_modes, list) and color_modes:
            self.color_type.clear()
            self.color_type.addItems([str(v) for v in color_modes])

        labels = (
            (getattr(self, 'augmentation_form', None), self.shift_spinbox, str(labels_map.get('shift', 'Смещение'))),
            (
                getattr(self, 'augmentation_form', None),
                self.augmentation_brightness_spinbox,
                str(labels_map.get('augmentation_brightness_strength', 'Сила яркости')),
            ),
            (
                getattr(self, 'augmentation_form', None),
                self.augmentation_contrast_spinbox,
                str(labels_map.get('augmentation_contrast_strength', 'Сила контраста')),
            ),
            (
                getattr(self, 'augmentation_form', None),
                self.augmentation_noise_probability_spinbox,
                str(labels_map.get('augmentation_noise_probability', 'Вероятность шума')),
            ),
            (
                getattr(self, 'augmentation_form', None),
                self.augmentation_noise_sigma_spinbox,
                str(labels_map.get('augmentation_noise_sigma', 'Сила шума')),
            ),
            (getattr(self, 'general_form', None), self.nn_model_type, str(labels_map.get('model', 'Модель'))),
            (getattr(self, 'general_form', None), self.color_type, str(labels_map.get('image_format', 'Формат изображения'))),
            (
                getattr(self, 'general_form', None),
                getattr(self, 'sample_size_widget', self.sample_x_size),
                str(labels_map.get('sample_size', 'Размер выборки')),
            ),
            (
                getattr(self, 'validation_form', None),
                self.validation_spinbox,
                str(labels_map.get('validation_percent', 'Процент валидации')),
            ),
            (getattr(self, 'prepare_samples_form', None), self.cut_corner_spinbox, str(labels_map.get('edge_cut', 'Срез края'))),
            (getattr(self, 'prepare_samples_form', None), getattr(self, 'target_size_widget', self.target_x_size), str(labels_map.get('target_size', 'Целевой размер'))),
            (getattr(self, 'optimizer_form', None), self.optimizer_type, str(labels_map.get('optimizer', 'Оптимизатор'))),
            (getattr(self, 'precision_loss_form', None), self.mixed_precision_type, str(labels_map.get('mixed_precision', 'Mixed precision'))),
            (getattr(self, 'precision_loss_form', None), self.loss_function_type, str(labels_map.get('loss_function', 'Функция потерь'))),
            (getattr(self, 'precision_loss_form', None), self.dice_loss_weight_spinbox, str(labels_map.get('dice_loss_weight', 'Вес Dice loss'))),
            (getattr(self, 'precision_loss_form', None), self.iou_loss_weight_spinbox, str(labels_map.get('iou_loss_weight', 'Вес IoU loss'))),
            (getattr(self, 'optimizer_form', None), self.learning_rate_spinbox, str(labels_map.get('learning_rate', 'Learning rate'))),
            (getattr(self, 'optimizer_form', None), self.weight_decay_spinbox, str(labels_map.get('weight_decay', 'Weight decay'))),
            (getattr(self, 'optimizer_form', None), self.batch_spinbox, str(labels_map.get('batch_size', 'Размер батча'))),
            (getattr(self, 'optimizer_form', None), self.overlap_spinbox, str(labels_map.get('overlap', 'Перекрытие'))),
            (
                getattr(self, 'optimizer_form', None),
                self.log_update_frequency_spinbox,
                str(labels_map.get('log_update_frequency', 'Частота логов (батчей)')),
            ),
            (getattr(self, 'warmup_form', None), self.warmup_epochs_spinbox, str(labels_map.get('warmup_epochs', 'Эпохи warmup'))),
            (getattr(self, 'warmup_form', None), self.warmup_start_factor_spinbox, str(labels_map.get('warmup_start_factor', 'Warmup start factor'))),
            (getattr(self, 'hard_mining_form', None), self.hard_mining_strength_spinbox, str(labels_map.get('hard_mining_strength', 'Hard mining strength'))),
            (getattr(self, 'hard_mining_form', None), self.hard_mining_ema_alpha_spinbox, str(labels_map.get('hard_mining_ema_alpha', 'Hard mining EMA alpha'))),
            (getattr(self, 'early_stopping_form', None), self.early_stopping_patience_spinbox, str(labels_map.get('early_stop_patience', 'Patience ранней остановки'))),
            (getattr(self, 'early_stopping_form', None), self.early_stopping_min_delta_spinbox, str(labels_map.get('early_stop_min_delta', 'Min delta ранней остановки'))),
        )
        for form, field, text in labels:
            if form is None:
                continue
            target = self._field_rows.get(field, field)
            label = form.labelForField(target)
            if label is not None:
                label.setText(text)
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for key, label in self._desc_labels.items():
            short_text = str(short_descriptions.get(key, descriptions.get(key, '')))
            detailed_text = str(descriptions.get(key, short_text))
            label.setText(short_text)
            label.setToolTip(detailed_text)
            field = self._desc_fields.get(key)
            if field is not None:
                self._apply_tooltip_to_widget_and_children(field, detailed_text)

    def connect_internal_signals(self):
        self.horizontal_rotation.clicked.connect(lambda: self.horisontal_rotate_clicked.emit())
        self.vertical_rotation.clicked.connect(lambda: self.vertical_rotate_clicked.emit())
        self.additional_augmentation_check_box.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.additional_augmentation_check_box.toggled.connect(lambda: self.cut_slider_shifted.emit())
        self.augmentation_brightness_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.augmentation_contrast_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.augmentation_noise_probability_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.augmentation_noise_sigma_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.shift_spinbox.valueChanged.connect(lambda: self.cut_slider_shifted.emit())
        self.nn_model_type.currentIndexChanged.connect(lambda: self.model_changed.emit())
        self.sample_x_size.valueChanged.connect(lambda: self.sample_size_changed.emit())
        self.sample_y_size.valueChanged.connect(lambda: self.sample_size_changed.emit())
        self.optimizer_type.currentIndexChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.learning_rate_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.weight_decay_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.mixed_precision_type.currentIndexChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.loss_function_type.currentIndexChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.dice_loss_weight_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.iou_loss_weight_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.optimizer_type.currentIndexChanged.connect(self._sync_active_optimizer_preset)
        self.learning_rate_spinbox.valueChanged.connect(self._sync_active_optimizer_preset)
        self.weight_decay_spinbox.valueChanged.connect(self._sync_active_optimizer_preset)
        self.multi_gpu_check_box.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.torch_compile_check_box.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.warmup_check_box.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.warmup_epochs_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.warmup_start_factor_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.hard_mining_check_box.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.hard_mining_strength_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.hard_mining_ema_alpha_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.log_update_frequency_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.skip_uniform_labels_check_box.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.early_stopping_check_box.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.early_stopping_patience_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.early_stopping_min_delta_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.restore_best_weights_check_box.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.validation_check_box.toggled.connect(lambda: self.validation_settings_changed.emit())
        self.validation_spinbox.valueChanged.connect(lambda: self.validation_settings_changed.emit())
        self.enable_crop_processing.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.enable_resize_processing.toggled.connect(lambda: self.optimizer_settings_changed.emit())
        self.cut_corner_spinbox.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.target_x_size.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.target_y_size.valueChanged.connect(lambda: self.optimizer_settings_changed.emit())
        self.reset_defaults_button.clicked.connect(self.reset_defaults_requested.emit)
        self._sync_active_optimizer_preset()

    def _apply_optimizer_preset(self, optimizer_name: str, learning_rate: float, weight_decay: float):
        self.optimizer_type.setCurrentText(optimizer_name)
        self.learning_rate_spinbox.setValue(learning_rate)
        self.weight_decay_spinbox.setValue(weight_decay)
        self._sync_active_optimizer_preset()
        self.optimizer_settings_changed.emit()

    def _sync_active_optimizer_preset(self):
        current_optimizer = self.optimizer_type.currentText()
        current_learning_rate = float(self.learning_rate_spinbox.value())
        current_weight_decay = float(self.weight_decay_spinbox.value())
        for btn, (_title, optimizer_name, learning_rate, weight_decay) in zip(self.optimizer_preset_buttons, OPTIMIZER_PRESETS):
            is_active = (
                current_optimizer == optimizer_name
                and abs(current_learning_rate - learning_rate) < 1e-12
                and abs(current_weight_decay - weight_decay) < 1e-12
            )
            btn.setChecked(is_active)

    def _sync_validation_controls(self, enabled: bool):
        self._set_field_enabled(
            self.validation_spinbox,
            self._training_controls_applicable and bool(enabled),
        )

    def _sync_warmup_controls(self, enabled: bool):
        control_enabled = self._training_controls_applicable and bool(enabled)
        self._set_field_enabled(self.warmup_epochs_spinbox, control_enabled)
        self._set_field_enabled(self.warmup_start_factor_spinbox, control_enabled)

    def _sync_hard_mining_controls(self, enabled: bool):
        control_enabled = self._training_controls_applicable and bool(enabled)
        self._set_field_enabled(self.hard_mining_strength_spinbox, control_enabled)
        self._set_field_enabled(self.hard_mining_ema_alpha_spinbox, control_enabled)

    def _sync_loss_controls(self, _index: int):
        mode = self.loss_function_type.currentText()
        self._set_field_enabled(
            self.dice_loss_weight_spinbox,
            self._training_controls_applicable and mode == 'bce_dice',
        )
        self._set_field_enabled(
            self.iou_loss_weight_spinbox,
            self._training_controls_applicable and mode == 'bce_iou',
        )

    def _sync_early_stopping_controls(self, enabled: bool):
        control_enabled = self._training_controls_applicable and bool(enabled)
        self._set_field_enabled(self.early_stopping_patience_spinbox, control_enabled)
        self._set_field_enabled(self.early_stopping_min_delta_spinbox, control_enabled)
        self.restore_best_weights_check_box.setEnabled(control_enabled)

    def _sync_preprocess_controls(self, _enabled: bool | None = None):
        self._set_field_enabled(
            self.cut_corner_spinbox,
            self._training_controls_applicable and self.enable_crop_processing.isChecked(),
        )
        self._set_field_enabled(
            self.target_size_widget,
            self._training_controls_applicable and self.enable_resize_processing.isChecked(),
        )

    def _sync_augmentation_controls(self, enabled: bool):
        control_enabled = self._training_controls_applicable and bool(enabled)
        self._set_field_enabled(self.augmentation_brightness_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_contrast_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_noise_probability_spinbox, control_enabled)
        self._set_field_enabled(self.augmentation_noise_sigma_spinbox, control_enabled)

    def model_type_init(self, models):
        self.models = models
        self.nn_model_type.clear()
        for name in models:
            self.nn_model_type.addItem(name)

    def set_model(self, model: str):
        if model not in self.models:
            return
        index = self.models.index(model)
        self.nn_model_type.setCurrentIndex(index)

    def restore_cut_mode(self, mode):
        if mode == SampleCutMode.disk.value:
            self.cut_dataset_type.setChecked(True)
        else:
            self.no_cut_dataset_type.setChecked(True)

    def apply_style(self, style):
        self.setStyleSheet(style)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = SettingsPanel()
    window.show()
    sys.exit(app.exec())





