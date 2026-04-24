# presenter/main_presenter.py
import multiprocessing as mp
import gc
import os
import threading
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QInputDialog
from PyQt6 import QtCore, QtWidgets

from neuralimage.application.dto import (
    MainWindowState,
    SettingsState,
    build_main_window_mode_state_entry,
    build_main_window_mode_state_entry_from_state,
    clone_main_window_state,
    normalize_main_window_mode_state,
    resolve_main_window_mode_state_entry,
)
from neuralimage.application.ports import StateStore
from neuralimage.application.services import (
    ActiveTaskMutationError,
    ProcessingSession,
    QueuedTask,
    build_processing_start_error_message,
    build_workflow_parameters,
    can_start_processing,
)
from neuralimage.application.services.training_artifacts import build_training_artifact_dir
from neuralimage.infrastructure.config.state_store import (
    WORKFLOW_SNAPSHOT_FILENAME,
    load_workflow_snapshot,
    save_workflow_snapshot,
)
from neuralimage.lib.data_interfaces import (
    CutSettings,
    SampleCutMode,
    WorkMode,
    TrainingParameters,
    RecognitionParameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
    normalize_work_mode,
    normalize_multi_gpu_mode,
    normalize_validation_source,
)
from neuralimage.lib.file_func import filter_files
from neuralimage.lib.loss_config import dominant_loss_function, resolve_loss_term_weights
from neuralimage.lib.logging_policy import should_forward_log_event
from neuralimage.lib.message_bus import MessageBus, AbstractMessageBus
from neuralimage.lib.runtime_paths import resolve_resource_path
from neuralimage.lib.images import SampleWorker
from neuralimage.lib.update_checker import (
    ReleaseInfo,
    UpdateInfo,
    collect_release_history,
    download_update_installer,
    fetch_update_info,
    is_newer_version,
    launch_update_installer,
    load_selected_update_channel,
    load_update_client_config,
    load_last_notified_version,
    load_update_manifest_url,
    save_last_notified_version,
    save_selected_update_channel,
    should_notify_version,
)
from neuralimage.lib.version import APP_VERSION
from neuralimage.model.NeuralNetwork import get_registered_model_names_by_type
from neuralimage.model.general_neural_handler import GeneralNeuralHandler
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.view import MainView, SettingsPanel
from neuralimage.view.developer_tools_dialog import DeveloperToolsDialog
from neuralimage.view.task_properties_dialog import TaskPropertiesDialog
import neuralimage.presenter.dialogs as presenter_dialogs
import neuralimage.presenter.plugin_flow as plugin_flow
import neuralimage.presenter.sample_counting as sample_counting
import neuralimage.presenter.state_sync as state_sync
import neuralimage.presenter.task_flow as task_flow
import neuralimage.presenter.update_flow as update_flow
from .background_workers import (
    AppUpdateCheckThread as PresenterAppUpdateCheckThread,
    AppUpdateDownloadThread as PresenterAppUpdateDownloadThread,
    GeneralNeuralHandlerThread as PresenterGeneralNeuralHandlerThread,
    RarePatchEditorPreparationThread as PresenterRarePatchEditorPreparationThread,
)


class _ValidationGradientPluginWindow(QtWidgets.QMainWindow):
    """Host one lite plugin widget as an independent top-level window."""

    def __init__(self, plugin, widget, title: str, on_closed: Callable[[], None], parent=None):
        super().__init__(parent)
        self._plugin = plugin
        self._on_closed = on_closed
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setWindowTitle(str(title))
        self.setCentralWidget(widget)
        self.resize(1400, 900)

    def closeEvent(self, event) -> None:
        try:
            if self._plugin is not None:
                self._plugin.shutdown()
        finally:
            self._plugin = None
            try:
                self._on_closed()
            finally:
                super().closeEvent(event)




def _format_auto_answer_button_text(text: str, seconds_left: int) -> str:
    return presenter_dialogs.format_auto_answer_button_text(text, seconds_left)


def _tk_filedialog(kind: str, filetypes=None) -> str | None:
    return presenter_dialogs.choose_path_dialog(kind, filetypes)


class SampleCountSignals(QObject):
    calculated = QtCore.pyqtSignal(int, str, object, int)
    failed = QtCore.pyqtSignal(int, str)


def _should_schedule_startup_update_check() -> bool:
    disabled_flag = str(os.getenv('NEURALIMAGE_DISABLE_AUTO_UPDATE_CHECK', '') or '').strip().lower()
    if disabled_flag in {'1', 'true', 'yes', 'on'}:
        return False
    return not bool(os.getenv('PYTEST_CURRENT_TEST'))


class MainPresenter(QObject):
    SIMPLE_WORKFLOW_PRESETS = {
        'conductors': resolve_resource_path('conductors_workflow.json'),
        'contacts': resolve_resource_path('contacts_workflow.json'),
        'memory': resolve_resource_path('memory_workflow.json'),
    }

    """
    Связывает View и Model. Содержит всю бизнес-логику
    (валидацию, формирование параметров, запуск/остановку потока).
    """

    def __init__(self, state_store: StateStore):
        super().__init__()
        self._state_store = state_store

        self.sample_calculator = SampleWorker()

        # Инициализируем панель с настройками нейросети
        self.settings_panel = SettingsPanel()

        # Получаем список активных моделей нейросетей
        self.active_nn_models = {}
        self.settings_panel.model_type_init(get_registered_model_names_by_type())

        self._update_client_config = load_update_client_config()
        self._selected_update_channel = load_selected_update_channel(
            self._update_client_config.default_channel,
            available_channels=self._update_client_config.available_channels,
        )

        # Инициализируем главное окно
        self.view = MainView(side_panel=self.settings_panel)
        self.view.configure_update_channels(
            self._update_client_config.available_channels,
            self._selected_update_channel,
        )

        self.stop_event = mp.Event()
        self._processing_session = ProcessingSession()
        self.neuaral_handler: GeneralNeuralHandlerThread | None = None
        self._rare_patch_editor_prepare_thread: RarePatchEditorPreparationThread | None = None
        self._augmentation_preview_dialog = None
        self._update_check_thread: AppUpdateCheckThread | None = None
        self._update_download_thread: AppUpdateDownloadThread | None = None
        self._validation_gradient_plugin = None
        self._validation_gradient_window: _ValidationGradientPluginWindow | None = None
        self._developer_tools_dialog: DeveloperToolsDialog | None = None
        self._update_check_manual = False
        self._sample_count_worker_thread: threading.Thread | None = None
        self._sample_count_request_serial = 0
        self._latest_sample_count_request_id = 0
        self._debounced_sample_count_request: tuple[int, str, CutSettings, object] | None = None
        self._pending_sample_count_request: tuple[int, str, CutSettings, object] | None = None
        self._sample_count_cache_path: str | None = None
        self._sample_count_cache_sizes: list[tuple[int, int]] | None = None
        self._sample_count_signals = SampleCountSignals()
        self._sample_count_signals.calculated.connect(self._on_sample_count_calculated)
        self._sample_count_signals.failed.connect(self._on_sample_count_failed)
        self._sample_count_debounce_timer = QtCore.QTimer(self)
        self._sample_count_debounce_timer.setSingleShot(True)
        self._sample_count_debounce_timer.setInterval(150)
        self._sample_count_debounce_timer.timeout.connect(self._dispatch_sample_count_request)

        # --------------------- 1. Подписка на сигналы View --------------------- #
        self._conncet_to_message_bus()
        self._connect_view_signals()
        self._connect_settings_signal()

        # --------------------- 3. Инициализация UI из Config --------------------- #
        self._load_initial_state()
        self._set_initial_sample_count_state()
        self.view.show()
        QtCore.QTimer.singleShot(0, self._calculate_expected_samples)
        if _should_schedule_startup_update_check():
            QtCore.QTimer.singleShot(0, self._start_update_check)



    # ------------------------------------------------------------------ #
    #   Подписки
    # ------------------------------------------------------------------ #
    def _conncet_to_message_bus(self):
        self.message_bus = MessageBus()
        self.message_bus.subscribe('logging', self._log_message_emit)
        self.message_bus.subscribe('training', self._train_message_emit)
        self.message_bus.subscribe('metrics', self._metrics_message_emit)
        self.message_bus.subscribe('error', self._error_message_emit)

    def _connect_view_signals(self):
        v = self.view
        v.sample_type_changed.connect(self._on_sample_type_changed)
        v.le_epochs.valueChanged.connect(
            lambda value: self.settings_panel.epochs_spinbox.setValue(int(value))
        )

        v.source_path_requested.connect(self._choose_source_folder)
        v.result_path_requested.connect(self._choose_result_folder)

        v.jpg_path_requested.connect(self._choose_jpg_label_path)
        v.label_path_requested.connect(self._chosse_cif_label_path)

        v.model_path_requested.connect(self._choose_model_path)
        v.open_config_requested.connect(self._on_open_config_requested)

        v.start_requested.connect(self._on_start_requested)
        v.stop_requested.connect(self._on_stop_requested)
        v.queue_remove_requested.connect(self._on_queue_remove_requested)
        v.queue_pause_toggle_requested.connect(self._on_queue_pause_toggle_requested)
        v.queue_context_remove_requested.connect(self._on_queue_context_remove_requested)
        v.queue_properties_requested.connect(self._on_queue_properties_requested)
        v.batch_preview_visibility_changed.connect(self._on_batch_preview_visibility_changed)
        v.release_memory_requested.connect(self._on_release_memory_requested)
        v.open_validation_gradient_requested.connect(self._on_open_validation_gradient_requested)
        v.developer_tools_requested.connect(self._on_developer_tools_requested)
        v.update_check_requested.connect(self._on_update_check_requested)
        v.update_channel_selected.connect(self._on_update_channel_selected)
        v.ui_language_selected.connect(self._on_ui_language_selected)
        v.theme_selected.connect(self._on_theme_selected)
        v.ui_mode_selected.connect(self._on_ui_mode_selected)
        v.simple_workflow_requested.connect(self._on_simple_workflow_requested)
        if hasattr(v, 'recursive_file_search_changed'):
            v.recursive_file_search_changed.connect(self._on_recursive_file_search_changed)

        v.request_close.connect(self._on_close_requested)

    def _connect_settings_signal(self):
        v = self.settings_panel
        v.epochs_spinbox.valueChanged.connect(lambda value: self.view.le_epochs.setValue(int(value)))
        v.cut_slider_shifted.connect(self._calculate_expected_samples)
        v.horisontal_rotate_clicked.connect(self._calculate_expected_samples)
        v.vertical_rotate_clicked.connect(self._calculate_expected_samples)
        v.model_changed.connect(self._calculate_expected_samples)
        v.sample_size_changed.connect(self._set_max_shift)
        v.optimizer_settings_changed.connect(self._update_settings_window_state)
        v.validation_settings_changed.connect(self._update_settings_window_state)
        v.validation_image_path_requested.connect(self._choose_validation_image_folder)
        v.validation_label_path_requested.connect(self._choose_validation_label_folder)
        v.reset_defaults_requested.connect(self._reset_settings_to_defaults)
        v.augmentation_preview_requested.connect(self._open_augmentation_preview)
        v.ui_language_changed.connect(self.view.apply_ui_language)
        v.rare_patch_editor_requested.connect(self._open_rare_patch_editor)

    def _on_recursive_file_search_changed(self) -> None:
        self._update_settings_window_state()
        self._calculate_expected_samples()

    def _publish_log_message(self, message: str) -> None:
        state_dict = object.__getattribute__(self, '__dict__')
        bus = state_dict.get('message_bus')
        if bus is None:
            return
        try:
            bus.publish('logging', str(message))
        except Exception:
            return

    # ------------------------------------------------------------------ #
    #   Инициализация UI
    # ------------------------------------------------------------------ #
    def _load_initial_state(self):
        state_sync.load_initial_state(self)

    def _load_main_window_settings(self):
        state_sync.load_main_window_settings(self)

    def _load_settings_panel_settings(self):
        state_sync.load_settings_panel_settings(self)

    def _set_initial_sample_count_state(self) -> None:
        state_sync.set_initial_sample_count_state(self)

    # ------------------------------------------------------------------ #
    #   Обновление класса состояния окон
    # ------------------------------------------------------------------ #

    def _current_view_main_window_mode_entry(self) -> dict[str, object]:
        return state_sync.current_view_main_window_mode_entry(self)

    def _store_view_state_for_mode(self, mode: str) -> None:
        state_sync.store_view_state_for_mode(self, mode)

    def _restore_mode_state_to_view(self, mode: str) -> None:
        state_sync.restore_mode_state_to_view(self, mode)

    def _update_main_window_state(self):
        state_sync.update_main_window_state(self)

    def _restore_work_mode_ui(self):
        state_sync.restore_work_mode_ui(self)

    def _apply_settings_to_panel(self) -> None:
        """
        Populate the widgets in ``self.settings_panel`` with the values stored in
        ``state`` (an instance of :class:`SettingsState`).

        The method mirrors the logic of ``_update_settings_window_state`` - every
        attribute that is read there is written back here.

        Parameters
        ----------
        state:
            An object that contains the same attribute names that
            ``_update_settings_window_state`` reads (e.g. ``step``, ``vertical_rotation``,
            ``horizontal_rotation`` ...). It can be a ``SettingsState`` dataclass,
            a ``namedtuple`` or any plain object with those attributes.

        Notes
        -----
        * Signals emitted by the widgets are **temporarily blocked** while the UI is
          being updated. This prevents spurious "value-changed" callbacks from
          running before the whole form is in a consistent state.
        * If a widget is optional (e.g. it may be ``None`` in some UI versions) we
          guard against ``AttributeError`` so the function is safe to reuse across
          different dialog layouts.
        """
        return state_sync.apply_settings_to_panel(self)
        # ------------------------------------------------------------------
        # 1. Берем короткую ссылку на панель для компактности кода.
        # ------------------------------------------------------------------
        s = self.settings_panel
        state = self.settings_state
        # --------------------------------------------------------------
        # 3. Записываем каждое поле состояния обратно в виджет.
        # --------------------------------------------------------------
        # 3.1 Check-box (bool)
        s.horizontal_rotation.setChecked(state.horizontal_rotation)
        s.vertical_rotation.setChecked(state.vertical_rotation)
        s.flip_x.setChecked(bool(getattr(state, 'flip_x', False)))
        s.flip_y.setChecked(bool(getattr(state, 'flip_y', False)))
        s.additional_augmentation_check_box.setChecked(state.additional_augmentation)
        s.augmentation_brightness_spinbox.setValue(state.augmentation_brightness_strength)
        s.augmentation_contrast_spinbox.setValue(state.augmentation_contrast_strength)
        s.augmentation_gamma_spinbox.setValue(float(getattr(state, 'augmentation_gamma_strength', 0.15)))
        s.augmentation_noise_probability_spinbox.setValue(state.augmentation_noise_probability)
        s.augmentation_noise_sigma_spinbox.setValue(state.augmentation_noise_sigma)
        s.augmentation_blur_probability_spinbox.setValue(float(getattr(state, 'augmentation_blur_probability', 0.25)))
        s.augmentation_blur_radius_spinbox.setValue(float(getattr(state, 'augmentation_blur_radius', 1.0)))
        if hasattr(s, '_sync_augmentation_controls'):
            s._sync_augmentation_controls(state.additional_augmentation)

        # 3.2  Spin boxes (int/float)
        s.shift_spinbox.setValue(state.step)
        train_patch_size = tuple(getattr(state, 'train_patch_size', None) or state.sample_size)
        recognition_patch_size = tuple(getattr(state, 'recognition_patch_size', None) or state.sample_size)
        s.train_patch_x_size.setValue(train_patch_size[0])
        s.train_patch_y_size.setValue(train_patch_size[1])
        s.recognition_patch_x_size.setValue(recognition_patch_size[0])
        s.recognition_patch_y_size.setValue(recognition_patch_size[1])

        # 3.3 Combo-box (str - используем setCurrentText, который выбирает
        #      entry if it exists, otherwise it will add a new entry.)
        s.set_model(state.model)
        if hasattr(s, 'set_color_mode_value'):
            s.set_color_mode_value(state.color_mode)
        else:
            s.color_type.setCurrentText(state.color_mode)
        s.shuffle_frames_check_box.setChecked(bool(getattr(state, 'shuffle', True)))
        s.shuffle_patches_in_frame_check_box.setChecked(
            bool(getattr(state, 'shuffle_patches_in_frame', getattr(state, 'shuffle', True)))
        )
        s.random_crop_check_box.setChecked(bool(getattr(state, 'random_crop', False)))
        s.crops_per_image_spinbox.setValue(int(getattr(state, 'crops_per_image', 64)))
        s.scale_augmentation_check_box.setChecked(bool(getattr(state, 'scale_augmentation', False)))
        s.scale_augmentation_strength_spinbox.setValue(float(getattr(state, 'scale_augmentation_strength', 0.2)))
        if hasattr(s, 'set_synthetic_defect_generator_config'):
            s.set_synthetic_defect_generator_config(getattr(state, 'synthetic_defect_generator', {}))
        s.cutout_check_box.setChecked(bool(getattr(state, 'cutout_enabled', False)))
        s.cutout_probability_spinbox.setValue(float(getattr(state, 'cutout_probability', 1.0)))
        s.cutout_holes_spinbox.setValue(int(getattr(state, 'cutout_holes', 1)))
        s.cutout_size_ratio_spinbox.setValue(float(getattr(state, 'cutout_size_ratio', 0.25)))
        s.random_artifacts_check_box.setChecked(bool(getattr(state, 'random_artifacts_enabled', False)))
        s.random_artifacts_probability_spinbox.setValue(float(getattr(state, 'random_artifacts_probability', 1.0)))
        s.random_artifacts_count_spinbox.setValue(int(getattr(state, 'random_artifacts_count', 1)))
        s.random_artifacts_size_ratio_spinbox.setValue(
            float(getattr(state, 'random_artifacts_size_ratio', 0.25))
        )
        for artifact_name, checkbox in getattr(s, 'random_artifact_type_checkboxes', {}).items():
            checkbox.setChecked(bool(getattr(state, f'random_artifacts_{artifact_name}_enabled', True)))
        s.mixup_check_box.setChecked(bool(getattr(state, 'mixup_enabled', False)))
        s.mixup_probability_spinbox.setValue(float(getattr(state, 'mixup_probability', 1.0)))
        s.mixup_alpha_spinbox.setValue(float(getattr(state, 'mixup_alpha', 0.2)))

        # 3.4  Validation controls
        s.validation_check_box.setChecked(state.use_validation)
        s.validation_spinbox.setValue(state.validation_percent)
        s.set_validation_source_value(
            normalize_validation_source(getattr(state, 'validation_source', 'split'))
        )
        s.set_validation_image_path(str(getattr(state, 'validation_image_folder', '')))
        s.set_validation_label_path(str(getattr(state, 'validation_label_folder', '')))
        s.save_validation_binary_images_check_box.setChecked(
            bool(getattr(state, 'save_validation_binary_images', False))
        )

        # 3.5 Режим нарезки

        s.restore_cut_mode(state.sample_cut_mode)
        if hasattr(s, '_sync_augmentation_controls'):
            s._sync_augmentation_controls(state.additional_augmentation)
        if hasattr(s, '_sync_training_augmentation_controls'):
            s._sync_training_augmentation_controls()
        s.epochs_spinbox.setValue(int(getattr(self.main_window_state, 'epochs', 20)))

        # 3.6 Размер батча / overlap
        train_batch_size = int(getattr(state, 'train_batch_size', None) or state.batch_size)
        recognition_batch_size = int(getattr(state, 'recognition_batch_size', None) or state.batch_size)
        s.train_batch_spinbox.setValue(train_batch_size)
        s.dataloader_num_workers_spinbox.setValue(int(getattr(state, 'dataloader_num_workers', -1)))
        s.recognition_batch_spinbox.setValue(recognition_batch_size)
        s.overlap_spinbox.setValue(state.overlap)
        s.recognition_jpeg_quality_spinbox.setValue(int(getattr(state, 'recognition_jpeg_quality', 95)))
        s.recognition_multiprocessing_check_box.setChecked(
            bool(getattr(state, 'recognition_multiprocessing_enabled', True))
        )
        s.recognition_binarize_output_check_box.setChecked(bool(getattr(state, 'recognition_binarize_output', True)))
        s.recognition_use_auto_threshold_check_box.setChecked(
            bool(getattr(state, 'recognition_use_auto_threshold', True))
        )
        s.recognition_threshold_spinbox.setValue(float(getattr(state, 'recognition_threshold', 0.5)))
        s.recognition_tta_check_box.setChecked(bool(getattr(state, 'recognition_tta_enabled', False)))
        s.recognition_postprocess_check_box.setChecked(bool(getattr(state, 'recognition_postprocess', False)))
        s.recognition_postprocess_kernel_size_spinbox.setValue(
            int(getattr(state, 'recognition_postprocess_kernel_size', 3))
        )
        if hasattr(s, 'set_confidence_output_mode'):
            s.set_confidence_output_mode(
                str(getattr(state, 'confidence_save_mode', 'off')),
                bool(getattr(state, 'confidence_tta_enabled', False)),
            )
        elif hasattr(s, 'set_confidence_save_mode_value'):
            s.set_confidence_save_mode_value(str(getattr(state, 'confidence_save_mode', 'off')))
        s.log_update_frequency_spinbox.setValue(state.log_update_frequency)
        s.optimizer_type.setCurrentText(state.optimizer_name)
        s.mixed_precision_type.setCurrentText(state.mixed_precision)
        s.deep_supervision_check_box.setChecked(bool(getattr(state, 'deep_supervision', False)))
        if hasattr(s, 'set_loss_term_weights'):
            s.set_loss_term_weights(
                resolve_loss_term_weights(
                    getattr(state, 'loss_term_weights', None),
                    fallback_loss_function=state.loss_function,
                )
            )
        if hasattr(s, '_sync_loss_controls'):
            s._sync_loss_controls()
        s.learning_rate_spinbox.setValue(state.learning_rate)
        s.weight_decay_spinbox.setValue(state.weight_decay)
        s.early_stopping_check_box.setChecked(state.early_stopping_enabled)
        s.early_stopping_patience_spinbox.setValue(state.early_stopping_patience)
        s.early_stopping_min_delta_spinbox.setValue(state.early_stopping_min_delta)
        s.restore_best_weights_check_box.setChecked(state.early_stopping_restore_best_weights)
        s.warmup_check_box.setChecked(state.warmup_enabled)
        s.warmup_epochs_spinbox.setValue(state.warmup_epochs)
        s.warmup_start_factor_spinbox.setValue(state.warmup_start_factor)
        s.set_scheduler_value(str(getattr(state, 'scheduler_name', 'off')))
        s.scheduler_plateau_factor_spinbox.setValue(float(getattr(state, 'scheduler_plateau_factor', 0.5)))
        s.scheduler_plateau_patience_spinbox.setValue(int(getattr(state, 'scheduler_plateau_patience', 3)))
        s.scheduler_plateau_threshold_spinbox.setValue(float(getattr(state, 'scheduler_plateau_threshold', 1e-4)))
        s.scheduler_plateau_min_lr_spinbox.setValue(float(getattr(state, 'scheduler_plateau_min_lr', 1e-6)))
        s.scheduler_plateau_cooldown_spinbox.setValue(int(getattr(state, 'scheduler_plateau_cooldown', 0)))
        s.scheduler_cosine_t_max_spinbox.setValue(int(getattr(state, 'scheduler_cosine_t_max', 10)))
        s.scheduler_cosine_eta_min_spinbox.setValue(float(getattr(state, 'scheduler_cosine_eta_min', 1e-6)))
        s.scheduler_one_cycle_max_lr_spinbox.setValue(float(getattr(state, 'scheduler_one_cycle_max_lr', 1e-3)))
        s.scheduler_one_cycle_pct_start_spinbox.setValue(
            float(getattr(state, 'scheduler_one_cycle_pct_start', 0.3))
        )
        s.set_scheduler_one_cycle_anneal_strategy_value(
            str(getattr(state, 'scheduler_one_cycle_anneal_strategy', 'cos'))
        )
        s.scheduler_one_cycle_div_factor_spinbox.setValue(
            float(getattr(state, 'scheduler_one_cycle_div_factor', 25.0))
        )
        s.scheduler_one_cycle_final_div_factor_spinbox.setValue(
            float(getattr(state, 'scheduler_one_cycle_final_div_factor', 10000.0))
        )
        s.scheduler_one_cycle_three_phase_check_box.setChecked(
            bool(getattr(state, 'scheduler_one_cycle_three_phase', False))
        )
        s.scheduler_step_lr_step_size_spinbox.setValue(int(getattr(state, 'scheduler_step_lr_step_size', 10)))
        s.scheduler_step_lr_gamma_spinbox.setValue(float(getattr(state, 'scheduler_step_lr_gamma', 0.1)))
        s.hard_mining_check_box.setChecked(state.hard_mining_enabled)
        s.hard_mining_strength_spinbox.setValue(state.hard_mining_strength)
        s.hard_mining_ema_alpha_spinbox.setValue(state.hard_mining_ema_alpha)
        s.hard_pixel_mining_check_box.setChecked(bool(getattr(state, 'hard_pixel_mining_enabled', False)))
        s.hard_pixel_mining_ratio_spinbox.setValue(float(getattr(state, 'hard_pixel_mining_ratio', 0.25)))
        s.skip_uniform_labels_check_box.setChecked(state.skip_uniform_labels)
        s.rare_patch_oversampling_check_box.setChecked(
            bool(getattr(state, 'rare_patch_oversampling_enabled', False))
        )
        s.rare_patch_oversampling_factor_spinbox.setValue(
            int(getattr(state, 'rare_patch_oversampling_factor', 2))
        )
        multi_gpu_mode = normalize_multi_gpu_mode(
            getattr(state, 'multi_gpu_mode', ''),
            use_multi_gpu_fallback=bool(getattr(state, 'use_multi_gpu', False)),
        )
        s.sync_patch_sizes_check_box.setChecked(bool(getattr(state, 'sync_patch_sizes', True)))
        if hasattr(self.view, 'set_recursive_file_search'):
            self.view.set_recursive_file_search(bool(getattr(state, 'recursive_file_search', False)))
        elif hasattr(s, 'recursive_file_search_check_box'):
            s.recursive_file_search_check_box.setChecked(bool(getattr(state, 'recursive_file_search', False)))
        s.multi_gpu_mode_combo.setCurrentText(multi_gpu_mode)
        s.torch_compile_check_box.setChecked(state.torch_compile_enabled)
        view = self.__dict__.get('view')
        if view is not None and hasattr(view, 'set_batch_preview_enabled'):
            view.set_batch_preview_enabled(state.show_batch_preview)

        # 3.7  Additional processing flags
        s.enable_crop_processing.setChecked(state.crop_enabled)
        s.enable_resize_processing.setChecked(state.resize_enabled)
        if hasattr(s, '_sync_preprocess_controls'):
            s._sync_preprocess_controls()
        if hasattr(s, '_sync_patch_size_controls'):
            s._sync_patch_size_controls()
        if hasattr(s, '_sync_rare_patch_oversampling_controls'):
            s._sync_rare_patch_oversampling_controls()
        if hasattr(s, '_sync_recognition_output_controls'):
            s._sync_recognition_output_controls()
        if hasattr(s, 'sync_business_logic_controls'):
            main_state = self.__dict__.get('main_window_state')
            work_mode = getattr(main_state, 'work_mode', '')
            s.sync_business_logic_controls(work_mode)

        # 3.8 Размер обрезки по краям
        s.cut_corner_spinbox.setValue(state.edge_cut_size)

        # 3.9 Compression factor
        s.compression_factor_spinbox.setValue(max(1, int(getattr(state, 'compression_factor', 1))))

    def _update_work_mode(self) -> str:
        return state_sync.update_work_mode(self)

    def _update_settings_window_state(self):
        return state_sync.update_settings_window_state(self)
        s = self.settings_panel

        h = s.horizontal_rotation.isChecked()
        v = s.vertical_rotation.isChecked()
        flip_x = s.flip_x.isChecked()
        flip_y = s.flip_y.isChecked()
        additional_augmentation = s.additional_augmentation_check_box.isChecked()
        augmentation_brightness_strength = s.augmentation_brightness_spinbox.value()
        augmentation_contrast_strength = s.augmentation_contrast_spinbox.value()
        augmentation_gamma_strength = s.augmentation_gamma_spinbox.value()
        augmentation_noise_probability = s.augmentation_noise_probability_spinbox.value()
        augmentation_noise_sigma = s.augmentation_noise_sigma_spinbox.value()
        augmentation_blur_probability = s.augmentation_blur_probability_spinbox.value()
        augmentation_blur_radius = s.augmentation_blur_radius_spinbox.value()
        step = s.shift_spinbox.value()
        train_patch_size = (s.train_patch_x_size.value(), s.train_patch_y_size.value())
        recognition_patch_size = (s.recognition_patch_x_size.value(), s.recognition_patch_y_size.value())
        model = s.get_selected_model()
        color_mode = s.get_color_mode_value() if hasattr(s, 'get_color_mode_value') else s.color_type.currentText()
        shuffle_frames = s.shuffle_frames_check_box.isChecked()
        shuffle_patches_in_frame = s.shuffle_patches_in_frame_check_box.isChecked()
        random_crop = s.random_crop_check_box.isChecked()
        crops_per_image = s.crops_per_image_spinbox.value()
        scale_augmentation = s.scale_augmentation_check_box.isChecked()
        scale_augmentation_strength = s.scale_augmentation_strength_spinbox.value()
        synthetic_defect_generator = (
            s.get_synthetic_defect_generator_config()
            if hasattr(s, 'get_synthetic_defect_generator_config')
            else {}
        )
        cutout_enabled = s.cutout_check_box.isChecked()
        cutout_probability = s.cutout_probability_spinbox.value()
        cutout_holes = s.cutout_holes_spinbox.value()
        cutout_size_ratio = s.cutout_size_ratio_spinbox.value()
        random_artifacts_enabled = s.random_artifacts_check_box.isChecked()
        random_artifacts_probability = s.random_artifacts_probability_spinbox.value()
        random_artifacts_count = s.random_artifacts_count_spinbox.value()
        random_artifacts_size_ratio = s.random_artifacts_size_ratio_spinbox.value()
        random_artifact_type_enabled = {
            artifact_name: checkbox.isChecked()
            for artifact_name, checkbox in getattr(s, 'random_artifact_type_checkboxes', {}).items()
        }
        mixup_enabled = s.mixup_check_box.isChecked()
        mixup_probability = s.mixup_probability_spinbox.value()
        mixup_alpha = s.mixup_alpha_spinbox.value()
        validation = s.validation_check_box.isChecked()
        validation_percent = s.validation_spinbox.value()
        validation_source = normalize_validation_source(s.get_validation_source_value())
        validation_image_folder = s.validation_image_path()
        validation_label_folder = s.validation_label_path()
        save_validation_binary_images = s.save_validation_binary_images_check_box.isChecked()
        cut_mode = self._update_cut_mode()
        train_batch_size = s.train_batch_spinbox.value()
        dataloader_num_workers = s.dataloader_num_workers_spinbox.value()
        recognition_batch_size = s.recognition_batch_spinbox.value()
        sync_patch_sizes = s.sync_patch_sizes_check_box.isChecked()
        if sync_patch_sizes:
            recognition_patch_size = train_patch_size
        overlap = s.overlap_spinbox.value()
        recognition_jpeg_quality = s.recognition_jpeg_quality_spinbox.value()
        recognition_multiprocessing_enabled = s.recognition_multiprocessing_check_box.isChecked()
        recognition_binarize_output = s.recognition_binarize_output_check_box.isChecked()
        recognition_use_auto_threshold = s.recognition_use_auto_threshold_check_box.isChecked()
        recognition_threshold = s.recognition_threshold_spinbox.value()
        recognition_tta_enabled = s.recognition_tta_check_box.isChecked()
        confidence_tta_enabled = (
            s.is_confidence_tta_enabled()
            if hasattr(s, 'is_confidence_tta_enabled')
            else False
        )
        recognition_postprocess = s.recognition_postprocess_check_box.isChecked()
        recognition_postprocess_kernel_size = s.recognition_postprocess_kernel_size_spinbox.value()
        confidence_save_mode = (
            s.get_confidence_save_mode_value()
            if hasattr(s, 'get_confidence_save_mode_value')
            else 'off'
        )
        log_update_frequency = s.log_update_frequency_spinbox.value()
        optimizer_name = s.optimizer_type.currentText()
        mixed_precision = s.mixed_precision_type.currentText()
        deep_supervision = s.deep_supervision_check_box.isChecked()
        current_state = getattr(self, 'settings_state', SettingsState())
        loss_term_weights = (
            s.get_loss_term_weights()
            if hasattr(s, 'get_loss_term_weights')
            else resolve_loss_term_weights({}, fallback_loss_function='bce')
        )
        loss_function = dominant_loss_function(
            loss_term_weights,
            fallback=getattr(current_state, 'loss_function', 'bce'),
        )
        dice_loss_weight = float(getattr(current_state, 'dice_loss_weight', 0.5))
        iou_loss_weight = float(getattr(current_state, 'iou_loss_weight', 0.5))
        learning_rate = s.learning_rate_spinbox.value()
        weight_decay = s.weight_decay_spinbox.value()
        early_stopping_enabled = s.early_stopping_check_box.isChecked()
        early_stopping_patience = s.early_stopping_patience_spinbox.value()
        early_stopping_min_delta = s.early_stopping_min_delta_spinbox.value()
        early_stopping_restore_best_weights = s.restore_best_weights_check_box.isChecked()
        warmup_enabled = s.warmup_check_box.isChecked()
        warmup_epochs = s.warmup_epochs_spinbox.value()
        warmup_start_factor = s.warmup_start_factor_spinbox.value()
        scheduler_name = s.get_scheduler_value()
        scheduler_plateau_factor = s.scheduler_plateau_factor_spinbox.value()
        scheduler_plateau_patience = s.scheduler_plateau_patience_spinbox.value()
        scheduler_plateau_threshold = s.scheduler_plateau_threshold_spinbox.value()
        scheduler_plateau_min_lr = s.scheduler_plateau_min_lr_spinbox.value()
        scheduler_plateau_cooldown = s.scheduler_plateau_cooldown_spinbox.value()
        scheduler_cosine_t_max = s.scheduler_cosine_t_max_spinbox.value()
        scheduler_cosine_eta_min = s.scheduler_cosine_eta_min_spinbox.value()
        scheduler_one_cycle_max_lr = s.scheduler_one_cycle_max_lr_spinbox.value()
        scheduler_one_cycle_pct_start = s.scheduler_one_cycle_pct_start_spinbox.value()
        scheduler_one_cycle_anneal_strategy = s.get_scheduler_one_cycle_anneal_strategy_value()
        scheduler_one_cycle_div_factor = s.scheduler_one_cycle_div_factor_spinbox.value()
        scheduler_one_cycle_final_div_factor = s.scheduler_one_cycle_final_div_factor_spinbox.value()
        scheduler_one_cycle_three_phase = s.scheduler_one_cycle_three_phase_check_box.isChecked()
        scheduler_step_lr_step_size = s.scheduler_step_lr_step_size_spinbox.value()
        scheduler_step_lr_gamma = s.scheduler_step_lr_gamma_spinbox.value()
        hard_mining_enabled = s.hard_mining_check_box.isChecked()
        hard_mining_strength = s.hard_mining_strength_spinbox.value()
        hard_mining_ema_alpha = s.hard_mining_ema_alpha_spinbox.value()
        hard_pixel_mining_enabled = s.hard_pixel_mining_check_box.isChecked()
        hard_pixel_mining_ratio = s.hard_pixel_mining_ratio_spinbox.value()
        skip_uniform_labels = s.skip_uniform_labels_check_box.isChecked()
        rare_patch_oversampling_enabled = s.rare_patch_oversampling_check_box.isChecked()
        rare_patch_oversampling_factor = s.rare_patch_oversampling_factor_spinbox.value()
        multi_gpu_mode = normalize_multi_gpu_mode(s.multi_gpu_mode_combo.currentText())
        use_multi_gpu = multi_gpu_mode != 'off'
        torch_compile_enabled = s.torch_compile_check_box.isChecked()
        show_batch_preview = self.view.is_batch_preview_enabled()
        crop_enabled = s.enable_crop_processing.isChecked()
        resize_enabled = s.enable_resize_processing.isChecked()
        edge_cut_size = s.cut_corner_spinbox.value()
        compression_factor = s.compression_factor_spinbox.value()
        if hasattr(self.view, 'is_recursive_file_search_enabled'):
            recursive_file_search = self.view.is_recursive_file_search_enabled()
        else:
            recursive_file_search = s.recursive_file_search_check_box.isChecked()

        state = SettingsState(step=step, vertical_rotation=v, horizontal_rotation=h,
                              flip_x=flip_x,
                              flip_y=flip_y,
                              additional_augmentation=additional_augmentation,
                              augmentation_brightness_strength=augmentation_brightness_strength,
                              augmentation_contrast_strength=augmentation_contrast_strength,
                              augmentation_gamma_strength=augmentation_gamma_strength,
                              augmentation_noise_probability=augmentation_noise_probability,
                              augmentation_noise_sigma=augmentation_noise_sigma,
                              augmentation_blur_probability=augmentation_blur_probability,
                              augmentation_blur_radius=augmentation_blur_radius,
                              sample_size=train_patch_size,
                              train_patch_size=train_patch_size,
                              recognition_patch_size=recognition_patch_size,
                              model=model,
                              color_mode=color_mode,
                              shuffle=shuffle_frames,
                              shuffle_patches_in_frame=shuffle_patches_in_frame,
                              random_crop=random_crop,
                              crops_per_image=crops_per_image,
                              scale_augmentation=scale_augmentation,
                              scale_augmentation_strength=scale_augmentation_strength,
                              synthetic_defect_generator=synthetic_defect_generator,
                              cutout_enabled=cutout_enabled,
                              cutout_probability=cutout_probability,
                              cutout_holes=cutout_holes,
                              cutout_size_ratio=cutout_size_ratio,
                              random_artifacts_enabled=random_artifacts_enabled,
                              random_artifacts_probability=random_artifacts_probability,
                              random_artifacts_count=random_artifacts_count,
                              random_artifacts_size_ratio=random_artifacts_size_ratio,
                              random_artifacts_dust_enabled=bool(random_artifact_type_enabled.get('dust', True)),
                              random_artifacts_resist_residue_enabled=bool(
                                  random_artifact_type_enabled.get('resist_residue', True)
                              ),
                              random_artifacts_etch_residue_enabled=bool(
                                  random_artifact_type_enabled.get('etch_residue', True)
                              ),
                              random_artifacts_particle_cluster_enabled=bool(
                                  random_artifact_type_enabled.get('particle_cluster', True)
                              ),
                              random_artifacts_flake_enabled=bool(
                                  random_artifact_type_enabled.get('flake', True)
                              ),
                              mixup_enabled=mixup_enabled,
                              mixup_probability=mixup_probability,
                              mixup_alpha=mixup_alpha,
                              use_validation=validation,
                              validation_percent=validation_percent,
                              validation_source=validation_source,
                              validation_image_folder=validation_image_folder,
                              validation_label_folder=validation_label_folder,
                              save_validation_binary_images=save_validation_binary_images,
                              sample_cut_mode=cut_mode,
                              batch_size=train_batch_size,
                              dataloader_num_workers=dataloader_num_workers,
                              train_batch_size=train_batch_size,
                              recognition_batch_size=recognition_batch_size,
                              sync_patch_sizes=sync_patch_sizes,
                              patch_batch_sync_mode='patch' if sync_patch_sizes else 'off',
                              overlap=overlap,
                              recognition_jpeg_quality=recognition_jpeg_quality,
                              recognition_multiprocessing_enabled=recognition_multiprocessing_enabled,
                              recognition_binarize_output=recognition_binarize_output,
                              recognition_use_auto_threshold=recognition_use_auto_threshold,
                              recognition_threshold=recognition_threshold,
                              recognition_tta_enabled=recognition_tta_enabled,
                              confidence_tta_enabled=confidence_tta_enabled,
                              recognition_postprocess=recognition_postprocess,
                              recognition_postprocess_kernel_size=recognition_postprocess_kernel_size,
                              confidence_save_mode=confidence_save_mode,
                              crop_enabled=crop_enabled, resize_enabled=resize_enabled, edge_cut_size=edge_cut_size,
                              compression_factor=compression_factor,
                              recursive_file_search=recursive_file_search,
                              optimizer_name=optimizer_name,
                              mixed_precision=mixed_precision,
                              deep_supervision=deep_supervision,
                              loss_function=loss_function,
                              loss_term_weights=loss_term_weights,
                              dice_loss_weight=dice_loss_weight,
                              iou_loss_weight=iou_loss_weight,
                              learning_rate=learning_rate,
                              weight_decay=weight_decay,
                              early_stopping_enabled=early_stopping_enabled,
                              early_stopping_patience=early_stopping_patience,
                              early_stopping_min_delta=early_stopping_min_delta,
                              early_stopping_restore_best_weights=early_stopping_restore_best_weights,
                              warmup_enabled=warmup_enabled,
                              warmup_epochs=warmup_epochs,
                              warmup_start_factor=warmup_start_factor,
                              scheduler_name=scheduler_name,
                              scheduler_plateau_factor=scheduler_plateau_factor,
                              scheduler_plateau_patience=scheduler_plateau_patience,
                              scheduler_plateau_threshold=scheduler_plateau_threshold,
                              scheduler_plateau_min_lr=scheduler_plateau_min_lr,
                              scheduler_plateau_cooldown=scheduler_plateau_cooldown,
                              scheduler_cosine_t_max=scheduler_cosine_t_max,
                              scheduler_cosine_eta_min=scheduler_cosine_eta_min,
                              scheduler_one_cycle_max_lr=scheduler_one_cycle_max_lr,
                              scheduler_one_cycle_pct_start=scheduler_one_cycle_pct_start,
                              scheduler_one_cycle_anneal_strategy=scheduler_one_cycle_anneal_strategy,
                              scheduler_one_cycle_div_factor=scheduler_one_cycle_div_factor,
                              scheduler_one_cycle_final_div_factor=scheduler_one_cycle_final_div_factor,
                              scheduler_one_cycle_three_phase=scheduler_one_cycle_three_phase,
                              scheduler_step_lr_step_size=scheduler_step_lr_step_size,
                              scheduler_step_lr_gamma=scheduler_step_lr_gamma,
                              hard_mining_enabled=hard_mining_enabled,
                              hard_mining_strength=hard_mining_strength,
                              hard_mining_ema_alpha=hard_mining_ema_alpha,
                              hard_pixel_mining_enabled=hard_pixel_mining_enabled,
                              hard_pixel_mining_ratio=hard_pixel_mining_ratio,
                              log_update_frequency=log_update_frequency,
                              skip_uniform_labels=skip_uniform_labels,
                              rare_patch_oversampling_enabled=rare_patch_oversampling_enabled,
                              rare_patch_oversampling_factor=rare_patch_oversampling_factor,
                              use_multi_gpu=use_multi_gpu,
                              multi_gpu_mode=multi_gpu_mode,
                              torch_compile_enabled=torch_compile_enabled,
                              show_batch_preview=show_batch_preview)

        self.settings_state = state

    def _on_batch_preview_visibility_changed(self, enabled: bool):
        self.settings_state.show_batch_preview = bool(enabled)

    def _open_augmentation_preview(self):
        self._update_main_window_state()
        self._update_settings_window_state()
        _work_mode, training_parameters, _recognition_parameters = build_workflow_parameters(
            self.main_window_state,
            self.settings_state,
        )
        training_parameters = replace(
            training_parameters,
            generation=replace(
                training_parameters.generation,
                tech_aug=build_tech_augmentation_config(None),
            ),
        )
        if training_parameters.label_path.is_dir() and filter_files(training_parameters.label_path, ('.cif',)):
            from neuralimage.lib.rare_patch_masks import prepare_label_folder_for_rare_patch_editor

            message_bus = self.__dict__.get('message_bus')
            try:
                resolved_label_folder, error_message = prepare_label_folder_for_rare_patch_editor(
                    training_parameters.label_path,
                    log_callback=(
                        (lambda message: message_bus.publish('logging', message))
                        if message_bus is not None
                        else None
                    ),
                )
            except Exception as exc:
                self.view.show_warning.emit(
                    f'Не удалось подготовить CIF-маски для предпросмотра аугментаций: {exc}'
                )
                return
            if error_message is not None:
                self.view.show_warning.emit(str(error_message))
                return
            training_parameters = replace(training_parameters, label_path=resolved_label_folder)

        from neuralimage.view.augmentation_preview_dialog import AugmentationPreviewDialog

        existing_dialog = getattr(self, '_augmentation_preview_dialog', None)
        if existing_dialog is not None:
            try:
                existing_dialog.close()
            finally:
                self._augmentation_preview_dialog = None

        dialog = AugmentationPreviewDialog(training_parameters, self.view)
        dialog.apply_to_main_requested.connect(self._apply_augmentation_preview_settings)
        dialog.destroyed.connect(lambda *_args: setattr(self, '_augmentation_preview_dialog', None))
        self._augmentation_preview_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

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
        panel.crops_per_image_spinbox.setValue(int(payload.get('crops_per_image', 64)))
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
        if hasattr(panel, 'set_synthetic_defect_generator_config'):
            panel.set_synthetic_defect_generator_config(payload.get('synthetic_defect_generator', {}))

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

    def _open_rare_patch_editor(self):
        self._update_main_window_state()
        sample_folder = Path(self.main_window_state.sample_folder)
        label_folder = Path(self.main_window_state.label_folder)
        if not sample_folder.is_dir() or not label_folder.is_dir():
            self.view.show_warning.emit(
                'Выберите существующие папки с изображениями и метками перед открытием редактора редких областей.'
            )
            return

        if self._rare_patch_editor_prepare_thread is not None:
            self.message_bus.publish(
                'logging',
                'Подготовка меток для редактора редких областей уже выполняется.',
            )
            return

        if filter_files(label_folder, ('.cif',)):
            self._start_rare_patch_editor_cif_conversion(sample_folder, label_folder)
            return

        self._open_rare_patch_editor_dialog(sample_folder, label_folder)

    def _start_rare_patch_editor_cif_conversion(self, sample_folder: Path, label_folder: Path) -> None:
        self.message_bus.publish(
            'logging',
            'Обнаружены CIF-метки. Готовлю binary_cif для редактора редких областей.',
        )
        self._set_rare_patch_editor_preparing(True)
        thread = PresenterRarePatchEditorPreparationThread(
            sample_folder=sample_folder,
            label_folder=label_folder,
            message_bus=self.message_bus,
        )
        thread.prepared.connect(self._on_rare_patch_editor_prepared)
        thread.finished.connect(thread.deleteLater)
        self._rare_patch_editor_prepare_thread = thread
        thread.start()

    def _set_rare_patch_editor_preparing(self, active: bool) -> None:
        button = getattr(self.settings_panel, 'edit_rare_regions_button', None)
        if button is not None:
            button.setEnabled(not active)

    @QtCore.pyqtSlot(str, str, str)
    def _on_rare_patch_editor_prepared(
        self,
        sample_folder: str,
        resolved_label_folder: str,
        error_message: str,
    ) -> None:
        self._set_rare_patch_editor_preparing(False)
        self._rare_patch_editor_prepare_thread = None

        if error_message:
            self.view.show_warning.emit(error_message)
            return

        self.view.set_label_path(resolved_label_folder)
        self.main_window_state.label_folder = resolved_label_folder
        self._open_rare_patch_editor_dialog(Path(sample_folder), Path(resolved_label_folder))

    def _open_rare_patch_editor_dialog(self, sample_folder: Path, label_folder: Path) -> None:
        from neuralimage.lib.rare_patch_masks import collect_matching_sample_label_pairs
        from neuralimage.view.rare_patch_editor_dialog import RarePatchEditorDialog

        _pairs, error_message = collect_matching_sample_label_pairs(sample_folder, label_folder)
        if error_message is not None:
            self.view.show_warning.emit(error_message)
            return

        dialog = RarePatchEditorDialog(sample_folder, label_folder, self.view)
        dialog.exec()

    def _log_message_emit(self, data):
        if not should_forward_log_event('logging', data):
            return
        self.view.log_message.emit(data)

    def _train_message_emit(self, data):
        if not should_forward_log_event('training', data):
            return
        self.view.log_message_with_delete_last.emit(data)

    def _metrics_message_emit(self, data):
        self.view.metrics_message.emit(data)

    def _error_message_emit(self, data):
        message = str(data) if data is not None else 'Произошла ошибка выполнения.'
        self.view.show_warning.emit(message)
        self.view.log_message.emit(f'Ошибка: {message}')

    def _update_cut_mode(self) -> str:
        return state_sync.update_cut_mode(self)

    # ------------------------------------------------------------------ #
    #   Сохранение окон в QSettings
    # ------------------------------------------------------------------ #
    def _save_windows_to_qsettings(self):
        self._update_main_window_state()
        self._update_settings_window_state()

        self._save_main_window_to_qsettings()
        self._save_nn_settings_to_qsettings()

    def _save_main_window_to_qsettings(self):
        self._state_store.save_main_window_state(self.main_window_state)

    def _save_nn_settings_to_qsettings(self):
        state = getattr(self, "settings_state", None) or SettingsState()
        self.settings_state = state
        self._state_store.save_settings_state(state)

    def _reset_settings_to_defaults(self):
        state_sync.reset_settings_to_defaults(self)

    def _choose_source_folder(self):
        path = _tk_filedialog('folder')
        if path:
            self.view.set_source_path(path)
            self.main_window_state.source_folder = path
            self._validate_start_button()

    def _choose_jpg_label_path(self):
        path = _tk_filedialog('folder')
        if path:
            self.view.set_jpg_path(path)
            self.main_window_state.sample_folder = path
            self.sample_calculator.set_path(Path(path))
            self._calculate_expected_samples()
            self._validate_start_button()

    def _chosse_cif_label_path(self):
        path = _tk_filedialog('folder')
        if path:
            self.view.set_label_path(path)
            self.main_window_state.label_folder = path

            self._validate_start_button()

    def _choose_validation_image_folder(self):
        path = _tk_filedialog('folder')
        if path:
            self.settings_panel.set_validation_image_path(path)
            self.settings_state.validation_image_folder = path
            self._validate_start_button()

    def _choose_validation_label_folder(self):
        path = _tk_filedialog('folder')
        if path:
            self.settings_panel.set_validation_label_path(path)
            self.settings_state.validation_label_folder = path
            self._validate_start_button()

    def _choose_result_folder(self):
        path = _tk_filedialog('folder')
        if path:
            self.view.set_result_path(path)
            self.main_window_state.result_folder = path
            self._validate_start_button()

    def _choose_model_path(self):
        path = _tk_filedialog('file', [("Модель", ".pth")])
        if path:
            self.view.model_path.setText(path)
            self._validate_start_button()

    def _on_open_config_requested(self):
        path = _tk_filedialog('file', [('JSON', '.json')])
        if not path:
            return
        try:
            main_state, settings_state = load_workflow_snapshot(path)
        except (OSError, ValueError) as error:
            self.view.show_warning.emit(f'Не удалось загрузить параметры из файла: {error}')
            return

        self._restore_task_state_to_ui(
            main_state,
            settings_state,
            log_message='Параметры восстановлены из файла.',
        )
        self._save_windows_to_qsettings()

    def _on_sample_type_changed(self, typ: str):
        """typ = ''train_and_recognition'', 'recognition_only', further_training."""
        mode = normalize_work_mode(typ)
        previous_mode = normalize_work_mode(getattr(self.main_window_state, 'work_mode', ''))
        if previous_mode:
            self._store_view_state_for_mode(previous_mode)
        self.main_window_state.work_mode = mode
        self._restore_mode_state_to_view(mode)
        self.view.apply_work_mode(mode)
        if hasattr(self.settings_panel, 'sync_business_logic_controls'):
            self.settings_panel.sync_business_logic_controls(mode)
        self._calculate_expected_samples()
        self._validate_start_button()

    def _on_start_requested(self):
        task_flow.on_start_requested(self)

    def _on_stop_requested(self):
        task_flow.on_stop_requested(self)

    def _on_release_memory_requested(self):
        gc.collect()
        self.message_bus.publish('logging', 'Выполнена очистка памяти Python.')

    def _on_ui_language_selected(self, language: str):
        self.settings_panel.set_ui_language(language)

    def _on_theme_selected(self, theme: str):
        self.view.apply_theme(theme)

    def _on_ui_mode_selected(self, mode: str):
        normalized_mode = 'advanced' if mode == 'advanced' else 'simple'
        self.view.apply_ui_mode(normalized_mode)
        self.main_window_state.ui_mode = normalized_mode
        self._save_main_window_to_qsettings()

    def _refresh_update_client_config(self) -> None:
        update_flow.refresh_update_client_config(
            self,
            load_update_client_config_fn=load_update_client_config,
            load_selected_update_channel_fn=load_selected_update_channel,
        )

    def _on_update_channel_selected(self, channel: str) -> None:
        update_flow.on_update_channel_selected(
            self,
            channel,
            save_selected_update_channel_fn=save_selected_update_channel,
            load_update_client_config_fn=load_update_client_config,
            load_selected_update_channel_fn=load_selected_update_channel,
        )

    def _on_simple_workflow_requested(self, preset_name: str):
        preset_path = self.SIMPLE_WORKFLOW_PRESETS.get(str(preset_name))
        if preset_path is None:
            self.view.show_warning.emit('Неизвестный простой профиль.')
            return
        try:
            main_state, settings_state = load_workflow_snapshot(preset_path)
        except (OSError, ValueError) as error:
            self.view.show_warning.emit(f'Не удалось загрузить профиль: {error}')
            return

        restored_main_state = replace(
            main_state,
            work_mode=self.main_window_state.work_mode,
            source_folder=self.main_window_state.source_folder,
            result_folder=self.main_window_state.result_folder,
            label_folder=self.main_window_state.label_folder,
            sample_folder=self.main_window_state.sample_folder,
            model_path=self.main_window_state.model_path,
            epochs=self.main_window_state.epochs,
            ui_mode='simple',
            mode_state=normalize_main_window_mode_state(getattr(self.main_window_state, 'mode_state', {})),
        )
        self._restore_task_state_to_ui(
            restored_main_state,
            settings_state,
            log_message=f'Загружен простой профиль: {preset_path.name}.',
        )
        self.view.set_simple_workflow_profile(str(preset_name))
        self._save_windows_to_qsettings()

    def _on_update_check_requested(self) -> None:
        self._start_update_check(manual=True)

    def _on_queue_remove_requested(self):
        row = self.view.get_selected_queue_row()
        self._remove_queue_row(row)

    def _on_queue_context_remove_requested(self, row: int):
        self._remove_queue_row(row)

    def _remove_queue_row(self, row: int):
        task_flow.remove_queue_row(self, row)

    def _on_queue_properties_requested(self, row: int):
        task_flow.on_queue_properties_requested(self, row, dialog_cls=TaskPropertiesDialog)

    def _on_task_restore_requested(self, main_state: MainWindowState, settings_state: SettingsState):
        task_flow.on_task_restore_requested(self, main_state, settings_state)

    def _restore_task_state_to_ui(
        self,
        main_state: MainWindowState,
        settings_state: SettingsState,
        *,
        log_message: str = 'Параметры задачи восстановлены в интерфейсе.',
    ):
        state_sync.restore_task_state_to_ui(
            self,
            main_state,
            settings_state,
            log_message=log_message,
        )

    def _on_queue_pause_toggle_requested(self):
        task_flow.on_queue_pause_toggle_requested(self)

    def _refresh_queue_view(self, selected_row: int = -1, selected_task_id: int | None = None):
        task_flow.refresh_queue_view(
            self,
            selected_row=selected_row,
            selected_task_id=selected_task_id,
        )

    def _start_next_task_if_possible(self):
        task_flow.start_next_task_if_possible(self)

    def _start_task(self, task: QueuedTask):
        task_flow.start_task(
            self,
            task,
            handler_thread_cls=GeneralNeuralHandlerThread,
            workflow_builder=build_workflow_parameters,
            artifact_dir_builder=build_training_artifact_dir,
            workflow_snapshot_saver=save_workflow_snapshot,
            workflow_snapshot_filename=WORKFLOW_SNAPSHOT_FILENAME,
        )

    def _on_task_finished(self):
        task_flow.on_task_finished(self)
    def _clear_validation_gradient_window_refs(self) -> None:
        plugin_flow.clear_validation_gradient_window_refs(self)

    def _on_open_validation_gradient_requested(self) -> None:
        plugin_flow.open_validation_gradient_requested(
            self,
            window_cls=plugin_flow.ValidationGradientPluginWindow,
        )

    def _shutdown_validation_gradient_plugin(self) -> None:
        plugin_flow.shutdown_validation_gradient_plugin(self)

    def _on_developer_tools_requested(self) -> None:
        if self._developer_tools_dialog is None:
            self._developer_tools_dialog = DeveloperToolsDialog(self.view)
        self._developer_tools_dialog.show()
        self._developer_tools_dialog.raise_()
        self._developer_tools_dialog.activateWindow()

    def _shutdown_developer_tools(self) -> None:
        if self._developer_tools_dialog is not None:
            self._developer_tools_dialog.shutdown()

    def _on_close_requested(self):
        ui_texts = get_ui_section('main_window')
        reply = QMessageBox.question(
            self.view,
            str(ui_texts.get('exit_title', 'Выход')),
            str(ui_texts.get('exit_text', 'Закрыть программу?')),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._save_windows_to_qsettings()
            self._shutdown_validation_gradient_plugin()
            self._shutdown_developer_tools()
            self.view.allow_close()

    # ------------------------------------------------------------------ #
    #   Обработчик UI событий панели
    # ------------------------------------------------------------------ #

    def _calculate_expected_samples(self):
        sample_counting.calculate_expected_samples(self)

    def get_cut_settings_from_window_state(self) -> CutSettings:
        return sample_counting.get_cut_settings_from_window_state(self)

    def _set_sample_number(self, calculator_settings: CutSettings) -> None:
        sample_counting.set_sample_number(self, calculator_settings)

    def _invalidate_sample_count_requests(self) -> None:
        sample_counting.invalidate_sample_count_requests(self)

    def _dispatch_sample_count_request(self) -> None:
        sample_counting.dispatch_sample_count_request(self)

    def _start_sample_count_request(self, request: tuple[int, str, CutSettings, object]) -> None:
        sample_counting.start_sample_count_request(
            self,
            request,
            sample_worker_cls=SampleWorker,
            threading_module=threading,
        )

    def _run_sample_count_request(
        self,
        request_id: int,
        sample_folder: str,
        calculator_settings: CutSettings,
        synthetic_config: object,
        cached_path: str | None,
        cached_sizes: list[tuple[int, int]] | None,
    ) -> None:
        sample_counting.run_sample_count_request(
            self,
            request_id,
            sample_folder,
            calculator_settings,
            synthetic_config,
            cached_path,
            cached_sizes,
            sample_worker_cls=SampleWorker,
        )

    @staticmethod
    def _normalize_sample_count_path(path: Path | str) -> str:
        return sample_counting.normalize_sample_count_path(path)

    @QtCore.pyqtSlot(int, str, object, int)
    def _on_sample_count_calculated(
        self,
        request_id: int,
        normalized_path: str,
        image_sizes: object,
        total_samples: int,
    ) -> None:
        sample_counting.on_sample_count_calculated(
            self,
            request_id,
            normalized_path,
            image_sizes,
            total_samples,
        )

    @QtCore.pyqtSlot(int, str)
    def _on_sample_count_failed(self, request_id: int, error_message: str) -> None:
        sample_counting.on_sample_count_failed(self, request_id, error_message)

    def _start_pending_sample_count_request_if_needed(self) -> None:
        sample_counting.start_pending_sample_count_request_if_needed(self)

    def _set_max_shift(self):
        sample_counting.set_max_shift(self)

    # ------------------------------------------------------------------ #
    #   Валидация и управление кнопкой "Начать"
    # ------------------------------------------------------------------ #
    def _validate_start_button(self):
        """Проверяем, что все обязательные поля заполнены."""
        self._update_main_window_state()
        self._update_settings_window_state()
        self.view.enable_start.emit(can_start_processing(self.main_window_state, self.settings_state))

    # ------------------------------------------------------------------ #
    #   Управление началом потоков
    # ------------------------------------------------------------------ #
    def _thread_ask(
        self,
        question: str,
        header: str = 'Обратите внимание',
        default_answer: bool = False,
        timeout_seconds: int = 0,
    ):
        answer = presenter_dialogs.ask_yes_no_with_timeout(
            parent=self.view,
            question=question,
            header=header,
            default_answer=default_answer,
            timeout_seconds=timeout_seconds,
        )
        handler = self.neuaral_handler
        if handler is not None:
            handler.answer.emit(answer)

    def _start_update_check(self, *, manual: bool = False) -> None:
        update_flow.start_update_check(
            self,
            manual=manual,
            load_update_manifest_url_fn=load_update_manifest_url,
            update_check_thread_cls=PresenterAppUpdateCheckThread,
            load_update_client_config_fn=load_update_client_config,
            load_selected_update_channel_fn=load_selected_update_channel,
        )

    def _clear_update_check_thread(self) -> None:
        update_flow.clear_update_check_thread(self)

    def _on_update_check_finished(self, update_info: object) -> None:
        update_flow.on_update_check_finished(
            self,
            update_info,
            update_info_cls=UpdateInfo,
            app_version=APP_VERSION,
            is_newer_version_fn=is_newer_version,
            load_last_notified_version_fn=load_last_notified_version,
            should_notify_version_fn=should_notify_version,
            save_last_notified_version_fn=save_last_notified_version,
        )

    @staticmethod
    def _markdown_to_message_html(markdown: str) -> str:
        return presenter_dialogs.markdown_to_message_html(markdown)

    def _show_update_notification(self, update_info: UpdateInfo, *, manual: bool = False) -> None:
        update_flow.show_update_notification(
            self,
            update_info,
            manual=manual,
            app_version=APP_VERSION,
            collect_release_history_fn=collect_release_history,
        )

    def _show_release_selector(self, update_info: UpdateInfo) -> None:
        update_flow.show_release_selector(self, update_info, app_version=APP_VERSION)

    def _confirm_release_install(self, release: ReleaseInfo) -> None:
        update_flow.confirm_release_install(self, release, app_version=APP_VERSION)

    @staticmethod
    def _resolve_latest_release(update_info: UpdateInfo) -> ReleaseInfo:
        for release in update_info.releases:
            if release.version == update_info.version:
                return release
        return ReleaseInfo(
            version=update_info.version,
            download_url=update_info.download_url,
            notes=update_info.release_notes,
        )

    def _start_update_download(self, release: ReleaseInfo) -> None:
        update_flow.start_update_download(
            self,
            release,
            update_download_thread_cls=PresenterAppUpdateDownloadThread,
        )

    def _clear_update_download_thread(self) -> None:
        update_flow.clear_update_download_thread(self)

    def _on_update_download_finished(self, installer_path: str, version: str) -> None:
        update_flow.on_update_download_finished(
            self,
            installer_path,
            version,
            launch_update_installer_fn=launch_update_installer,
        )

    def _on_update_download_failed(self, error_message: str) -> None:
        update_flow.on_update_download_failed(self, error_message)


class RarePatchEditorPreparationThread(QThread):
    prepared = QtCore.pyqtSignal(str, str, str)

    def __init__(
        self,
        *,
        sample_folder: Path,
        label_folder: Path,
        message_bus: AbstractMessageBus,
    ) -> None:
        super().__init__()
        self._sample_folder = Path(sample_folder)
        self._label_folder = Path(label_folder)
        self._message_bus = message_bus

    def run(self) -> None:
        from neuralimage.lib.rare_patch_masks import (
            collect_matching_sample_label_pairs,
            prepare_label_folder_for_rare_patch_editor,
        )

        try:
            resolved_label_folder, error_message = prepare_label_folder_for_rare_patch_editor(
                self._label_folder,
                log_callback=lambda message: self._message_bus.publish('logging', message),
            )
            if error_message is None:
                _pairs, error_message = collect_matching_sample_label_pairs(
                    self._sample_folder,
                    resolved_label_folder,
                )
        except Exception as exc:
            resolved_label_folder = self._label_folder
            error_message = f'Не удалось подготовить метки для редактора редких областей: {exc}'

        self.prepared.emit(
            str(self._sample_folder),
            str(resolved_label_folder),
            '' if error_message is None else str(error_message),
        )


class GeneralNeuralHandlerThread(QThread):
    ask = QtCore.pyqtSignal(str, str, bool, int)  # text, title, default answer, timeout
    answer = QtCore.pyqtSignal(bool)

    def __init__(self, work_mode: WorkMode,
                 message_bus: AbstractMessageBus,
                 recognition_parameters: RecognitionParameters | None = None,
                 tranining_parameters: TrainingParameters | None = None,
                 callback: Callable[..., None] | None = None):
        super().__init__()
        self._last_answer = False
        self._waiting_for_answer = False
        self.main_logic = GeneralNeuralHandler(work_mode=work_mode,
                                               recogniton_parameters=recognition_parameters,
                                               tranining_parameters=tranining_parameters,
                                               question_module=self.check,
                                               message_bus=message_bus)
        self.answer.connect(self._store_answer)

    def run(self):
        self.main_logic.start()

    def check(self, text, theme, default_answer: bool = False, timeout_seconds: int | None = None):
        self._last_answer = bool(default_answer)
        self._waiting_for_answer = True
        self.ask.emit(
            text,
            theme,
            bool(default_answer),
            max(0, int(timeout_seconds or 0)),
        )
        loop = QtCore.QEventLoop()
        def _quit_loop(_value: bool) -> None:
            if loop.isRunning():
                loop.quit()
        self.answer.connect(_quit_loop)
        try:
            loop.exec()
            return self._last_answer
        finally:
            self._waiting_for_answer = False
            try:
                self.answer.disconnect(_quit_loop)
            except TypeError:
                pass

    def stop(self):
        if self._waiting_for_answer:
            self.answer.emit(False)
        self.main_logic.stop_execution()

    @QtCore.pyqtSlot(bool)
    def _store_answer(self, val: bool):
        self._last_answer = val


class AppUpdateCheckThread(QThread):
    checked = QtCore.pyqtSignal(object)

    def __init__(self, *, manifest_url: str, channel: str) -> None:
        super().__init__()
        self._manifest_url = str(manifest_url).strip()
        self._channel = str(channel or '').strip().lower()

    def run(self) -> None:
        self.checked.emit(fetch_update_info(self._manifest_url, expected_channel=self._channel))


class AppUpdateDownloadThread(QThread):
    finished_download = QtCore.pyqtSignal(str, str)
    failed_download = QtCore.pyqtSignal(str)

    def __init__(self, *, release: ReleaseInfo) -> None:
        super().__init__()
        self._release = release

    def run(self) -> None:
        try:
            installer_path = download_update_installer(self._release)
        except Exception as exc:
            self.failed_download.emit(str(exc))
            return
        self.finished_download.emit(str(installer_path), self._release.version)

