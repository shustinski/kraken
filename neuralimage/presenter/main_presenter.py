# presenter/main_presenter.py
import multiprocessing as mp
import gc
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from PyQt6 import QtCore, QtWidgets

from lib.message_bus import MessageBus, AbstractMessageBus
from model.NeuralNetwork import get_registered_models
from model.general_neural_handler import GeneralNeuralHandler
from view import MainView, SettingsPanel
from view.window_dataclasses import MainWindowState, SettingsState
from lib.data_interfaces import (
    CutSettings,
    SampleCutMode,
    WorkMode,
    TrainingParameters,
    RecognitionParameters,
    normalize_work_mode,
)

from lib.images import SampleWorker
from presenter.state_store import (
    load_main_window_state,
    load_settings_state,
    save_main_window_state,
    save_settings_state,
)
from presenter.workflow_mapper import build_workflow_parameters
from presenter.validation import can_start_processing
from lib.ui_texts import get_ui_section


def _tk_filedialog(kind: str, filetypes=None) -> str | None:
    """Qt-based file dialog helper."""
    if kind == 'folder':
        path = QFileDialog.getExistingDirectory(None, 'Выберите папку')
    else:  # 'file'
        filter_str = ''
        if filetypes:
            labels = []
            for title, ext in filetypes:
                cleaned = ext if str(ext).startswith('*.') else f'*{ext}'
                labels.append(f'{title} ({cleaned})')
            filter_str = ';;'.join(labels)
        path, _ = QFileDialog.getOpenFileName(None, 'Выберите файл', '', filter_str)
    return path if path else None


@dataclass
class QueuedTask:
    task_id: int
    main_window_state: MainWindowState
    settings_state: SettingsState
    paused: bool = False


class MainPresenter(QObject):
    """
    Связывает View и Model. Содержит всю бизнес-логику
    (валидацию, формирование параметров, запуск/остановку потока).
    """

    def __init__(self):
        super().__init__()

        self.sample_calculator = SampleWorker()

        # Инициализируем панель с настройками нейросети
        self.settings_panel = SettingsPanel()

        # Получаем список активных моделей нейросетей
        self.active_nn_models = get_registered_models()
        self.settings_panel.model_type_init(list(self.active_nn_models.keys()))

        # Инициализируем главное окно
        self.view = MainView(side_panel=self.settings_panel)

        self.stop_event = mp.Event()
        self._queued_tasks: list[QueuedTask] = []
        self._next_task_id = 1
        self._active_task: QueuedTask | None = None
        self._active_stop_requested = False
        self.neuaral_handler: GeneralNeuralHandlerThread | None = None

        # --------------------- 1. Подписка на сигналы View --------------------- #
        self._conncet_to_message_bus()
        self._connect_view_signals()
        self._connect_settings_signal()

        # --------------------- 3. Инициализация UI из Config --------------------- #
        self._load_initial_state()
        self.view.show()



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

        v.source_path_requested.connect(self._choose_source_folder)
        v.result_path_requested.connect(self._choose_result_folder)

        v.jpg_path_requested.connect(self._choose_jpg_label_path)
        v.label_path_requested.connect(self._chosse_cif_label_path)

        v.model_path_requested.connect(self._choose_model_path)

        v.start_requested.connect(self._on_start_requested)
        v.stop_requested.connect(self._on_stop_requested)
        v.queue_remove_requested.connect(self._on_queue_remove_requested)
        v.queue_pause_toggle_requested.connect(self._on_queue_pause_toggle_requested)
        v.batch_preview_visibility_changed.connect(self._on_batch_preview_visibility_changed)
        v.release_memory_requested.connect(self._on_release_memory_requested)

        v.request_close.connect(self._on_close_requested)

    def _connect_settings_signal(self):
        v = self.settings_panel
        v.cut_slider_shifted.connect(self._calculate_expected_samples)
        v.horisontal_rotate_clicked.connect(self._calculate_expected_samples)
        v.vertical_rotate_clicked.connect(self._calculate_expected_samples)
        v.model_changed.connect(self._calculate_expected_samples)
        v.sample_size_changed.connect(self._set_max_shift)
        v.optimizer_settings_changed.connect(self._update_settings_window_state)
        v.validation_settings_changed.connect(self._update_settings_window_state)
        v.reset_defaults_requested.connect(self._reset_settings_to_defaults)

    # ------------------------------------------------------------------ #
    #   Инициализация UI
    # ------------------------------------------------------------------ #
    def _load_initial_state(self):
        # Читаем настройки из Settings и заполняем View
        self._load_main_window_settings()
        self._load_settings_panel_settings()

        # Последняя проверка доступности кнопки "Начать"
        self.view.restore_from_dataclass(self.main_window_state)
        self._apply_settings_to_panel()
        self.view.set_batch_preview_enabled(self.settings_state.show_batch_preview)
        self.settings_panel.set_model(self.settings_state.model)
        self._restore_work_mode_ui()

        self.settings_panel.connect_internal_signals()
        self.view.connect_internal_signals()

        self._refresh_queue_view()
        self._validate_start_button()

    def _load_main_window_settings(self):
        self.main_window_state = load_main_window_state()
        self.sample_calculator.set_path(Path(self.main_window_state.sample_folder))

    def _load_settings_panel_settings(self):
        self.settings_state = load_settings_state()

    # ------------------------------------------------------------------ #
    #   Обновление класса состояния окон
    # ------------------------------------------------------------------ #

    def _update_main_window_state(self):
        v = self.view
        work_mode = self._update_work_mode()

        source_path = v.lbl_source.text()
        result_path = v.lbl_result.text()

        # Радиокнопка: сохраняем строковое значение, а не bool-состояние
        label_path = v.label_path.text()
        sample_path = v.sample_path.text()

        model_path = v.model_path.text()
        epochs = v.le_epochs.value()

        self.main_window_state = MainWindowState(work_mode=work_mode,
                                                 source_folder=source_path,
                                                 result_folder=result_path,
                                                 label_folder=label_path,
                                                 sample_folder=sample_path,
                                                 model_path=model_path,
                                                 epochs=epochs)

    def _restore_work_mode_ui(self):
        mode = normalize_work_mode(getattr(self.main_window_state, 'work_mode', ''))
        v = self.view
        mode_to_button = {
            WorkMode.train_and_recognition.value: v.rb_train_and_recognition,
            WorkMode.train_only.value: v.rb_train_only,
            WorkMode.further_training.value: v.rb_further_train_model,
            WorkMode.recognition_only.value: v.rb_recognition,
        }
        if mode not in mode_to_button:
            mode = WorkMode.train_and_recognition.value
            self.main_window_state.work_mode = mode
        mode_to_button[mode].setChecked(True)
        self._on_sample_type_changed(mode)

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
        s.additional_augmentation_check_box.setChecked(state.additional_augmentation)
        s.augmentation_brightness_spinbox.setValue(state.augmentation_brightness_strength)
        s.augmentation_contrast_spinbox.setValue(state.augmentation_contrast_strength)
        s.augmentation_noise_probability_spinbox.setValue(state.augmentation_noise_probability)
        s.augmentation_noise_sigma_spinbox.setValue(state.augmentation_noise_sigma)
        if hasattr(s, '_sync_augmentation_controls'):
            s._sync_augmentation_controls(state.additional_augmentation)

        # 3.2  Spin boxes (int/float)
        s.shift_spinbox.setValue(state.step)
        s.sample_x_size.setValue(state.sample_size[0])
        s.sample_y_size.setValue(state.sample_size[1])

        # 3.3 Combo-box (str - используем setCurrentText, который выбирает
        #      entry if it exists, otherwise it will add a new entry.)
        s.nn_model_type.setCurrentText(state.model)
        s.color_type.setCurrentText(state.color_mode)

        # 3.4  Validation controls
        s.validation_check_box.setChecked(state.use_validation)
        s.validation_spinbox.setValue(state.validation_percent)

        # 3.5 Режим нарезки

        s.restore_cut_mode(state.sample_cut_mode)

        # 3.6 Размер батча / overlap
        s.batch_spinbox.setValue(state.batch_size)
        s.overlap_spinbox.setValue(state.overlap)
        s.log_update_frequency_spinbox.setValue(state.log_update_frequency)
        s.optimizer_type.setCurrentText(state.optimizer_name)
        s.mixed_precision_type.setCurrentText(state.mixed_precision)
        s.loss_function_type.setCurrentText(state.loss_function)
        s.dice_loss_weight_spinbox.setValue(state.dice_loss_weight)
        s.iou_loss_weight_spinbox.setValue(state.iou_loss_weight)
        if hasattr(s, '_sync_loss_controls'):
            s._sync_loss_controls(s.loss_function_type.currentIndex())
        s.learning_rate_spinbox.setValue(state.learning_rate)
        s.weight_decay_spinbox.setValue(state.weight_decay)
        s.early_stopping_check_box.setChecked(state.early_stopping_enabled)
        s.early_stopping_patience_spinbox.setValue(state.early_stopping_patience)
        s.early_stopping_min_delta_spinbox.setValue(state.early_stopping_min_delta)
        s.restore_best_weights_check_box.setChecked(state.early_stopping_restore_best_weights)
        s.warmup_check_box.setChecked(state.warmup_enabled)
        s.warmup_epochs_spinbox.setValue(state.warmup_epochs)
        s.warmup_start_factor_spinbox.setValue(state.warmup_start_factor)
        s.hard_mining_check_box.setChecked(state.hard_mining_enabled)
        s.hard_mining_strength_spinbox.setValue(state.hard_mining_strength)
        s.hard_mining_ema_alpha_spinbox.setValue(state.hard_mining_ema_alpha)
        s.skip_uniform_labels_check_box.setChecked(state.skip_uniform_labels)
        s.multi_gpu_check_box.setChecked(state.use_multi_gpu)
        s.torch_compile_check_box.setChecked(state.torch_compile_enabled)
        view = self.__dict__.get('view')
        if view is not None and hasattr(view, 'set_batch_preview_enabled'):
            view.set_batch_preview_enabled(state.show_batch_preview)

        # 3.7  Additional processing flags
        s.enable_crop_processing.setChecked(state.crop_enabled)
        s.enable_resize_processing.setChecked(state.resize_enabled)
        if hasattr(s, '_sync_preprocess_controls'):
            s._sync_preprocess_controls()
        if hasattr(s, 'sync_business_logic_controls'):
            main_state = self.__dict__.get('main_window_state')
            work_mode = getattr(main_state, 'work_mode', '')
            s.sync_business_logic_controls(work_mode)

        # 3.8 Размер обрезки по краям
        s.cut_corner_spinbox.setValue(state.edge_cut_size)

        # 3.9  Target size (two spin boxes)
        s.target_x_size.setValue(state.target_size[0])
        s.target_y_size.setValue(state.target_size[1])

    def _update_work_mode(self) -> str:
        v = self.view
        if v.rb_train_and_recognition.isChecked():
            return WorkMode.train_and_recognition.value
        elif v.rb_train_only.isChecked():
            return WorkMode.train_only.value
        elif v.rb_further_train_model.isChecked():
            return WorkMode.further_training.value
        elif v.rb_recognition.isChecked():
            return WorkMode.recognition_only.value
        else:
            return ''

    def _update_settings_window_state(self):
        s = self.settings_panel

        h = s.horizontal_rotation.isChecked()
        v = s.vertical_rotation.isChecked()
        additional_augmentation = s.additional_augmentation_check_box.isChecked()
        augmentation_brightness_strength = s.augmentation_brightness_spinbox.value()
        augmentation_contrast_strength = s.augmentation_contrast_spinbox.value()
        augmentation_noise_probability = s.augmentation_noise_probability_spinbox.value()
        augmentation_noise_sigma = s.augmentation_noise_sigma_spinbox.value()
        step = s.shift_spinbox.value()
        sample_size = (s.sample_x_size.value(), s.sample_y_size.value())
        model = s.nn_model_type.currentText()
        color_mode = s.color_type.currentText()
        validation = s.validation_check_box.isChecked()
        validation_percent = s.validation_spinbox.value()
        cut_mode = self._update_cut_mode()
        batch_size = s.batch_spinbox.value()
        overlap = s.overlap_spinbox.value()
        log_update_frequency = s.log_update_frequency_spinbox.value()
        optimizer_name = s.optimizer_type.currentText()
        mixed_precision = s.mixed_precision_type.currentText()
        loss_function = s.loss_function_type.currentText()
        dice_loss_weight = s.dice_loss_weight_spinbox.value()
        iou_loss_weight = s.iou_loss_weight_spinbox.value()
        learning_rate = s.learning_rate_spinbox.value()
        weight_decay = s.weight_decay_spinbox.value()
        early_stopping_enabled = s.early_stopping_check_box.isChecked()
        early_stopping_patience = s.early_stopping_patience_spinbox.value()
        early_stopping_min_delta = s.early_stopping_min_delta_spinbox.value()
        early_stopping_restore_best_weights = s.restore_best_weights_check_box.isChecked()
        warmup_enabled = s.warmup_check_box.isChecked()
        warmup_epochs = s.warmup_epochs_spinbox.value()
        warmup_start_factor = s.warmup_start_factor_spinbox.value()
        hard_mining_enabled = s.hard_mining_check_box.isChecked()
        hard_mining_strength = s.hard_mining_strength_spinbox.value()
        hard_mining_ema_alpha = s.hard_mining_ema_alpha_spinbox.value()
        skip_uniform_labels = s.skip_uniform_labels_check_box.isChecked()
        use_multi_gpu = s.multi_gpu_check_box.isChecked()
        torch_compile_enabled = s.torch_compile_check_box.isChecked()
        show_batch_preview = self.view.is_batch_preview_enabled()
        crop_enabled = s.enable_crop_processing.isChecked()
        resize_enabled = s.enable_resize_processing.isChecked()
        edge_cut_size = s.cut_corner_spinbox.value()
        target_size = (s.target_x_size.value(), s.target_y_size.value())

        state = SettingsState(step=step, vertical_rotation=v, horizontal_rotation=h,
                              additional_augmentation=additional_augmentation,
                              augmentation_brightness_strength=augmentation_brightness_strength,
                              augmentation_contrast_strength=augmentation_contrast_strength,
                              augmentation_noise_probability=augmentation_noise_probability,
                              augmentation_noise_sigma=augmentation_noise_sigma,
                              sample_size=sample_size,
                              model=model, color_mode=color_mode, use_validation=validation,
                              validation_percent=validation_percent,
                              sample_cut_mode=cut_mode, batch_size=batch_size, overlap=overlap,
                              crop_enabled=crop_enabled, resize_enabled=resize_enabled, edge_cut_size=edge_cut_size,
                              target_size=target_size,
                              optimizer_name=optimizer_name,
                              mixed_precision=mixed_precision,
                              loss_function=loss_function,
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
                              hard_mining_enabled=hard_mining_enabled,
                              hard_mining_strength=hard_mining_strength,
                              hard_mining_ema_alpha=hard_mining_ema_alpha,
                              log_update_frequency=log_update_frequency,
                              skip_uniform_labels=skip_uniform_labels,
                              use_multi_gpu=use_multi_gpu,
                              torch_compile_enabled=torch_compile_enabled,
                              show_batch_preview=show_batch_preview)

        self.settings_state = state

    def _on_batch_preview_visibility_changed(self, enabled: bool):
        self.settings_state.show_batch_preview = bool(enabled)

    def _log_message_emit(self, data):
        self.view.log_message.emit(data)

    def _train_message_emit(self, data):
        self.view.log_message_with_delete_last.emit(data)

    def _metrics_message_emit(self, data):
        self.view.metrics_message.emit(data)

    def _error_message_emit(self, data):
        message = str(data) if data is not None else 'Произошла ошибка выполнения.'
        self.view.show_warning.emit(message)
        self.view.log_message.emit(f'Ошибка: {message}')

    def _update_cut_mode(self) -> str:
        s = self.settings_panel

        if s.cut_dataset_type.isChecked():
            return SampleCutMode.disk.value
        elif s.no_cut_dataset_type.isChecked():
            return SampleCutMode.online.value
        else:
            return SampleCutMode.online.value

    # ------------------------------------------------------------------ #
    #   Сохранение окон в QSettings
    # ------------------------------------------------------------------ #
    def _save_windows_to_qsettings(self):
        self._update_main_window_state()
        self._update_settings_window_state()

        self._save_main_window_to_qsettings()
        self._save_nn_settings_to_qsettings()

    def _save_main_window_to_qsettings(self):
        save_main_window_state(self.main_window_state)

    def _save_nn_settings_to_qsettings(self):
        state = getattr(self, "settings_state", None) or SettingsState()
        self.settings_state = state
        save_settings_state(state)

    def _reset_settings_to_defaults(self):
        self.settings_state = SettingsState()
        self._apply_settings_to_panel()
        self._update_settings_window_state()
        self._set_max_shift()
        self._calculate_expected_samples()
        self._validate_start_button()

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

    def _on_sample_type_changed(self, typ: str):
        """typ = ''train_and_recognition'', 'recognition_only', further_training."""
        v = self.view

        v.lbl_source.setEnabled(True)
        v.lbl_result.setEnabled(True)
        v.sample_path_group.setEnabled(True)
        v.model_path.setEnabled(True)

        match typ:
            case WorkMode.train_and_recognition.value:
                v.model_path.setEnabled(False)
            case WorkMode.recognition_only.value:
                v.sample_path_group.setEnabled(False)
            case WorkMode.train_only.value:
                v.lbl_source.setEnabled(False)
                v.lbl_result.setEnabled(False)
                v.model_path.setEnabled(False)
        self.main_window_state.work_mode = typ
        if hasattr(self.settings_panel, 'sync_business_logic_controls'):
            self.settings_panel.sync_business_logic_controls(typ)
        self._validate_start_button()

    def _on_start_requested(self):
        self._save_windows_to_qsettings()
        self._update_main_window_state()
        self._update_settings_window_state()
        if not can_start_processing(self.main_window_state):
            self.message_bus.publish('logging', 'Задача не добавлена. Проверьте обязательные поля.')
            return

        task = QueuedTask(
            task_id=self._next_task_id,
            main_window_state=replace(self.main_window_state),
            settings_state=replace(self.settings_state),
        )
        self._next_task_id += 1
        self._queued_tasks.append(task)
        self.message_bus.publish('logging', f'Задача #{task.task_id} добавлена в очередь.')
        self._refresh_queue_view(selected_task_id=task.task_id)
        self._start_next_task_if_possible()

    def _on_stop_requested(self):
        if self.neuaral_handler is None:
            return
        self._active_stop_requested = True
        self.neuaral_handler.stop()
        if self._active_task is not None:
            self.message_bus.publish('logging', f'Остановлена активная задача #{self._active_task.task_id}.')

    def _on_release_memory_requested(self):
        gc.collect()
        self.message_bus.publish('logging', 'Выполнена очистка памяти Python.')

    def _on_queue_remove_requested(self):
        row = self.view.get_selected_queue_row()
        if row < 0 or row >= len(self._queued_tasks):
            return
        task = self._queued_tasks[row]
        if self._active_task is not None and task.task_id == self._active_task.task_id:
            self.message_bus.publish('logging', f'Нельзя убрать активную задачу #{task.task_id}.')
            return
        self._queued_tasks.pop(row)
        self.message_bus.publish('logging', f'Задача #{task.task_id} удалена из очереди.')
        self._refresh_queue_view(selected_row=min(row, len(self._queued_tasks) - 1))

    def _on_queue_pause_toggle_requested(self):
        row = self.view.get_selected_queue_row()
        if row < 0 or row >= len(self._queued_tasks):
            return
        task = self._queued_tasks[row]
        if self._active_task is not None and task.task_id == self._active_task.task_id:
            self.message_bus.publish('logging', f'Нельзя поставить на паузу активную задачу #{task.task_id}.')
            return
        task.paused = not task.paused
        state = 'поставлена на паузу' if task.paused else 'снята с паузы'
        self.message_bus.publish('logging', f'Задача #{task.task_id} {state}.')
        self._refresh_queue_view(selected_task_id=task.task_id)
        self._start_next_task_if_possible()

    def _refresh_queue_view(self, selected_row: int = -1, selected_task_id: int | None = None):
        items: list[str] = []
        resolved_selected_row = selected_row
        for idx, task in enumerate(self._queued_tasks):
            status = 'в очереди'
            if task.paused:
                status = 'на паузе'
            if self._active_task is not None and task.task_id == self._active_task.task_id:
                status = 'выполняется'
            mode = task.main_window_state.work_mode or 'unknown'
            items.append(f'#{task.task_id} | {mode} | {status}')
            if selected_task_id is not None and task.task_id == selected_task_id:
                resolved_selected_row = idx
        self.view.set_task_queue_items(items, resolved_selected_row)

    def _start_next_task_if_possible(self):
        if self.neuaral_handler is not None and self.neuaral_handler.isRunning():
            return
        next_task = next((task for task in self._queued_tasks if not task.paused), None)
        if next_task is None:
            self._active_task = None
            self.view.toggle_start_stop.emit(False)
            return

        self._active_task = next_task
        self._active_stop_requested = False
        self._refresh_queue_view(selected_task_id=next_task.task_id)
        self.view.toggle_start_stop.emit(True)
        self._start_task(next_task)

    def _start_task(self, task: QueuedTask):
        os.environ['NEURALIMAGE_TORCH_COMPILE'] = '1' if task.settings_state.torch_compile_enabled else '0'
        self.message_bus.publish(
            'logging',
            f'torch.compile {"enabled" if task.settings_state.torch_compile_enabled else "disabled"} by UI setting.',
        )
        work_mode, training_settings, recognition_parameters = build_workflow_parameters(
            task.main_window_state, task.settings_state
        )
        if work_mode is None:
            self.message_bus.publish('error', f'Задача #{task.task_id}: не удалось определить режим работы.')
            self._queued_tasks = [t for t in self._queued_tasks if t.task_id != task.task_id]
            self._active_task = None
            self.view.toggle_start_stop.emit(False)
            self._refresh_queue_view()
            self._start_next_task_if_possible()
            return

        self.neuaral_handler = GeneralNeuralHandlerThread(
            work_mode=work_mode,
            recognition_parameters=recognition_parameters,
            tranining_parameters=training_settings,
            message_bus=self.message_bus,
            callback=self._on_stop_requested,
        )
        self.neuaral_handler.ask.connect(self._thread_ask)
        self.neuaral_handler.finished.connect(self._on_task_finished)
        self.neuaral_handler.start()
        self.message_bus.publish('logging', f'Запущена задача #{task.task_id}.')

    def _on_task_finished(self):
        finished_task = self._active_task
        if finished_task is not None:
            self._queued_tasks = [t for t in self._queued_tasks if t.task_id != finished_task.task_id]
            if self._active_stop_requested:
                self.message_bus.publish('logging', f'Задача #{finished_task.task_id} остановлена.')
            else:
                self.message_bus.publish('logging', f'Задача #{finished_task.task_id} завершена.')
        self.neuaral_handler = None
        self._active_task = None
        self._active_stop_requested = False
        self._refresh_queue_view()
        self._start_next_task_if_possible()
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
            self.view.allow_close()

    # ------------------------------------------------------------------ #
    #   Обработчик UI событий панели
    # ------------------------------------------------------------------ #

    def _calculate_expected_samples(self):
        self._update_settings_window_state()
        calculator_settings = self.get_cut_settings_from_window_state()
        self.sample_calculator.set_settings(calculator_settings)
        self._set_sample_number()

    def get_cut_settings_from_window_state(self) -> CutSettings:
        s = self.settings_state
        return CutSettings(step=s.step,
                           x_size=s.sample_size[0],
                           y_size=s.sample_size[1],
                           vertical_rotation=s.vertical_rotation,
                           horizontal_rotation=s.horizontal_rotation,
                           color_mode=s.color_mode,
                           model=s.model,
                           additional_augmentation=s.additional_augmentation,
                           )

    def _set_sample_number(self):
        total_samples = len(self.sample_calculator)
        self.settings_panel.samples_number.setText(f'Кадров в выборке: {total_samples}')

    def _set_max_shift(self):
        s = self.settings_panel
        x_size = s.sample_x_size.value()
        y_size = s.sample_y_size.value()

        min_size = min(x_size, y_size)

        s.shift_spinbox.setMaximum(min_size)
        if s.shift_spinbox.value() > min_size:
            s.shift_spinbox.setValue(min_size)

    # ------------------------------------------------------------------ #
    #   Валидация и управление кнопкой "Начать"
    # ------------------------------------------------------------------ #
    def _validate_start_button(self):
        """Проверяем, что все обязательные поля заполнены."""
        self._update_main_window_state()
        self.view.enable_start.emit(can_start_processing(self.main_window_state))

    # ------------------------------------------------------------------ #
    #   Управление началом потоков
    # ------------------------------------------------------------------ #
    def _thread_ask(self, question: str, header: str = 'Обратите внимание'):
        reply = QMessageBox.question(None, header, question,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No,
                                     )

        answer = True if reply == QMessageBox.StandardButton.Yes else False

        handler = self.neuaral_handler
        if handler is not None:
            handler.answer.emit(answer)


class GeneralNeuralHandlerThread(QThread):
    ask = QtCore.pyqtSignal(str, str)  # title, text
    answer = QtCore.pyqtSignal(bool)

    def __init__(self, work_mode: WorkMode,
                 message_bus: AbstractMessageBus,
                 recognition_parameters: RecognitionParameters | None = None,
                 tranining_parameters: TrainingParameters | None = None,
                 callback: Callable[..., None] | None = None):
        super().__init__()
        self._last_answer = False
        self.main_logic = GeneralNeuralHandler(work_mode=work_mode,
                                               recogniton_parameters=recognition_parameters,
                                               tranining_parameters=tranining_parameters,
                                               question_module=self.check,
                                               message_bus=message_bus)
        self.answer.connect(self._store_answer)

    def run(self):
        self.main_logic.start()

    def check(self, text, theme):
        self.ask.emit(
            text,
            theme
        )
        # Ждем ответ в локальном event-loop
        loop = QtCore.QEventLoop()
        self.answer.connect(loop.quit)
        loop.exec()  # блокирует только этот метод
        answer = self._last_answer
        return answer

    def stop(self):
        self.main_logic.stop_execution()

    @QtCore.pyqtSlot(bool)
    def _store_answer(self, val: bool):
        self._last_answer = val

