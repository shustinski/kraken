"""main_presenter: construction."""
from __future__ import annotations
import multiprocessing as mp
import threading
from PyQt6 import QtCore
from neuralimage.application.ports import StateStore
from neuralimage.application.services import (
    ProcessingSession,
)
from neuralimage.lib.data_interfaces import (
    CutSettings,
)
from neuralimage.lib.images import SampleWorker
from neuralimage.lib.update_checker import (
    load_selected_update_channel,
    load_update_client_config,
)
from neuralimage.model.NeuralNetwork import get_registered_model_names_by_type
from neuralimage.view import MainView, SettingsPanel

from neuralimage.presenter.main_presenter_parts.rare_patch_editor_preparation import (
    SampleCountSignals,
    _should_schedule_startup_update_check,
)


class ConstructionMixin:
    def __init__(self, state_store: StateStore):
        super().__init__()
        self._state_store = state_store

        self.sample_calculator = SampleWorker()

        # Инициализируем панель с настройками нейросети
        self.settings_panel = SettingsPanel()
        self._settings_configuration_valid = not bool(
            getattr(self.settings_panel, '_configuration_errors', ())
        )

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
        self._shutdown_requested = False
        self._shutdown_complete = False
        self._update_check_manual = False
        self._sample_count_worker_thread: threading.Thread | None = None
        self._sample_count_cancel_event = threading.Event()
        self._sample_count_request_serial = 0
        self._latest_sample_count_request_id = 0
        self._debounced_sample_count_request: tuple[int, str, CutSettings, object] | None = None
        self._pending_sample_count_request: tuple[int, str, CutSettings, object] | None = None
        self._sample_count_cache_path: str | None = None
        self._sample_count_cache_sizes: list[tuple[int, int]] | None = None
        self._sample_count_signals = SampleCountSignals()
        self._sample_count_signals.calculated.connect(self._on_sample_count_calculated)
        self._sample_count_signals.failed.connect(self._on_sample_count_failed)
        self._queue_metrics_received.connect(self._on_queue_metrics_received)
        self._queue_error_received.connect(self._on_queue_error_received)
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
        from neuralimage.remote.ui import install_remote_controls
        install_remote_controls(self)
        QtCore.QTimer.singleShot(0, self._calculate_expected_samples)
        if _should_schedule_startup_update_check():
            QtCore.QTimer.singleShot(0, self._start_update_check)
