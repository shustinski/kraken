import multiprocessing as mp
import gc
import os
import threading
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Callable
from neuralimage.infrastructure.dataset_source import DatasetStreamSource
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
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.view import MainView, SettingsPanel
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

from .main_presenter_parts.construction import ConstructionMixin
from .main_presenter_parts.connect_view_signals import ConnectViewSignalsMixin
from .main_presenter_parts.apply_settings_panel import ApplySettingsPanelMixin
from .main_presenter_parts.update_work_mode import UpdateWorkModeMixin
from .main_presenter_parts.update_settings_window_state import UpdateSettingsWindowStateMixin
from .main_presenter_parts.open_augmentation_preview import OpenAugmentationPreviewMixin
from .main_presenter_parts.apply_augmentation_preview_settings import ApplyAugmentationPreviewSettingsMixin
from .main_presenter_parts.open_rare_patch_editor import OpenRarePatchEditorMixin
from .main_presenter_parts.shutdown import ShutdownMixin
from .main_presenter_parts.thread_ask import ThreadAskMixin
from .main_presenter_parts.resolve_latest_release import ResolveLatestReleaseMixin
from .main_presenter_parts.rare_patch_editor_preparation import (
    _ValidationGradientPluginWindow,
    _format_auto_answer_button_text,
    _tk_filedialog,
    SampleCountSignals,
    _should_schedule_startup_update_check,
    RarePatchEditorPreparationThread,
)
from .main_presenter_parts.general_neural_handler_thread import GeneralNeuralHandlerThread
from .main_presenter_parts.app_update_download_thread import (
    AppUpdateCheckThread,
    AppUpdateDownloadThread,
)

class MainPresenter(
    ConstructionMixin,
    ConnectViewSignalsMixin,
    ApplySettingsPanelMixin,
    UpdateWorkModeMixin,
    UpdateSettingsWindowStateMixin,
    OpenAugmentationPreviewMixin,
    ApplyAugmentationPreviewSettingsMixin,
    OpenRarePatchEditorMixin,
    ShutdownMixin,
    ThreadAskMixin,
    ResolveLatestReleaseMixin,
    QObject,
):
    _queue_metrics_received = QtCore.pyqtSignal(object)
    _queue_error_received = QtCore.pyqtSignal(str)
    SIMPLE_WORKFLOW_PRESETS = {
        'conductors': resolve_resource_path('conductors_workflow.json'),
        'contacts': resolve_resource_path('contacts_workflow.json'),
        'memory': resolve_resource_path('memory_workflow.json'),
    }
    """
    Связывает View и Model. Содержит всю бизнес-логику
    (валидацию, формирование параметров, запуск/остановку потока).
    """

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _ValidationGradientPluginWindow,
    _format_auto_answer_button_text,
    _tk_filedialog,
    SampleCountSignals,
    _should_schedule_startup_update_check,
    RarePatchEditorPreparationThread,
    GeneralNeuralHandlerThread,
    AppUpdateCheckThread,
    AppUpdateDownloadThread,
)
