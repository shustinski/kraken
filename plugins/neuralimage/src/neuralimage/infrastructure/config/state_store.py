import configparser
import json
import os
from dataclasses import is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from PyQt6.QtCore import QSettings
from neuralimage.application.dto import MainWindowState, SettingsState, normalize_main_window_mode_state
from neuralimage.application.ports import StateStore
from neuralimage.application.services.workflow_mapper import build_workflow_parameters
from neuralimage.lib.data_interfaces import (
    WorkMode,
    normalize_confidence_save_mode,
    normalize_multi_gpu_mode,
    normalize_patch_batch_sync_mode,
    normalize_scheduler_name,
    normalize_validation_source,
    normalize_work_mode,
)
from neuralimage.lib.loss_config import (
    deserialize_loss_term_weights,
    dominant_loss_function,
    normalize_loss_term_name,
    resolve_loss_term_weights,
    serialize_loss_term_weights,
)

from .state_store_parts.module_state import (
    MAIN_WINDOW_ORG,
    MAIN_WINDOW_APP,
    SETTINGS_ORG,
    SETTINGS_APP,
    INI_SECTION_MAIN,
    INI_SECTION_SETTINGS,
    WORKFLOW_SNAPSHOT_FILENAME,
    WORKFLOW_SNAPSHOT_FORMAT_VERSION,
)
from .state_store_parts.build_main_window_state import (
    _build_main_window_state,
    _main_window_state_to_storage_dict,
    _main_window_state_from_storage_dict,
)
from .state_store_parts.build_settings_state import _build_settings_state
from .state_store_parts.settings_state_storage_dict import _settings_state_from_storage_dict
from .state_store_parts.settings_state_storage_dict_2 import _settings_state_to_storage_dict
from .state_store_parts.q_settings_state_store import (
    _coerce_bool,
    _coerce_str,
    _coerce_int,
    _coerce_float,
    _coerce_json_object,
    _jsonify_value,
    _sanitize_workflow_snapshot_payload,
    resolve_workflow_snapshot_path,
    create_workflow_snapshot_payload,
    save_workflow_snapshot,
    load_workflow_snapshot,
    QSettingsStateStore,
)
from .state_store_parts.ini_state_store import IniStateStore
from .state_store_parts.create_state_store import (
    create_state_store,
    load_main_window_state,
    save_main_window_state,
    load_settings_state,
    save_settings_state,
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    MAIN_WINDOW_ORG,
    MAIN_WINDOW_APP,
    SETTINGS_ORG,
    SETTINGS_APP,
    INI_SECTION_MAIN,
    INI_SECTION_SETTINGS,
    WORKFLOW_SNAPSHOT_FILENAME,
    WORKFLOW_SNAPSHOT_FORMAT_VERSION,
    _build_main_window_state,
    _main_window_state_to_storage_dict,
    _main_window_state_from_storage_dict,
    _build_settings_state,
    _settings_state_from_storage_dict,
    _settings_state_to_storage_dict,
    _coerce_bool,
    _coerce_str,
    _coerce_int,
    _coerce_float,
    _coerce_json_object,
    _jsonify_value,
    _sanitize_workflow_snapshot_payload,
    resolve_workflow_snapshot_path,
    create_workflow_snapshot_payload,
    save_workflow_snapshot,
    load_workflow_snapshot,
    QSettingsStateStore,
    IniStateStore,
    create_state_store,
    load_main_window_state,
    save_main_window_state,
    load_settings_state,
    save_settings_state,
)
