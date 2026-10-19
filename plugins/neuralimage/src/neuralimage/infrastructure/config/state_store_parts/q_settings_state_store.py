"""state_store: q settings state store."""
from __future__ import annotations
import json
import os
from dataclasses import is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from PyQt6.QtCore import QSettings
from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.services.workflow_mapper import build_workflow_parameters
from neuralimage.lib.data_interfaces import (
    WorkMode,
)

from neuralimage.infrastructure.config.state_store_parts.module_state import (
    MAIN_WINDOW_APP,
    MAIN_WINDOW_ORG,
    SETTINGS_APP,
    SETTINGS_ORG,
    WORKFLOW_SNAPSHOT_FILENAME,
    WORKFLOW_SNAPSHOT_FORMAT_VERSION,
)

from neuralimage.infrastructure.config.state_store_parts.build_settings_state import _build_settings_state

from neuralimage.infrastructure.config.state_store_parts.settings_state_storage_dict import _settings_state_from_storage_dict

from neuralimage.infrastructure.config.state_store_parts.settings_state_storage_dict_2 import _settings_state_to_storage_dict

def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    if value is None:
        return default
    return bool(value)

def _coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)

def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _coerce_json_object(value: Any, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raw = _coerce_str(value, '')
    if not raw.strip():
        return dict(default)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return dict(default)
    if not isinstance(payload, dict):
        return dict(default)
    return payload

def _jsonify_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonify_value(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonify_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_value(item) for item in value]
    return value

def _sanitize_workflow_snapshot_payload(training: Any, recognition: Any) -> tuple[Any, Any]:
    serialized_training = _jsonify_value(training)
    serialized_recognition = _jsonify_value(recognition)
    if isinstance(serialized_recognition, dict):
        serialized_recognition.pop('source_files', None)
    return serialized_training, serialized_recognition

def resolve_workflow_snapshot_path(main_state: MainWindowState) -> Path:
    sample_folder = Path(main_state.sample_folder)
    label_folder = Path(main_state.label_folder)
    fallback_parent = sample_folder.parent if str(sample_folder) else Path.cwd()
    if not str(sample_folder) or not str(label_folder):
        return fallback_parent / WORKFLOW_SNAPSHOT_FILENAME
    target_dir = sample_folder.parent
    if sample_folder.parent == label_folder.parent:
        target_dir = sample_folder.parent
    return target_dir / WORKFLOW_SNAPSHOT_FILENAME

def create_workflow_snapshot_payload(
    main_state: MainWindowState,
    settings_state: SettingsState,
    workflow_snapshot: tuple[WorkMode | None, Any, Any] | None = None,
) -> dict[str, Any]:
    from neuralimage.infrastructure.config.state_store_parts.build_main_window_state import _main_window_state_to_storage_dict
    work_mode, training, recognition = workflow_snapshot or build_workflow_parameters(main_state, settings_state)
    serialized_training, serialized_recognition = _sanitize_workflow_snapshot_payload(training, recognition)
    return {
        'format_version': WORKFLOW_SNAPSHOT_FORMAT_VERSION,
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'main_window_state': _main_window_state_to_storage_dict(main_state),
        'settings_state': _settings_state_to_storage_dict(settings_state),
        'workflow': {
            'work_mode': work_mode.value if work_mode is not None else None,
            'training': serialized_training,
            'recognition': serialized_recognition,
        },
    }

def save_workflow_snapshot(
    main_state: MainWindowState,
    settings_state: SettingsState,
    destination: Path | None = None,
    workflow_snapshot: tuple[WorkMode | None, Any, Any] | None = None,
) -> Path:
    snapshot_path = Path(destination) if destination is not None else resolve_workflow_snapshot_path(main_state)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = create_workflow_snapshot_payload(main_state, settings_state, workflow_snapshot=workflow_snapshot)
    with snapshot_path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return snapshot_path

def load_workflow_snapshot(snapshot_path: Path | str) -> tuple[MainWindowState, SettingsState]:
    from neuralimage.infrastructure.config.state_store_parts.build_main_window_state import _main_window_state_from_storage_dict
    path = Path(snapshot_path)
    try:
        with path.open('r', encoding='utf-8') as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f'Некорректный JSON: {error}') from error

    if not isinstance(payload, dict):
        raise ValueError('Файл параметров должен содержать JSON-объект.')
    if payload.get('format_version') != WORKFLOW_SNAPSHOT_FORMAT_VERSION:
        raise ValueError('Неподдерживаемая версия файла параметров.')

    main_payload = payload.get('main_window_state')
    settings_payload = payload.get('settings_state')
    if not isinstance(main_payload, dict) or not isinstance(settings_payload, dict):
        raise ValueError('В файле параметров отсутствуют main_window_state или settings_state.')

    main_state = _main_window_state_from_storage_dict(main_payload)
    settings_state = _settings_state_from_storage_dict(settings_payload)
    return main_state, settings_state

class QSettingsStateStore:
    def _settings(self, organization: str, application: str) -> QSettings:
        root = os.getenv('NEURALIMAGE_SETTINGS_DIR')
        if root:
            settings_root = Path(root)
            settings_root.mkdir(parents=True, exist_ok=True)
            return QSettings(
                str(settings_root / f'{organization}_{application}.ini'),
                QSettings.Format.IniFormat,
            )
        return QSettings(organization, application)

    def load_main_window_state(self) -> MainWindowState:
        from .build_main_window_state import _build_main_window_state

        settings = self._settings(MAIN_WINDOW_ORG, MAIN_WINDOW_APP)
        state = _build_main_window_state(
            read_str=lambda key, default: settings.value(key, default, type=str),
            read_int=lambda key, default: settings.value(key, default, type=int),
        )
        settings.sync()
        return state

    def save_main_window_state(self, state: MainWindowState) -> None:
        from .build_main_window_state import _main_window_state_to_storage_dict

        settings = self._settings(MAIN_WINDOW_ORG, MAIN_WINDOW_APP)
        for key, value in _main_window_state_to_storage_dict(state).items():
            settings.setValue(key, value)
        settings.sync()

    def load_settings_state(self) -> SettingsState:
        settings = self._settings(SETTINGS_ORG, SETTINGS_APP)
        state = _build_settings_state(
            read_bool=lambda key, default: settings.value(key, default, type=bool),
            read_int=lambda key, default: settings.value(key, default, type=int),
            read_float=lambda key, default: settings.value(key, default, type=float),
            read_str=lambda key, default: settings.value(key, default, type=str),
        )
        settings.sync()
        return state

    def save_settings_state(self, state: SettingsState) -> None:
        settings = self._settings(SETTINGS_ORG, SETTINGS_APP)
        for key, value in _settings_state_to_storage_dict(state).items():
            settings.setValue(key, value)
        settings.sync()
