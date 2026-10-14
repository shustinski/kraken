"""state_store: build main window state."""
from __future__ import annotations
import json
from typing import Any
from neuralimage.application.dto import MainWindowState, normalize_main_window_mode_state
from neuralimage.lib.data_interfaces import (
    normalize_work_mode,
)

from neuralimage.infrastructure.config.state_store_parts.q_settings_state_store import (
    _coerce_int,
    _coerce_json_object,
    _coerce_str,
)

def _build_main_window_state(
    *,
    read_str,
    read_int,
) -> MainWindowState:
    defaults = MainWindowState()
    mode_state = _coerce_json_object(read_str('mode_state_json', ''), default={})
    return MainWindowState(
        work_mode=normalize_work_mode(read_str('work_mode', defaults.work_mode)),
        source_folder=read_str('source_path', defaults.source_folder),
        result_folder=read_str('result_path', defaults.result_folder),
        model_path=read_str('model_path', defaults.model_path),
        label_folder=read_str('label_path', defaults.label_folder),
        sample_folder=read_str('sample_path', defaults.sample_folder),
        epochs=read_int('epochs', defaults.epochs),
        ui_mode=read_str('ui_mode', defaults.ui_mode),
        mode_state=normalize_main_window_mode_state(mode_state),
    )

def _main_window_state_to_storage_dict(state: MainWindowState) -> dict[str, str | int]:
    return {
        'work_mode': normalize_work_mode(state.work_mode),
        'source_path': state.source_folder,
        'result_path': state.result_folder,
        'model_path': state.model_path,
        'label_path': state.label_folder,
        'sample_path': state.sample_folder,
        'epochs': int(state.epochs),
        'ui_mode': str(getattr(state, 'ui_mode', 'simple') or 'simple'),
        'mode_state_json': json.dumps(
            normalize_main_window_mode_state(getattr(state, 'mode_state', {})),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }

def _main_window_state_from_storage_dict(data: dict[str, Any]) -> MainWindowState:
    return _build_main_window_state(
        read_str=lambda key, default: _coerce_str(data.get(key), default),
        read_int=lambda key, default: _coerce_int(data.get(key), default),
    )
