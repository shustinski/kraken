"""state_store: settings state storage dict."""
from __future__ import annotations
from typing import Any
from neuralimage.application.dto import SettingsState

from neuralimage.infrastructure.config.state_store_parts.build_settings_state import _build_settings_state

def _settings_state_from_storage_dict(data: dict[str, Any]) -> SettingsState:
    from neuralimage.infrastructure.config.state_store_parts.q_settings_state_store import _coerce_bool, _coerce_float, _coerce_int, _coerce_str
    return _build_settings_state(
        read_bool=lambda key, default: _coerce_bool(data.get(key), default),
        read_int=lambda key, default: _coerce_int(data.get(key), default),
        read_float=lambda key, default: _coerce_float(data.get(key), default),
        read_str=lambda key, default: _coerce_str(data.get(key), default),
    )
