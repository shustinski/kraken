"""state_store: create state store."""
from __future__ import annotations
import os
from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.application.ports import StateStore

from neuralimage.infrastructure.config.state_store_parts.ini_state_store import IniStateStore

from neuralimage.infrastructure.config.state_store_parts.q_settings_state_store import QSettingsStateStore

def create_state_store(*, default_backend: str = 'qsettings') -> StateStore:
    backend = os.getenv('NEURALIMAGE_STATE_BACKEND', default_backend).strip().lower()
    if backend == 'ini':
        return IniStateStore()
    return QSettingsStateStore()

def load_main_window_state() -> MainWindowState:
    return create_state_store().load_main_window_state()

def save_main_window_state(state: MainWindowState) -> None:
    create_state_store().save_main_window_state(state)

def load_settings_state() -> SettingsState:
    return create_state_store().load_settings_state()

def save_settings_state(state: SettingsState) -> None:
    create_state_store().save_settings_state(state)
