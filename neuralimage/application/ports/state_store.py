from typing import Protocol

from application.dto import MainWindowState, SettingsState


class StateStore(Protocol):
    def load_main_window_state(self) -> MainWindowState:
        ...

    def save_main_window_state(self, state: MainWindowState) -> None:
        ...

    def load_settings_state(self) -> SettingsState:
        ...

    def save_settings_state(self, state: SettingsState) -> None:
        ...
