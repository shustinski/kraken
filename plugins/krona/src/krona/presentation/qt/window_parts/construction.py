"""window: construction."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtGui import QIcon
from krona.application.ports import SceneData
from krona.application.logic_functions import ExtractLogicFunctions
from krona.application.use_cases import LoadSceneData
from krona.presentation.qt.theme.manager import ThemeManager
from krona.presentation.qt.ui_strings import load_ui_strings
from updater.qt import QtUpdateController

from krona.presentation.qt.window_parts.resolve_window_icon_path import _resolve_window_icon_path


class ConstructionMixin:
    def __init__(
        self,
        load_scene_data_use_case: LoadSceneData,
        extract_logic_functions_use_case: ExtractLogicFunctions,
        initial_path: str | Path | None = None,
        theme_manager: ThemeManager | None = None,
        initial_theme: str = "Dark",
        initial_language: str = "English",
        ui_strings: dict[str, str] | None = None,
    ):
        super().__init__()
        self._load_scene_data = load_scene_data_use_case
        self._extract_logic_functions = extract_logic_functions_use_case
        self._current_path = Path(initial_path) if initial_path else None
        self._theme_manager = theme_manager
        self._current_theme = initial_theme
        self._current_language = initial_language
        self._scene_data: SceneData | None = None
        self._logic_data: dict | None = None
        self._logic_highlight_targets: dict[str, dict] = {}
        self._filtered_net_indexes: list[int] = []
        self._filtered_instance_indexes: list[int] = []
        self._instance_to_connections: dict[str, list[tuple[str, str]]] = {}
        self._ui = ui_strings if ui_strings is not None else load_ui_strings(self._current_language)
        self._update_controller: QtUpdateController | None = None

        self.setWindowTitle(self._t("window.title", "EDF Netlist Viewer"))
        icon_path = _resolve_window_icon_path()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1300, 820)
        self.setAcceptDrops(True)
        self._init_ui()
        if self._current_path:
            self.load_edf(self._current_path)

    def _t(self, key: str, default: str, **kwargs) -> str:
        template = self._ui.get(key, default)
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, ValueError):
                return default.format(**kwargs)
        return template
