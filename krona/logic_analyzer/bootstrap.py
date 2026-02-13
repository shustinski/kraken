from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

from logic_analyzer.application.logic_functions import ExtractLogicFunctions
from logic_analyzer.application.use_cases import LoadSceneData, LoadTopLevelNetlist
from logic_analyzer.infrastructure.edif_repository import EdifRepository
from logic_analyzer.presentation.qt.theme.manager import ThemeManager
from logic_analyzer.presentation.qt.window import EdifViewerWindow


class ApplicationContainer:
    def __init__(self):
        self.repository = EdifRepository()
        self.load_netlist = LoadTopLevelNetlist(self.repository)
        self.load_scene_data = LoadSceneData(self.repository)
        self.extract_logic_functions = ExtractLogicFunctions(self.repository)


def parse_to_dict(path: str | Path) -> dict:
    container = ApplicationContainer()
    return container.load_netlist.execute(path)


def logic_functions_to_dict(path: str | Path) -> dict:
    container = ApplicationContainer()
    return container.extract_logic_functions.execute(path)


def run_gui(initial_path: str | Path | None = None) -> int:
    container = ApplicationContainer()
    app = QApplication(sys.argv)
    theme_manager = ThemeManager()
    selected_theme = theme_manager.load_saved_theme(default="Dark")
    theme_manager.apply_theme(app, selected_theme)
    window = EdifViewerWindow(
        load_scene_data_use_case=container.load_scene_data,
        extract_logic_functions_use_case=container.extract_logic_functions,
        initial_path=initial_path,
        theme_manager=theme_manager,
        initial_theme=selected_theme,
    )
    window.show()
    return app.exec()
