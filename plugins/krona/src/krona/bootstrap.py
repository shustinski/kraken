from __future__ import annotations

from pathlib import Path
import sys

from krona.application.logic_functions import ExtractLogicFunctions
from krona.application.structural_analysis import AnalyzeSequentialStructure
from krona.application.use_cases import LoadSceneData, LoadTopLevelNetlist
from krona.infrastructure.edif_repository import EdifRepository


class ApplicationContainer:
    def __init__(self):
        self.repository = EdifRepository()
        self.load_netlist = LoadTopLevelNetlist(self.repository)
        self.load_scene_data = LoadSceneData(self.repository)
        self.extract_logic_functions = ExtractLogicFunctions(self.repository)
        self.analyze_sequential_structure = AnalyzeSequentialStructure(self.repository)


def parse_to_dict(path: str | Path) -> dict:
    container = ApplicationContainer()
    return container.load_netlist.execute(path)


def logic_functions_to_dict(path: str | Path) -> dict:
    container = ApplicationContainer()
    return container.extract_logic_functions.execute(path)


def structural_analysis_to_dict(path: str | Path) -> dict:
    container = ApplicationContainer()
    return container.analyze_sequential_structure.execute(path)


def run_gui(initial_path: str | Path | None = None) -> int:
    from PyQt6.QtWidgets import QApplication
    from kraken_core.qt import configure_application_identity
    from krona.presentation.qt.theme.manager import ThemeManager
    from krona.presentation.qt.ui_strings import load_ui_strings
    from krona.presentation.qt.window import EdifViewerWindow

    container = ApplicationContainer()
    app = QApplication(sys.argv)
    configure_application_identity(app, app_id="Kraken.Krona", icon_name="krona")
    theme_manager = ThemeManager()
    selected_theme = theme_manager.load_saved_theme(default="Dark")
    selected_language = theme_manager.load_saved_language(default="English")
    theme_manager.apply_theme(app, selected_theme)
    window = EdifViewerWindow(
        load_scene_data_use_case=container.load_scene_data,
        extract_logic_functions_use_case=container.extract_logic_functions,
        initial_path=initial_path,
        theme_manager=theme_manager,
        initial_theme=selected_theme,
        initial_language=selected_language,
        ui_strings=load_ui_strings(selected_language),
    )
    window.show()
    return app.exec()
