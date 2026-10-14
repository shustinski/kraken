from __future__ import annotations
from pathlib import Path
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,

)
from krona.application.ports import SceneData
from krona.application.logic_functions import ExtractLogicFunctions
from krona.application.use_cases import LoadSceneData
from krona.presentation.qt.scene_renderer import EdifSceneRenderer
from krona.presentation.qt.theme.manager import ThemeManager
from krona.presentation.qt.ui_strings import available_languages, load_ui_strings
from krona.version import __version__
from kraken_core.qt import resolve_icon_path
from kraken_core.theme import add_theme_menu
from updater.qt import QtUpdateController

from .window_parts.construction import ConstructionMixin
from .window_parts.construction_2 import Construction2Mixin
from .window_parts.build_logic_tab import BuildLogicTabMixin
from .window_parts.load_edf import LoadEdfMixin
from .window_parts.refresh_logic_tab import RefreshLogicTabMixin
from .window_parts.logic_port_labels_by import LogicPortLabelsByMixin
from .window_parts.render_sequential_elements_summary import RenderSequentialElementsSummaryMixin
from .window_parts.refresh_diagnostics_tab import RefreshDiagnosticsTabMixin
from .window_parts.reload_ui_texts import ReloadUiTextsMixin
from .window_parts.resolve_window_icon_path import _resolve_window_icon_path

class EdifViewerWindow(
    ConstructionMixin,
    Construction2Mixin,
    BuildLogicTabMixin,
    LoadEdfMixin,
    RefreshLogicTabMixin,
    LogicPortLabelsByMixin,
    RenderSequentialElementsSummaryMixin,
    RefreshDiagnosticsTabMixin,
    ReloadUiTextsMixin,
    QMainWindow,
):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _resolve_window_icon_path,
)
