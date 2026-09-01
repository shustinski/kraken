"""PyQt6 presentation components for Kraken Project Manager."""

from .analysis_ui import AnalysisRunsPanel, AnalysisSetupDialog
from .models import LayerListItem, LayerListModel, ProjectListItem, ProjectListModel
from .pages import (
    AdministrationPage,
    MyWorkPage,
    PerformersPage,
    PluginsPage,
    ProjectCatalogPage,
    ProjectWorkspacePage,
    StatisticsPage,
)
from .shell import ProjectManagerShell
from .widgets import (
    FrameCellData,
    FrameContext,
    FrameMatrixView,
    FrameMatrixWidget,
    FrameRect,
    FrameSelection,
    GridDimensions,
    GridDimensionsWidget,
    GridOrientation,
    MatrixLod,
)

__all__ = [
    "AdministrationPage",
    "AnalysisRunsPanel",
    "AnalysisSetupDialog",
    "FrameCellData",
    "FrameContext",
    "FrameMatrixView",
    "FrameMatrixWidget",
    "FrameRect",
    "FrameSelection",
    "GridDimensions",
    "GridDimensionsWidget",
    "GridOrientation",
    "LayerListItem",
    "LayerListModel",
    "MatrixLod",
    "MyWorkPage",
    "PerformersPage",
    "PluginsPage",
    "ProjectCatalogPage",
    "ProjectListItem",
    "ProjectListModel",
    "ProjectManagerShell",
    "ProjectWorkspacePage",
    "StatisticsPage",
]

