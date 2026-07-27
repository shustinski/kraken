"""PyQt6 presentation components for Kraken Project Manager."""

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
    FrameRect,
    FrameSelection,
    GridDimensions,
    GridDimensionsWidget,
    GridOrientation,
    MatrixLod,
)

__all__ = [
    "AdministrationPage",
    "FrameCellData",
    "FrameContext",
    "FrameMatrixView",
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

