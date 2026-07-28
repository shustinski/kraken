"""PyQt6 presentation components for Kraken Project Manager."""

from .models import LayerListItem, LayerListModel, ProjectListItem, ProjectListModel
from .layer_management import (
    LayerManagerDialog,
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)
from .layer_creation import LayerCreationDialog
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
    "LayerCreationDialog",
    "LayerManagerDialog",
    "LayerPipelineSnapshot",
    "MatrixLod",
    "MyWorkPage",
    "PerformersPage",
    "PluginsPage",
    "ProjectCatalogPage",
    "ProjectListItem",
    "ProjectListModel",
    "ProjectManagerShell",
    "ProjectWorkspacePage",
    "PipelineLane",
    "PipelineNode",
    "StatisticsPage",
]

