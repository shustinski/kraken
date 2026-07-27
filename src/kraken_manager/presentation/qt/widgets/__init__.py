"""Reusable project-manager widgets."""

from .clickable_label import ClickableLabel
from .frame_matrix import (
    FrameCellData,
    FrameContext,
    FrameMatrixView,
    FrameMatrixWidget,
    FrameRect,
    FrameSelection,
    MatrixLod,
)
from .grid_dimensions import GridDimensions, GridDimensionsWidget, GridOrientation

__all__ = [
    "ClickableLabel",
    "FrameCellData",
    "FrameContext",
    "FrameMatrixView",
    "FrameMatrixWidget",
    "FrameRect",
    "FrameSelection",
    "GridDimensions",
    "GridDimensionsWidget",
    "GridOrientation",
    "MatrixLod",
]

