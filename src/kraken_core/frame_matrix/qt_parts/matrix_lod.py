"""qt: matrix lod."""
from __future__ import annotations
from enum import StrEnum

class MatrixLod(StrEnum):
    """Rendering detail exposed for toolbars and diagnostics."""

    OVERVIEW = "overview"
    CELLS = "cells"
    DETAILS = "details"
