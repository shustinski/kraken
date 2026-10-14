"""pipeline_mixin: color selection entry."""
from __future__ import annotations
from typing import TypedDict

class ColorSelectionEntry(TypedDict):
    rgb: list[int]
    enabled: bool
