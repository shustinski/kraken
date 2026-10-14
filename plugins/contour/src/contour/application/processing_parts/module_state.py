"""processing: module state."""
from __future__ import annotations

VIA_SIZE_MODE_RANGE = "range"

VIA_SIZE_MODE_FIXED = "fixed"

VIA_SEARCH_MODE_HYBRID = "hybrid"

VIA_SEARCH_MODE_BLOB = "blob"

VIA_SEARCH_MODE_TEMPLATE = "template"

VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG = "bright_tophat_dog"

VIA_SEARCH_MODE_HEURISTIC = "heuristic"

VIA_CHANNEL_MODE_COLUMNS = "columns"

VIA_CHANNEL_MODE_GRAYSCALE = "grayscale"

VIA_CHANNEL_MODE_RED_BLUE = "red_blue"

ALGORITHM_BACKEND_LEGACY = "legacy"

ALGORITHM_BACKEND_SEM = "sem"

VIA_DISPLAY_MODE_RECTANGLE = "rectangle"

VIA_DISPLAY_MODE_CIRCLE = "circle"

RECOGNITION_MODE_DISABLED = "disabled"

RECOGNITION_MODE_VIA = "via"

RECOGNITION_MODE_METAL = "metal"  # legacy value; normalizes to conductors

RECOGNITION_MODE_CONDUCTORS = "conductors"
