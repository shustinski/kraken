"""bright_tophat_dog: module state."""
from __future__ import annotations

# BGR for overlay: green = accepted, yellow = soft, red = hard
_COLOR_ACCEPT = (0, 220, 0)

_COLOR_SOFT = (0, 220, 255)

_COLOR_HARD = (0, 0, 255)

_MATCHED_PCT_OFFSET = 0.5

_RADIAL_SMOOTH_SIGMA = 1.0
