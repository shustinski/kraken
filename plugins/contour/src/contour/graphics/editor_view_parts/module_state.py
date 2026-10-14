"""editor_view: module state."""
from __future__ import annotations
from ...infrastructure.runtime_config import config_bool, config_float, config_int
from contour.graphics.tools import EditorTool

_WHEEL_ZOOM_COALESCE_MS = config_int("editor", "wheel_zoom_coalesce_ms", 3, minimum=0)

_ZOOM_ANIMATION_FRAME_MS = config_int("editor", "zoom_animation_frame_ms", 16, minimum=1)

_ZOOM_EASING_FRACTION = min(1.0, config_float("editor", "zoom_easing_fraction", 0.55, minimum=0.01))

_ZOOM_SETTLE_RATIO = config_float("editor", "zoom_settle_ratio", 0.001, minimum=0.000001)

_ZOOM_VECTOR_BATCH_THRESHOLD = config_int("editor", "zoom_vector_batch_threshold", 1000, minimum=1)

_MINIMAP_DRAWING_TOOLS = frozenset(
    {
        EditorTool.ADD_POLYGON,
        EditorTool.BRUSH,
        EditorTool.TRACE_PEN,
    }
)

_OPENGL_VIEWPORT_ENABLED = config_bool("editor", "opengl_viewport_enabled", True)

_OPENGL_DISABLED_PLATFORMS = {"offscreen", "minimal"}

_PYRAMID_VISIBLE_UPDATE_MS = config_int("editor", "pyramid_visible_update_ms", 24, minimum=1)

_PYRAMID_CACHE_LIMIT = config_int("editor", "pyramid_cache_limit", 192, minimum=1)
