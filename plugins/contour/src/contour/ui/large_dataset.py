"""Thresholds for Contour UI behavior with very large frame sets."""

from __future__ import annotations

# Above this count, avoid QWidget-per-row side lists. The frame matrix still
# contains every loaded base image; only thumbnail decoding is lazy.
LARGE_FRAME_COUNT_THRESHOLD = 750

# UI thread: max thumbnail icons applied per timer tick.
THUMBNAIL_ICONS_APPLY_PER_TICK = 8

# Main-thread icon apply interval (ms).
THUMBNAIL_APPLY_INTERVAL_MS = 16

# Debounce visible thumbnail queueing while scrolling (ms).
THUMBNAIL_VISIBLE_LOAD_DEBOUNCE_MS = 120

# Debounce matrix repaint after scroll settles (ms).
THUMBNAIL_SCROLL_SETTLE_MS = 150

# Max worker + queued thumbnail decodes before deferring more queueing.
THUMBNAIL_MAX_ACTIVE_DECODES = 6

# Background decode: paths queued per radial-fill pump tick.
THUMBNAIL_RADIAL_LOADS_PER_PUMP = 2

# Interval between radial-fill pump ticks (ms).
THUMBNAIL_RADIAL_PUMP_INTERVAL_MS = 24

# Max decoded thumbnail size (long edge scaled down to fit this box).
THUMBNAIL_MAX_SOURCE_WIDTH = 512
THUMBNAIL_MAX_SOURCE_HEIGHT = 512

# Asset filter side lists (image+vector / image-only / vector-only) are skipped until below threshold.
ASSET_FILTER_LISTS_MAX_FRAMES = LARGE_FRAME_COUNT_THRESHOLD


def clamp_thumbnail_source_size(width: int, height: int) -> tuple[int, int]:
    """Fit thumbnail decode size inside THUMBNAIL_MAX_SOURCE_* while preserving aspect ratio."""

    source_w = max(1, int(width))
    source_h = max(1, int(height))
    if source_w <= THUMBNAIL_MAX_SOURCE_WIDTH and source_h <= THUMBNAIL_MAX_SOURCE_HEIGHT:
        return source_w, source_h
    scale = min(
        THUMBNAIL_MAX_SOURCE_WIDTH / float(source_w),
        THUMBNAIL_MAX_SOURCE_HEIGHT / float(source_h),
    )
    return max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale)))
