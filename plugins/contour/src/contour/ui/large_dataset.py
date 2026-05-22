"""Thresholds for Contour UI behavior with very large frame sets."""

from __future__ import annotations

# Above this count, avoid QWidget-per-row lists and full thumbnail matrices.
LARGE_FRAME_COUNT_THRESHOLD = 750

# Thumbnail matrix shows only a sliding window of frames around the current index.
THUMBNAIL_SPARSE_WINDOW_RADIUS = 120

# UI thread: max thumbnail icons applied per event-loop tick.
THUMBNAIL_ICONS_APPLY_PER_TICK = 2

# Background decode: paths queued per radial-fill pump tick.
THUMBNAIL_RADIAL_LOADS_PER_PUMP = 1

# Interval between radial-fill pump ticks (ms).
THUMBNAIL_RADIAL_PUMP_INTERVAL_MS = 45

# Asset filter side lists (image+vector / image-only / vector-only) are skipped until below threshold.
ASSET_FILTER_LISTS_MAX_FRAMES = LARGE_FRAME_COUNT_THRESHOLD
