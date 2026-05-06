"""Pure helpers for QGraphicsView-style pan/zoom (unit-testable).

``QGraphicsView.scale`` does not combine cleanly with ``translate`` for zoom-to-cursor
when the view already has a non-identity transform; scrollbar correction derived from
``mapFromScene`` keeps the picked scene point under the same viewport pixel.
"""

from __future__ import annotations


def viewport_scroll_correction_after_scale_reanchor(
    viewport_pixel_xy: tuple[int, int],
    scene_anchor_viewport_xy_after_scale: tuple[int, int],
) -> tuple[int, int]:
    """Scrollbar deltas so a fixed scene anchor stays under the same viewport pixel.

    After uniform ``scale`` about the graphics view origin, ``mapFromScene(anchor)``
    moves on the viewport; add ``(mapped - viewport_pixel)`` to scroll bars.

    Matches ``QGraphicsView.horizontalScrollBar`` / ``verticalScrollBar`` adjustment.
    """
    px, py = viewport_pixel_xy
    ax, ay = scene_anchor_viewport_xy_after_scale
    return (ax - px, ay - py)


def scroll_values_after_viewport_drag(
    scroll_x: float,
    scroll_y: float,
    viewport_delta_x: float,
    viewport_delta_y: float,
) -> tuple[float, float]:
    """Hand-drag semantics: scrollbar moves opposite to viewport mouse delta."""
    return (scroll_x - viewport_delta_x, scroll_y - viewport_delta_y)


def image_coordinate_under_cursor(
    cursor_viewport_xy: tuple[float, float],
    *,
    viewport_size: tuple[float, float],
    image_size: tuple[float, float],
    scale: float,
    pan_offset_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Image coordinate under a viewport cursor for centered image rendering."""
    vx, vy = viewport_size
    iw, ih = image_size
    scale = max(1e-12, float(scale))
    rendered_w = float(iw) * scale
    rendered_h = float(ih) * scale
    center_x = max(0.0, (float(vx) - rendered_w) / 2.0)
    center_y = max(0.0, (float(vy) - rendered_h) / 2.0)
    px, py = pan_offset_xy
    cx, cy = cursor_viewport_xy
    return ((float(cx) - center_x - float(px)) / scale, (float(cy) - center_y - float(py)) / scale)


def pan_offset_after_zoom_to_cursor(
    cursor_viewport_xy: tuple[float, float],
    *,
    viewport_size: tuple[float, float],
    image_size: tuple[float, float],
    old_scale: float,
    new_scale: float,
    old_pan_offset_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Pan offset preserving the image coordinate under the cursor after zoom."""
    anchor_x, anchor_y = image_coordinate_under_cursor(
        cursor_viewport_xy,
        viewport_size=viewport_size,
        image_size=image_size,
        scale=old_scale,
        pan_offset_xy=old_pan_offset_xy,
    )
    vx, vy = viewport_size
    iw, ih = image_size
    rendered_w = float(iw) * float(new_scale)
    rendered_h = float(ih) * float(new_scale)
    center_x = max(0.0, (float(vx) - rendered_w) / 2.0)
    center_y = max(0.0, (float(vy) - rendered_h) / 2.0)
    cx, cy = cursor_viewport_xy
    return (
        float(cx) - center_x - anchor_x * float(new_scale),
        float(cy) - center_y - anchor_y * float(new_scale),
    )


def polygon_overlay_visibility_after_space_toggle(
    currently_hidden_via_space_toggle: bool,
) -> tuple[bool, bool]:
    """Return ``(new_hidden_flag, overlays_visible)`` after one Space press.

    Does not describe polygon data or selection — only visibility flags.
    """
    new_hidden = not currently_hidden_via_space_toggle
    return (new_hidden, not new_hidden)
