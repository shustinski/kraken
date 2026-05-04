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


def polygon_overlay_visibility_after_space_toggle(
    currently_hidden_via_space_toggle: bool,
) -> tuple[bool, bool]:
    """Return ``(new_hidden_flag, overlays_visible)`` after one Space press.

    Does not describe polygon data or selection — only visibility flags.
    """
    new_hidden = not currently_hidden_via_space_toggle
    return (new_hidden, not new_hidden)
