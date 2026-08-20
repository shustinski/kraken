"""Pure minimap coordinate mapping (no Qt)."""

from __future__ import annotations

MINIMAP_MAX_LONG_SIDE = 180.0
MINIMAP_VIEWPORT_MARGIN_PX = 12.0

Point = tuple[float, float]
Size = tuple[float, float]
Rect = tuple[float, float, float, float]


def has_usable_image_size(width: float, height: float) -> bool:
    return float(width) > 1.0 and float(height) > 1.0


def fitted_minimap_size(
    image_width: float,
    image_height: float,
    max_long_side: float = MINIMAP_MAX_LONG_SIDE,
) -> Size:
    """Scale the image so the long side equals ``max_long_side``."""
    width = max(0.0, float(image_width))
    height = max(0.0, float(image_height))
    if width <= 0.0 or height <= 0.0 or max_long_side <= 0.0:
        return (0.0, 0.0)
    scale = float(max_long_side) / max(width, height)
    return (width * scale, height * scale)


def image_rect_in_minimap(image_size: Size, minimap_size: Size) -> Rect:
    """Fit the image into the minimap widget, centered (letterbox if needed)."""
    image_width, image_height = image_size
    minimap_width, minimap_height = minimap_size
    if image_width <= 0.0 or image_height <= 0.0 or minimap_width <= 0.0 or minimap_height <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    scale = min(minimap_width / image_width, minimap_height / image_height)
    drawn_width = image_width * scale
    drawn_height = image_height * scale
    return (
        (minimap_width - drawn_width) / 2.0,
        (minimap_height - drawn_height) / 2.0,
        drawn_width,
        drawn_height,
    )


def intersect_rects(first: Rect, second: Rect) -> Rect | None:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return None
    return (left, top, right - left, bottom - top)


def map_point_from_rect(point: Point, source: Rect, destination: Rect) -> Point:
    source_x, source_y, source_width, source_height = source
    dest_x, dest_y, dest_width, dest_height = destination
    if source_width == 0.0 or source_height == 0.0:
        return (dest_x, dest_y)
    normalized_x = (point[0] - source_x) / source_width
    normalized_y = (point[1] - source_y) / source_height
    return (dest_x + normalized_x * dest_width, dest_y + normalized_y * dest_height)


def map_rect_from_rect(rect: Rect, source: Rect, destination: Rect) -> Rect:
    x1, y1 = map_point_from_rect((rect[0], rect[1]), source, destination)
    x2, y2 = map_point_from_rect((rect[0] + rect[2], rect[1] + rect[3]), source, destination)
    left = min(x1, x2)
    top = min(y1, y2)
    return (left, top, abs(x2 - x1), abs(y2 - y1))


def viewport_frame_in_minimap(
    image_scene_rect: Rect,
    viewport_scene_rect: Rect,
    minimap_image_rect: Rect,
) -> Rect | None:
    """Map viewport ∩ image into minimap image coordinates."""
    visible = intersect_rects(image_scene_rect, viewport_scene_rect)
    if visible is None:
        return None
    return map_rect_from_rect(visible, image_scene_rect, minimap_image_rect)


def minimap_point_to_scene(
    minimap_point: Point,
    image_scene_rect: Rect,
    minimap_image_rect: Rect,
) -> Point:
    """Map a point on the minimap image into scene/image coordinates."""
    return map_point_from_rect(minimap_point, minimap_image_rect, image_scene_rect)
