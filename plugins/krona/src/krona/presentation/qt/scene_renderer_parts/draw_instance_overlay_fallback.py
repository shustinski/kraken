"""scene_renderer: draw instance overlay fallback."""
from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from krona.infrastructure.edif_parser import EdifTextParser


class DrawInstanceOverlayFallbackMixin:
    def _draw_instance_overlay_fallback(self, instance_block: str, transform: Affine2D) -> None:
        instance_name = EdifTextParser.parse_header_name(instance_block, "instance")
        pins = self._parse_instance_pin_points(instance_block)
        if not pins:
            # Fallback for instances without explicit portInstance display data.
            cx, cy = transform.apply(0, 0)
            w_world = 70.0
            h_world = 34.0
            left = cx - w_world / 2.0
            right = cx + w_world / 2.0
            top = cy - h_world / 2.0
            bottom = cy + h_world / 2.0
            sx1, sy1 = self._to_scene(left, top)
            sx2, sy2 = self._to_scene(right, bottom)
            x = min(sx1, sx2)
            y = min(sy1, sy2)
            w = abs(sx2 - sx1)
            h = abs(sy2 - sy1)

            body_pen = QPen(QColor(30, 41, 59, 210))
            body_pen.setWidthF(max(0.8, 0.7 * self._scale * 10.0))
            body_brush = QBrush(QColor(241, 245, 249, 90))
            rect_item = self._scene.addRect(x, y, w, h, body_pen, body_brush)
            self._tag_item(rect_item, instance_name)
            self._stats.primitive_count += 1
            self._draw_text(
                text=self._parse_instance_label(instance_block),
                world_x=cx,
                world_y=cy,
                justify="CENTERCENTER",
                color=(17, 24, 39),
                text_height=10,
            )
            return
        world = [(pin_name, *transform.apply(x, y)) for pin_name, x, y in pins]
        xs = [x for _, x, _ in world]
        ys = [y for _, _, y in world]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Create a realistic symbol body around pin cloud.
        pad = 22.0
        body_left = min_x - pad
        body_right = max_x + pad
        body_top = min_y - pad
        body_bottom = max_y + pad

        sx1, sy1 = self._to_scene(body_left, body_top)
        sx2, sy2 = self._to_scene(body_right, body_bottom)
        x = min(sx1, sx2)
        y = min(sy1, sy2)
        w = abs(sx2 - sx1)
        h = abs(sy2 - sy1)

        body_pen = QPen(QColor(30, 41, 59, 210))
        body_pen.setWidthF(max(0.8, 0.7 * self._scale * 10.0))
        body_brush = QBrush(QColor(241, 245, 249, 90))
        rect_item = self._scene.addRect(x, y, w, h, body_pen, body_brush)
        self._tag_item(rect_item, instance_name)
        self._stats.primitive_count += 1

        pin_pen = QPen(QColor(15, 23, 42, 240))
        pin_pen.setWidthF(max(0.8, 0.5 * self._scale * 10.0))
        pin_brush = QBrush(QColor(15, 23, 42, 220))
        for pin_name, wx, wy in world:
            px, py = self._to_scene(wx, wy)
            pin_item = self._scene.addEllipse(px - 2.0, py - 2.0, 4.0, 4.0, pin_pen, pin_brush)
            self._tag_item(pin_item, instance_name)
            self._stats.primitive_count += 1

            # Pin label near each connection point.
            font = QFont("Arial")
            font.setPointSizeF(6.8)
            text_item = self._scene.addSimpleText(pin_name)
            text_item.setFont(font)
            text_item.setBrush(QBrush(QColor(30, 41, 59, 220)))
            text_item.setPos(px + 3.0, py - 8.0)
            self._stats.text_count += 1

        label = self._parse_instance_label(instance_block)
        cx = (body_left + body_right) / 2.0
        cy = (body_top + body_bottom) / 2.0
        self._draw_text(
            text=label,
            world_x=cx,
            world_y=cy,
            justify="CENTERCENTER",
            color=(17, 24, 39),
            text_height=10,
        )

if TYPE_CHECKING:
    from krona.presentation.qt.scene_renderer_parts.affine2_d import Affine2D
