"""scene_renderer: draw instance hitbox."""
from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtCore import Qt
from krona.infrastructure.edif_parser import EdifTextParser


class DrawInstanceHitboxMixin:
    def _draw_instance_hitbox(self, instance_block: str, transform: Affine2D, draw_block: str | None = None) -> None:
        instance_name = EdifTextParser.parse_header_name(instance_block, "instance")
        kind = self._port_visualization_kind(instance_block)
        if kind in {"input", "output"}:
            local_connection = None
            if draw_block is not None:
                local_connection = self._parse_port_local_point_from_symbol(draw_block, kind)
            if local_connection is None:
                pin_points = self._parse_instance_pin_points(instance_block)
                if pin_points:
                    selected_pin = next((p for p in pin_points if self._port_name_matches(kind, p[0])), pin_points[0])
                    local_connection = (selected_pin[1], selected_pin[2])
            if local_connection is None:
                local_connection = (0, 0)
            base_local_x = local_connection[0] - 8.0 if kind == "input" else local_connection[0] + 8.0
            base_local_y = local_connection[1]
            if kind == "input":
                local_shape = [(-16, -7), (-2, -7), (8, 0), (-2, 7), (-16, 7), (-8, 0), (-16, -7)]
            else:
                local_shape = [(16, -7), (2, -7), (-8, 0), (2, 7), (16, 7), (8, 0), (16, -7)]
            world_shape = [transform.apply(base_local_x + dx, base_local_y + dy) for dx, dy in local_shape]
            scene_shape = [self._to_scene(wx, wy) for wx, wy in world_shape]
            min_x = min(x for x, _ in scene_shape)
            max_x = max(x for x, _ in scene_shape)
            min_y = min(y for _, y in scene_shape)
            max_y = max(y for _, y in scene_shape)
            width = max(2.0, max_x - min_x)
            height = max(2.0, max_y - min_y)
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            pad = 0.8
            item = self._scene.addRect(
                cx - width / 2.0 - pad,
                cy - height / 2.0 - pad,
                width + 2.0 * pad,
                height + 2.0 * pad,
                QPen(QColor(0, 255, 180, 210)) if self._show_hitboxes else QPen(Qt.PenStyle.NoPen),
                QBrush(Qt.GlobalColor.transparent),
            )
            if self._show_hitboxes:
                pen = item.pen()
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidthF(1.0)
                pen.setCosmetic(True)
                item.setPen(pen)
            item.setData(1, instance_name)
            item.setZValue(6000)
            return

        world_points: list[tuple[float, float]] = []
        if draw_block:
            for point in self._collect_drawable_points(draw_block):
                world_points.append(transform.apply(point.x, point.y))
        if not world_points:
            for _, px, py in self._parse_instance_pin_points(instance_block):
                world_points.append(transform.apply(px, py))
        if not world_points:
            wx, wy = transform.apply(0, 0)
            world_points.append((wx, wy))

        scene_points = [self._to_scene(wx, wy) for wx, wy in world_points]
        min_x = min(x for x, _ in scene_points)
        max_x = max(x for x, _ in scene_points)
        min_y = min(y for _, y in scene_points)
        max_y = max(y for _, y in scene_points)

        min_size = 4.0
        width = max(min_size, max_x - min_x)
        height = max(min_size, max_y - min_y)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        pad = 1.0
        item = self._scene.addRect(
            cx - width / 2.0 - pad,
            cy - height / 2.0 - pad,
            width + 2.0 * pad,
            height + 2.0 * pad,
            QPen(QColor(0, 255, 180, 210)) if self._show_hitboxes else QPen(Qt.PenStyle.NoPen),
            QBrush(Qt.GlobalColor.transparent),
        )
        if self._show_hitboxes:
            pen = item.pen()
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidthF(1.0)
            pen.setCosmetic(True)
            item.setPen(pen)
        item.setData(1, instance_name)
        item.setZValue(6000)

if TYPE_CHECKING:
    from krona.presentation.qt.scene_renderer_parts.affine2_d import Affine2D
