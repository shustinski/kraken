"""scene_renderer: draw port glyph at."""
from __future__ import annotations
from typing import TYPE_CHECKING
import re
from PyQt6.QtGui import QColor, QPen
from krona.infrastructure.edif_parser import EdifTextParser


class DrawPortGlyphAtMixin:
    def _draw_port_instances(
        self,
        instance_block: str,
        transform: Affine2D,
        color: tuple[int, int, int],
        symbol_block: str | None = None,
    ) -> None:
        kind = self._port_visualization_kind(instance_block)
        if kind is None:
            return

        instance_name = EdifTextParser.parse_header_name(instance_block, "instance")
        local_connection = None
        if symbol_block is not None:
            local_connection = self._parse_port_local_point_from_symbol(symbol_block, kind)
        if local_connection is None:
            pin_points = self._parse_instance_pin_points(instance_block)
            if pin_points:
                selected_pin = next((p for p in pin_points if self._port_name_matches(kind, p[0])), pin_points[0])
                local_connection = (selected_pin[1], selected_pin[2])

        if local_connection is None:
            local_connection = (0, 0)

        if kind == "input":
            base_local_x = local_connection[0] - 8.0
        else:
            base_local_x = local_connection[0] + 8.0
        base_local_y = local_connection[1]
        cx, cy = transform.apply(base_local_x, base_local_y)
        conn_world_x, conn_world_y = transform.apply(local_connection[0], local_connection[1])
        draw_color = self._visible_port_color(kind, color)
        pen = QPen(QColor(*draw_color, 230))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        # Draw classic schematic port glyph.
        if kind == "input":
            local_shape = [(-16, -7), (-2, -7), (8, 0), (-2, 7), (-16, 7), (-8, 0), (-16, -7)]
        else:
            local_shape = [(16, -7), (2, -7), (-8, 0), (2, 7), (16, 7), (8, 0), (16, -7)]

        world_shape = [transform.apply(base_local_x + dx, base_local_y + dy) for dx, dy in local_shape]
        for start, end in zip(world_shape, world_shape[1:]):
            x1, y1 = self._to_scene(start[0], start[1])
            x2, y2 = self._to_scene(end[0], end[1])
            line = self._scene.addLine(x1, y1, x2, y2, pen)
            line.setZValue(920)
            self._tag_item(line, instance_name)
            self._stats.primitive_count += 1

        # Output/input port glyphs are drawn without extra junction markers.
        if kind == "output":
            # Reinforce visible connection where wire meets output tip.
            sx1, sy1 = self._to_scene(conn_world_x, conn_world_y)
            sx2, sy2 = self._to_scene(conn_world_x + 2.0, conn_world_y)
            stub = self._scene.addLine(sx1, sy1, sx2, sy2, pen)
            stub.setZValue(925)
            self._tag_item(stub, instance_name)
            self._stats.primitive_count += 1

        # Keep explicit label on instance-based ports so INSINx/OUTx names stay visible.
        gap_world = 4.0 / max(self._scale, 1e-6)
        if kind == "input":
            self._draw_text(
                text=instance_name,
                world_x=conn_world_x - gap_world,
                world_y=conn_world_y,
                justify="CENTERRIGHT",
                color=draw_color,
                text_height=24,
                text_scale=0.1,
            )
        else:
            self._draw_text(
                text=instance_name,
                world_x=conn_world_x + gap_world,
                world_y=conn_world_y,
                justify="CENTERLEFT",
                color=draw_color,
                text_height=24,
                text_scale=0.1,
            )

    def _draw_port_glyph_at(
        self,
        world_x: float,
        world_y: float,
        kind: str,
        color: tuple[int, int, int],
        label_text: str,
        label_origin: tuple[float, float] | None = None,
        label_justify: str = "UPPERLEFT",
    ) -> None:
        draw_color = self._visible_port_color(kind, color)
        pen = QPen(QColor(*draw_color, 230))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        if kind == "input":
            shape = [(-16, -7), (-2, -7), (8, 0), (-2, 7), (-16, 7), (-8, 0), (-16, -7)]
            label_dx = -36.0
            base_x = world_x - 8.0
        else:
            shape = [(16, -7), (2, -7), (-8, 0), (2, 7), (16, 7), (8, 0), (16, -7)]
            label_dx = 10.0
            base_x = world_x + 8.0
        world_shape = [(base_x + dx, world_y + dy) for dx, dy in shape]
        for start, end in zip(world_shape, world_shape[1:]):
            x1, y1 = self._to_scene(start[0], start[1])
            x2, y2 = self._to_scene(end[0], end[1])
            item = self._scene.addLine(x1, y1, x2, y2, pen)
            item.setZValue(920)
            self._stats.primitive_count += 1
        # Output/input port glyphs are drawn without extra junction markers.
        if kind == "output":
            # Reinforce visible connection where wire meets output tip.
            sx1, sy1 = self._to_scene(world_x, world_y)
            sx2, sy2 = self._to_scene(world_x + 2.0, world_y)
            stub = self._scene.addLine(sx1, sy1, sx2, sy2, pen)
            stub.setZValue(925)
            self._stats.primitive_count += 1

        # For OUTx/INSINx/INx top-level ports, anchor text to the connection point.
        if re.fullmatch(r"(OUT|INSIN|IN)\d+", label_text.upper()):
            gap_world = 4.0 / max(self._scale, 1e-6)
            if kind == "input":
                self._draw_text(
                    text=label_text,
                    world_x=world_x - gap_world,
                    world_y=world_y,
                    justify="CENTERRIGHT",
                    color=draw_color,
                    text_height=24,
                    text_scale=0.1,
                )
            else:
                self._draw_text(
                    text=label_text,
                    world_x=world_x + gap_world,
                    world_y=world_y,
                    justify="CENTERLEFT",
                    color=draw_color,
                    text_height=24,
                    text_scale=0.1,
                )
            return

        if label_origin is None:
            label_world_x = world_x + (label_dx / max(self._scale, 1e-6))
            label_world_y = world_y + (14.0 / max(self._scale, 1e-6))
            self._draw_text(
                text=label_text,
                world_x=label_world_x,
                world_y=label_world_y,
                justify="UPPERLEFT",
                color=draw_color,
                text_height=24,
                text_scale=0.1,
            )
        else:
            self._draw_text(
                text=label_text,
                world_x=label_origin[0],
                world_y=label_origin[1],
                justify=label_justify,
                color=draw_color,
                text_height=24,
                text_scale=0.1,
            )

if TYPE_CHECKING:
    from krona.presentation.qt.scene_renderer_parts.affine2_d import Affine2D
