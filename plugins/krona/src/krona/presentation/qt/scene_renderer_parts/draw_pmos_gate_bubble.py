"""scene_renderer: draw pmos gate bubble."""
from __future__ import annotations
from PyQt6.QtGui import QBrush, QColor, QPen
from krona.infrastructure.edif_parser import EdifTextParser


class DrawPmosGateBubbleMixin:
    @staticmethod
    def _parse_gate_local_point_from_instance(instance_block: str) -> tuple[int, int] | None:
        for port_instance in EdifTextParser.find_direct_classes(instance_block, "(portInstance"):
            name_block = EdifTextParser.first_direct_or_none(port_instance, "(name ")
            if not name_block:
                continue
            pin_name = EdifTextParser.parse_header_name(name_block, "name").upper()
            if pin_name not in {"GATE", "G"}:
                continue
            display_block = EdifTextParser.first_any_or_none(name_block, "(display ")
            if not display_block:
                continue
            origin_block = EdifTextParser.first_any_or_none(display_block, "(origin")
            if not origin_block:
                continue
            points = EdifTextParser.parse_points(origin_block)
            if not points:
                continue
            return points[0].x, points[0].y
        return None

    @staticmethod
    def _parse_gate_local_point_from_symbol(symbol_block: str) -> tuple[int, int] | None:
        for port_impl in EdifTextParser.find_direct_classes(symbol_block, "(portImplementation"):
            name_block = EdifTextParser.first_direct_or_none(port_impl, "(name ")
            if not name_block:
                continue
            pin_name = EdifTextParser.parse_header_name(name_block, "name").upper()
            if pin_name not in {"GATE", "G"}:
                continue
            connect_location = EdifTextParser.first_any_or_none(port_impl, "(connectLocation")
            if not connect_location:
                continue
            dot_block = EdifTextParser.first_any_or_none(connect_location, "(dot")
            if dot_block:
                dot_points = EdifTextParser.parse_points(dot_block)
                if dot_points:
                    return dot_points[0].x, dot_points[0].y
            path_block = EdifTextParser.first_any_or_none(connect_location, "(path")
            if path_block:
                point_list = EdifTextParser.first_any_or_none(path_block, "(pointList")
                if point_list:
                    path_points = EdifTextParser.parse_points(point_list)
                    if path_points:
                        return path_points[0].x, path_points[0].y
        return None

    @staticmethod
    def _port_name_matches(kind: str, pin_name: str) -> bool:
        name = pin_name.upper()
        if kind == "input":
            return "VPULSECR" in name or name.startswith("IN")
        return name.startswith("OUT")

    def _parse_port_local_point_from_symbol(self, symbol_block: str, kind: str) -> tuple[int, int] | None:
        for port_impl in EdifTextParser.find_direct_classes(symbol_block, "(portImplementation"):
            name_block = EdifTextParser.first_direct_or_none(port_impl, "(name ")
            if not name_block:
                continue
            pin_name = EdifTextParser.parse_header_name(name_block, "name")
            if not self._port_name_matches(kind, pin_name):
                continue
            connect_location = EdifTextParser.first_any_or_none(port_impl, "(connectLocation")
            if connect_location:
                dot_block = EdifTextParser.first_any_or_none(connect_location, "(dot")
                if dot_block:
                    dot_points = EdifTextParser.parse_points(dot_block)
                    if dot_points:
                        return dot_points[0].x, dot_points[0].y
                path_block = EdifTextParser.first_any_or_none(connect_location, "(path")
                if path_block:
                    point_list = EdifTextParser.first_any_or_none(path_block, "(pointList")
                    if point_list:
                        path_points = EdifTextParser.parse_points(point_list)
                        if path_points:
                            return path_points[0].x, path_points[0].y
        return None

    def _draw_pmos_gate_bubble(
        self,
        instance_block: str,
        transform: Affine2D,
        symbol_block: str | None,
        color: tuple[int, int, int],
    ) -> None:
        if not self._is_pmos_instance(instance_block):
            return
        gate_point = None
        if symbol_block is not None:
            gate_point = self._parse_gate_local_point_from_symbol(symbol_block)
        if gate_point is None:
            gate_point = self._parse_gate_local_point_from_instance(instance_block)
        if not gate_point:
            return
        wx, wy = transform.apply(gate_point[0], gate_point[1])
        sx, sy = self._to_scene(wx, wy)

        # 10x smaller than previous radius (5.0 -> 0.5).
        bubble_radius_px = 0.5
        pen = QPen(QColor(*color, 240))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        brush = QBrush(QColor(255, 255, 255, 245))
        bubble = self._scene.addEllipse(
            sx - bubble_radius_px,
            sy - bubble_radius_px,
            2.0 * bubble_radius_px,
            2.0 * bubble_radius_px,
            pen,
            brush,
        )
        self._tag_item(bubble, EdifTextParser.parse_header_name(instance_block, "instance"))
        bubble.setZValue(1000)
        self._stats.primitive_count += 1
