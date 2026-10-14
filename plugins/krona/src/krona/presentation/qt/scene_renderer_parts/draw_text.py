"""scene_renderer: draw text."""
from __future__ import annotations
import re
from PyQt6.QtGui import QBrush, QColor, QFont
from krona.infrastructure.edif_parser import EdifTextParser

from krona.presentation.qt.scene_renderer_parts.affine2_d import re_search_first_quoted


class DrawTextMixin:
    def _draw_text(
        self,
        text: str,
        world_x: float,
        world_y: float,
        justify: str,
        color: tuple[int, int, int],
        text_height: int,
        text_scale: float = 1.0,
    ) -> None:
        if not text.strip():
            return
        sx, sy = self._to_scene(world_x, world_y)
        item = self._scene.addSimpleText(text)
        font = QFont("Arial")
        scaled_height = text_height * 0.65 * text_scale
        if text_scale < 1.0:
            font.setPointSizeF(max(0.6, scaled_height))
        else:
            font.setPointSizeF(max(6.0, scaled_height))
        item.setFont(font)
        item.setBrush(QBrush(QColor(*color)))
        rect = item.boundingRect()
        dx = 0.0 if justify.endswith("LEFT") else (-rect.width() if justify.endswith("RIGHT") else -rect.width() / 2.0)
        if justify.startswith("UPPER"):
            dy = 0.0
        elif justify.startswith("LOWER"):
            dy = -rect.height()
        else:
            dy = -rect.height() / 2.0
        item.setPos(sx + dx, sy + dy)
        self._stats.text_count += 1

    @staticmethod
    def _parse_instance_label(instance_block: str) -> str:
        instance_name = EdifTextParser.parse_header_name(instance_block, "instance")
        designator_block = EdifTextParser.first_direct_or_none(instance_block, "(designator")
        if not designator_block:
            return instance_name
        match = re_search_first_quoted(designator_block)
        if not match:
            return instance_name
        return f"{instance_name} ({match})"

    @staticmethod
    def _parse_instance_pin_points(instance_block: str) -> list[tuple[str, int, int]]:
        points: list[tuple[str, int, int]] = []
        for port_instance in EdifTextParser.find_direct_classes(instance_block, "(portInstance"):
            name_block = EdifTextParser.first_direct_or_none(port_instance, "(name ")
            if not name_block:
                continue
            pin_name = EdifTextParser.parse_header_name(name_block, "name")
            display_block = EdifTextParser.first_any_or_none(name_block, "(display ")
            if not display_block:
                continue
            origin_block = EdifTextParser.first_any_or_none(display_block, "(origin")
            if not origin_block:
                continue
            pts = EdifTextParser.parse_points(origin_block)
            if not pts:
                continue
            points.append((pin_name, pts[0].x, pts[0].y))
        return points

    @staticmethod
    def _is_port_instance_for_visualization(instance_block: str) -> bool:
        instance_name = EdifTextParser.parse_header_name(instance_block, "instance").upper()
        return instance_name.startswith("INSIN") or instance_name.startswith("OUT")

    @staticmethod
    def _port_visualization_kind(instance_block: str) -> str | None:
        instance_name = EdifTextParser.parse_header_name(instance_block, "instance").upper()
        if re.fullmatch(r"INSIN\d+", instance_name):
            return "input"
        if re.fullmatch(r"OUT\d+", instance_name):
            return "output"
        return None

    @staticmethod
    def _is_hidden_instance(instance_block: str) -> bool:
        instance_name = EdifTextParser.parse_header_name(instance_block, "instance").upper()
        if re.fullmatch(r"INSIN\d+", instance_name) or re.fullmatch(r"OUT\d+", instance_name):
            return False
        view_ref = EdifTextParser.first_direct_or_none(instance_block, "(viewRef ")
        cell_ref = EdifTextParser.first_any_or_none(instance_block, "(cellRef ")
        view_name = EdifTextParser.parse_header_name(view_ref, "viewRef").upper() if view_ref else ""
        cell_name = EdifTextParser.parse_header_name(cell_ref, "cellRef").upper() if cell_ref else ""
        return "VPULSECR" in instance_name or "VPULSECR" in view_name or "VPULSECR" in cell_name
