"""scene_renderer: draw port implementation."""
from __future__ import annotations
import re
from krona.infrastructure.edif_parser import EdifTextParser


class DrawPortImplementationMixin:
    def _draw_port_implementation(self, port_impl_block: str, color: tuple[int, int, int]) -> None:
        name_block = EdifTextParser.first_direct_or_none(port_impl_block, "(name ")
        if name_block:
            port_name = EdifTextParser.parse_header_name(name_block, "name")
        else:
            # Some EDIF variants encode the name in the header:
            # (portImplementation &0 ...)
            port_name = EdifTextParser.parse_header_name(port_impl_block, "portImplementation")
        upper_name = port_name.upper()
        nested_instance = EdifTextParser.first_direct_or_none(port_impl_block, "(instance ")
        nested_instance_name = (
            EdifTextParser.parse_header_name(nested_instance, "instance").upper() if nested_instance else ""
        )
        if re.fullmatch(r"OUT\d+", upper_name):
            kind = "output"
        elif re.fullmatch(r"(INSIN|IN)\d+", upper_name):
            kind = "input"
        elif upper_name in {"GND", "0", "&0"} or nested_instance_name.startswith("GNDGROUND"):
            kind = "ground"
        else:
            return

        world_x = None
        world_y = None
        connect_location = EdifTextParser.first_any_or_none(port_impl_block, "(connectLocation")
        if connect_location:
            dot_block = EdifTextParser.first_any_or_none(connect_location, "(dot")
            if dot_block:
                points = EdifTextParser.parse_points(dot_block)
                if points:
                    world_x, world_y = points[0].x, points[0].y
        if world_x is None or world_y is None:
            if nested_instance:
                transform_block = EdifTextParser.first_direct_or_none(nested_instance, "(transform")
                origin_block = EdifTextParser.first_any_or_none(transform_block, "(origin") if transform_block else None
                if origin_block:
                    points = EdifTextParser.parse_points(origin_block)
                    if points:
                        world_x, world_y = points[0].x, points[0].y
        if world_x is None or world_y is None:
            return
        if kind == "ground":
            label = None if upper_name in {"0", "&0"} else port_name
            self._draw_ground_glyph_at(world_x, world_y, color, label_text=label)
            return
        label_origin = None
        label_justify = "UPPERLEFT"
        display_block = EdifTextParser.first_any_or_none(name_block, "(display ") if name_block else None
        if display_block:
            origin_block = EdifTextParser.first_any_or_none(display_block, "(origin")
            if origin_block:
                pts = EdifTextParser.parse_points(origin_block)
                if pts:
                    label_origin = (pts[0].x, pts[0].y)
            justify_block = EdifTextParser.first_any_or_none(display_block, "(justify ")
            if justify_block:
                label_justify = EdifTextParser.parse_header_name(justify_block, "justify")
        self._draw_port_glyph_at(
            world_x,
            world_y,
            kind,
            color,
            port_name,
            label_origin=label_origin,
            label_justify=label_justify,
        )
