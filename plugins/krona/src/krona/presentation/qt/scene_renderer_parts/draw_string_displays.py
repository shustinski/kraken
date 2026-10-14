"""scene_renderer: draw string displays."""
from __future__ import annotations
from krona.infrastructure.edif_parser import EdifTextParser


class DrawStringDisplaysMixin:
    def _draw_string_displays(
        self,
        block: str,
        transform: Affine2D,
        library_name: str | None,
        text_scale: float = 1.0,
    ) -> None:
        for s_block in EdifTextParser.find_classes(block, "(stringDisplay "):
            parts = s_block.split('"')
            if len(parts) < 3:
                continue
            value = parts[1]
            display = EdifTextParser.first_any_or_none(s_block, "(display ")
            if not display:
                continue
            visible = EdifTextParser.first_any_or_none(display, "(visible")
            if visible and "(false)" in visible:
                continue
            origin = EdifTextParser.first_any_or_none(display, "(origin")
            if not origin:
                continue
            points = EdifTextParser.parse_points(origin)
            if not points:
                continue
            x, y = transform.apply(points[0].x, points[0].y)
            justify_block = EdifTextParser.first_any_or_none(display, "(justify ")
            justify = EdifTextParser.parse_header_name(justify_block, "justify") if justify_block else "UPPERLEFT"
            style = {"color": (20, 20, 20), "text_height": 9}
            fg_override = EdifTextParser.first_any_or_none(display, "(figureGroupOverride ")
            if fg_override:
                fg_name = EdifTextParser.parse_header_name(fg_override, "figureGroupOverride")
                if library_name and (library_name, fg_name) in self._fig_styles:
                    style.update(self._fig_styles[(library_name, fg_name)])
                style.update(self._style_from_block(fg_override))
            else:
                display_group = EdifTextParser.parse_header_name(display, "display")
                if library_name and (library_name, display_group) in self._fig_styles:
                    style.update(self._fig_styles[(library_name, display_group)])
            self._draw_text(value, x, y, justify, style["color"], style["text_height"], text_scale=text_scale)
