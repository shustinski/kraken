"""scene_renderer: parse color."""
from __future__ import annotations
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QGraphicsScene
from krona.application.ports import SceneData
from krona.infrastructure.edif_parser import EdifTextParser

from krona.presentation.qt.scene_renderer_parts.affine2_d import (
    Affine2D,
    RenderStats,
)


class ParseColorMixin:
    def __init__(self, scene: QGraphicsScene, scale: float = 0.10):
        self._scene = scene
        self._scale = scale
        self._fig_styles: dict[tuple[str, str], dict] = {}
        self._stats = RenderStats()
        self._highlight_nets: set[str] = set()
        self._highlight_instances: set[str] = set()
        self._show_hitboxes: bool = False
        self._dark_mode: bool = True

    def render(self, scene_data: SceneData) -> RenderStats:
        self._scene.clear()
        self._stats = RenderStats()
        self._build_figure_group_style_index(scene_data.source_text)
        self._draw_top_page(scene_data)
        self._draw_logic_highlight_overlay(scene_data)
        return self._stats

    def set_logic_highlight(self, nets: set[str] | list[str] | tuple[str, ...], instances: set[str] | list[str] | tuple[str, ...]) -> None:
        self._highlight_nets = set(nets)
        self._highlight_instances = set(instances)

    def set_show_hitboxes(self, enabled: bool) -> None:
        self._show_hitboxes = bool(enabled)

    def set_dark_mode(self, enabled: bool) -> None:
        self._dark_mode = bool(enabled)

    def _to_scene(self, x: float, y: float) -> tuple[float, float]:
        return x * self._scale, -y * self._scale

    @staticmethod
    def _tag_item(item, instance_name: str | None) -> None:
        if item is None or not instance_name:
            return
        item.setData(0, instance_name)

    @staticmethod
    def _luma(color: tuple[int, int, int]) -> float:
        r, g, b = color
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _visible_port_color(self, kind: str, base_color: tuple[int, int, int]) -> tuple[int, int, int]:
        if self._luma(base_color) >= 72:
            return base_color
        if kind == "input":
            return 122, 189, 255
        if kind == "output":
            return 255, 138, 138
        if kind == "ground":
            return 188, 196, 210
        return 210, 220, 235

    def _build_figure_group_style_index(self, source_text: str) -> None:
        self._fig_styles = {}
        for lib_block in EdifTextParser.find_direct_classes(source_text, "(library "):
            lib_name = EdifTextParser.parse_header_name(lib_block, "library")
            tech = EdifTextParser.first_direct_or_none(lib_block, "(technology")
            if not tech:
                continue
            for fg_block in EdifTextParser.find_direct_classes(tech, "(figureGroup "):
                fg_name = EdifTextParser.parse_header_name(fg_block, "figureGroup")
                self._fig_styles[(lib_name, fg_name)] = self._style_from_block(fg_block)

    @staticmethod
    def _parse_color(block: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        sub = EdifTextParser.first_any_or_none(block, "(color ")
        if not sub:
            return default
        tokens = sub.replace("(", " ").replace(")", " ").split()
        if len(tokens) < 4:
            return default
        try:
            raw_r, raw_g, raw_b = int(tokens[1]), int(tokens[2]), int(tokens[3])
        except ValueError:
            return default
        return (
            max(0, min(255, int(round(raw_r * 2.55)))),
            max(0, min(255, int(round(raw_g * 2.55)))),
            max(0, min(255, int(round(raw_b * 2.55)))),
        )

    @staticmethod
    def _parse_int(block: str, keyword: str, default: int) -> int:
        sub = EdifTextParser.first_any_or_none(block, f"({keyword} ")
        if not sub:
            return default
        tokens = sub.replace("(", " ").replace(")", " ").split()
        if len(tokens) < 2:
            return default
        try:
            return int(tokens[1])
        except ValueError:
            return default

    def _style_from_block(self, block: str) -> dict:
        return {
            "color": self._parse_color(block, (20, 20, 20)),
            "path_width": self._parse_int(block, "pathWidth", 0),
            "text_height": self._parse_int(block, "textHeight", 9),
        }

    @staticmethod
    def _orientation_affine(orientation: str, tx: int, ty: int) -> Affine2D:
        matrix = {
            "R0": (1, 0, 0, 1),
            "R90": (0, -1, 1, 0),
            "R180": (-1, 0, 0, -1),
            "R270": (0, 1, -1, 0),
            "MX": (1, 0, 0, -1),
            "MY": (-1, 0, 0, 1),
            "MXR90": (0, 1, 1, 0),
            "MYR90": (0, -1, -1, 0),
        }.get(orientation, (1, 0, 0, 1))
        return Affine2D(*matrix, tx=tx, ty=ty)

    @staticmethod
    def _transform_from_block(block: str) -> Affine2D:
        from krona.presentation.qt.scene_renderer import EdifSceneRenderer

        transform_block = EdifTextParser.first_direct_or_none(block, "(transform")
        if not transform_block:
            return Affine2D()
        orient = EdifTextParser.first_any_or_none(transform_block, "(orientation")
        orientation = EdifTextParser.parse_header_name(orient, "orientation") if orient else "R0"
        origin = EdifTextParser.first_any_or_none(transform_block, "(origin")
        tx, ty = 0, 0
        if origin:
            points = EdifTextParser.parse_points(origin)
            if points:
                tx, ty = points[0].x, points[0].y
        return EdifSceneRenderer._orientation_affine(orientation, tx, ty)

    @staticmethod
    def _resolve_view_draw_block(view_block: str) -> str | None:
        symbol_any = EdifTextParser.first_any_or_none(view_block, "(symbol")
        if symbol_any:
            return symbol_any
        contents = EdifTextParser.first_direct_or_none(view_block, "(contents")
        if not contents:
            return None
        pages = EdifTextParser.find_direct_classes(contents, "(page ")
        return pages[0] if pages else contents

    @staticmethod
    def _figure_group_name(figure_block: str) -> str | None:
        stripped = figure_block.strip()
        if not stripped.startswith("(figure"):
            return None
        tail = stripped[len("(figure"):].lstrip()
        if not tail or tail[0] in "(\n\r\t":
            return None
        token = []
        for char in tail:
            if char.isspace() or char == ")":
                break
            token.append(char)
        return "".join(token) if token else None

    def _resolve_style(self, figure_block: str, library_name: str | None) -> dict:
        style = {"color": (20, 20, 20), "path_width": 0, "text_height": 9}
        fg_name = self._figure_group_name(figure_block)
        if library_name and fg_name and (library_name, fg_name) in self._fig_styles:
            style.update(self._fig_styles[(library_name, fg_name)])
        override = EdifTextParser.first_direct_or_none(figure_block, "(figureGroupOverride ")
        if override:
            ov_name = EdifTextParser.parse_header_name(override, "figureGroupOverride")
            if library_name and (library_name, ov_name) in self._fig_styles:
                style.update(self._fig_styles[(library_name, ov_name)])
            style.update(self._style_from_block(override))
        return style

    def _resolve_instance_primary_color(self, draw_block: str, library_name: str | None) -> tuple[int, int, int]:
        for child in EdifTextParser.direct_children(draw_block):
            if child.lstrip().startswith("(figure "):
                return self._resolve_style(child, library_name)["color"]
        return 20, 20, 20

    def _pen_from_style(self, style: dict) -> QPen:
        pen = QPen(QColor(*style["color"]))
        width = style["path_width"] * self._scale
        if width <= 0:
            # EDIF pathWidth 0 is a hairline stroke.
            pen.setWidthF(1.0)
            pen.setCosmetic(True)
        else:
            pen.setWidthF(width)
            pen.setCosmetic(False)
        return pen
