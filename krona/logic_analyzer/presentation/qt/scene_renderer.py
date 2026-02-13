from __future__ import annotations

from dataclasses import dataclass
import math
import re

from PyQt6.QtGui import QBrush, QColor, QFont, QPen
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsScene

from logic_analyzer.application.ports import SceneData
from logic_analyzer.infrastructure.edif_parser import EdifTextParser


@dataclass(frozen=True)
class Affine2D:
    a: int = 1
    b: int = 0
    c: int = 0
    d: int = 1
    tx: int = 0
    ty: int = 0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty

    def compose(self, child: "Affine2D") -> "Affine2D":
        return Affine2D(
            a=self.a * child.a + self.b * child.c,
            b=self.a * child.b + self.b * child.d,
            c=self.c * child.a + self.d * child.c,
            d=self.c * child.b + self.d * child.d,
            tx=self.a * child.tx + self.b * child.ty + self.tx,
            ty=self.c * child.tx + self.d * child.ty + self.ty,
        )


@dataclass
class RenderStats:
    primitive_count: int = 0
    wire_segment_count: int = 0
    text_count: int = 0


class EdifSceneRenderer:
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

    def _draw_polyline(self, points: list[tuple[float, float]], pen: QPen, instance_name: str | None = None) -> None:
        if len(points) < 2:
            if points:
                x, y = self._to_scene(points[0][0], points[0][1])
                item = self._scene.addEllipse(x - 1.5, y - 1.5, 3.0, 3.0, pen, QBrush(pen.color()))
                self._tag_item(item, instance_name)
                self._stats.wire_segment_count += 1
            return
        for start, end in zip(points, points[1:]):
            x1, y1 = self._to_scene(start[0], start[1])
            x2, y2 = self._to_scene(end[0], end[1])
            item = self._scene.addLine(x1, y1, x2, y2, pen)
            self._tag_item(item, instance_name)
            self._stats.wire_segment_count += 1

    def _draw_arc(self, points: list[tuple[float, float]], pen: QPen, instance_name: str | None = None) -> None:
        if len(points) < 3:
            self._draw_polyline(points, pen, instance_name=instance_name)
            return
        s1, s2, s3 = (self._to_scene(points[0][0], points[0][1]), self._to_scene(points[1][0], points[1][1]), self._to_scene(points[2][0], points[2][1]))
        x1, y1 = s1
        x2, y2 = s2
        x3, y3 = s3
        det = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(det) < 1e-6:
            self._draw_polyline([s1, s2, s3], pen, instance_name=instance_name)
            return
        ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / det
        uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / det
        radius = math.hypot(x1 - ux, y1 - uy)
        if radius < 0.2:
            return
        a1 = math.atan2(y1 - uy, x1 - ux)
        a2 = math.atan2(y2 - uy, x2 - ux)
        a3 = math.atan2(y3 - uy, x3 - ux)

        def normalize(angle: float) -> float:
            while angle < 0:
                angle += 2 * math.pi
            while angle >= 2 * math.pi:
                angle -= 2 * math.pi
            return angle

        def mid_on_ccw(start: float, end: float, mid: float) -> bool:
            start = normalize(start)
            end = normalize(end)
            mid = normalize(mid)
            if end < start:
                end += 2 * math.pi
            if mid < start:
                mid += 2 * math.pi
            return start <= mid <= end

        sweep = normalize(a3) - normalize(a1)
        if sweep < 0:
            sweep += 2 * math.pi
        if not mid_on_ccw(a1, a1 + sweep, a2):
            sweep -= 2 * math.pi
        steps = max(10, int(abs(sweep) * radius / 10.0))
        pts: list[tuple[float, float]] = []
        for idx in range(steps + 1):
            angle = a1 + sweep * (idx / steps)
            pts.append((ux + radius * math.cos(angle), uy + radius * math.sin(angle)))
        for start, end in zip(pts, pts[1:]):
            item = self._scene.addLine(start[0], start[1], end[0], end[1], pen)
            self._tag_item(item, instance_name)
            self._stats.wire_segment_count += 1

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

    @staticmethod
    def _is_pmos_instance(instance_block: str) -> bool:
        cell_ref = EdifTextParser.first_any_or_none(instance_block, "(cellRef ")
        view_ref = EdifTextParser.first_direct_or_none(instance_block, "(viewRef ")
        cell_name = EdifTextParser.parse_header_name(cell_ref, "cellRef") if cell_ref else ""
        view_name = EdifTextParser.parse_header_name(view_ref, "viewRef") if view_ref else ""
        if "PMOS" in cell_name.upper() or "PMOS" in view_name.upper():
            return True
        return "PMOS" in instance_block.upper()

    @staticmethod
    def _is_mos_instance(instance_block: str) -> bool:
        upper = instance_block.upper()
        return "NMOS" in upper or "PMOS" in upper

    def _draw_mos_contrast_overlay(self, draw_block: str, transform: Affine2D) -> None:
        light_pen = QPen(QColor(214, 230, 255, 170))
        light_pen.setWidthF(1.2)
        light_pen.setCosmetic(True)
        for figure_block in EdifTextParser.find_direct_classes(draw_block, "(figure "):
            for path_block in EdifTextParser.find_classes(figure_block, "(path"):
                for point_list_block in EdifTextParser.find_classes(path_block, "(pointList"):
                    points = EdifTextParser.parse_points(point_list_block)
                    if len(points) < 2:
                        continue
                    world = [transform.apply(p.x, p.y) for p in points]
                    for start, end in zip(world, world[1:]):
                        x1, y1 = self._to_scene(start[0], start[1])
                        x2, y2 = self._to_scene(end[0], end[1])
                        item = self._scene.addLine(x1, y1, x2, y2, light_pen)
                        item.setZValue(905)

            for arc_block in EdifTextParser.find_classes(figure_block, "(arc"):
                points = EdifTextParser.parse_points(arc_block)
                if len(points) < 3:
                    continue
                world = [
                    transform.apply(points[0].x, points[0].y),
                    transform.apply(points[1].x, points[1].y),
                    transform.apply(points[2].x, points[2].y),
                ]
                self._draw_arc(world, light_pen)

            for rect_block in EdifTextParser.find_classes(figure_block, "(rectangle"):
                points = EdifTextParser.parse_points(rect_block)
                if len(points) < 2:
                    continue
                p1 = transform.apply(points[0].x, points[0].y)
                p2 = transform.apply(points[1].x, points[1].y)
                x1, y1 = self._to_scene(p1[0], p1[1])
                x2, y2 = self._to_scene(p2[0], p2[1])
                item = self._scene.addRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), light_pen)
                item.setZValue(905)

            for circle_block in EdifTextParser.find_classes(figure_block, "(circle"):
                points = EdifTextParser.parse_points(circle_block)
                if len(points) < 2:
                    continue
                center = transform.apply(points[0].x, points[0].y)
                edge = transform.apply(points[1].x, points[1].y)
                radius = math.hypot(edge[0] - center[0], edge[1] - center[1])
                cx, cy = self._to_scene(center[0], center[1])
                item = self._scene.addEllipse(
                    cx - radius * self._scale,
                    cy - radius * self._scale,
                    2 * radius * self._scale,
                    2 * radius * self._scale,
                    light_pen,
                )
                item.setZValue(905)

            for dot_block in EdifTextParser.find_classes(figure_block, "(dot"):
                points = EdifTextParser.parse_points(dot_block)
                if not points:
                    continue
                p = transform.apply(points[0].x, points[0].y)
                x, y = self._to_scene(p[0], p[1])
                item = self._scene.addEllipse(x - 1.8, y - 1.8, 3.6, 3.6, light_pen, QBrush(light_pen.color()))
                item.setZValue(905)

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

    def _draw_ground_glyph_at(
        self,
        world_x: float,
        world_y: float,
        color: tuple[int, int, int],
        label_text: str | None = None,
    ) -> None:
        draw_color = self._visible_port_color("ground", color)
        pen = QPen(QColor(*draw_color, 230))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        lines = [
            ((-10, 0), (10, 0)),
            ((-7, -3), (7, -3)),
            ((-4, -6), (4, -6)),
            ((-1, -9), (1, -9)),
        ]
        for start, end in lines:
            x1, y1 = self._to_scene(world_x + start[0], world_y + start[1])
            x2, y2 = self._to_scene(world_x + end[0], world_y + end[1])
            item = self._scene.addLine(x1, y1, x2, y2, pen)
            item.setZValue(920)
            self._stats.primitive_count += 1

        if label_text:
            gap_world = 4.0 / max(self._scale, 1e-6)
            self._draw_text(
                text=label_text,
                world_x=world_x + gap_world,
                world_y=world_y,
                justify="CENTERLEFT",
                color=draw_color,
                text_height=24,
                text_scale=0.1,
            )

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

    def _draw_container(
        self,
        block: str,
        transform: Affine2D,
        library_name: str | None,
        scene_data: SceneData,
        depth: int = 0,
        click_owner_instance: str | None = None,
    ) -> None:
        if depth > 10:
            return
        for child in EdifTextParser.direct_children(block):
            stripped = child.lstrip()
            if stripped.startswith("(figure "):
                figure_block = child
                figure_group_name = self._figure_group_name(figure_block)
                if figure_group_name and figure_group_name.upper() == "PAGEBORDER":
                    continue
                style = self._resolve_style(figure_block, library_name)
                pen = self._pen_from_style(style)
                brush = QBrush(QColor(*style["color"]))

                for path_block in EdifTextParser.find_classes(figure_block, "(path"):
                    for point_list_block in EdifTextParser.find_classes(path_block, "(pointList"):
                        points = EdifTextParser.parse_points(point_list_block)
                        if not points:
                            continue
                        world = [transform.apply(p.x, p.y) for p in points]
                        self._draw_polyline(world, pen, instance_name=click_owner_instance)
                        self._stats.primitive_count += 1

                for arc_block in EdifTextParser.find_classes(figure_block, "(arc"):
                    points = EdifTextParser.parse_points(arc_block)
                    if len(points) >= 3:
                        world = [
                            transform.apply(points[0].x, points[0].y),
                            transform.apply(points[1].x, points[1].y),
                            transform.apply(points[2].x, points[2].y),
                        ]
                        self._draw_arc(world, pen, instance_name=click_owner_instance)
                        self._stats.primitive_count += 1

                for rect_block in EdifTextParser.find_classes(figure_block, "(rectangle"):
                    points = EdifTextParser.parse_points(rect_block)
                    if len(points) >= 2:
                        p1 = transform.apply(points[0].x, points[0].y)
                        p2 = transform.apply(points[1].x, points[1].y)
                        x1, y1 = self._to_scene(p1[0], p1[1])
                        x2, y2 = self._to_scene(p2[0], p2[1])
                        item = self._scene.addRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), pen)
                        self._tag_item(item, click_owner_instance)
                        self._stats.primitive_count += 1

                for circle_block in EdifTextParser.find_classes(figure_block, "(circle"):
                    points = EdifTextParser.parse_points(circle_block)
                    if len(points) >= 2:
                        center = transform.apply(points[0].x, points[0].y)
                        edge = transform.apply(points[1].x, points[1].y)
                        radius = math.hypot(edge[0] - center[0], edge[1] - center[1])
                        cx, cy = self._to_scene(center[0], center[1])
                        item = self._scene.addEllipse(
                            cx - radius * self._scale,
                            cy - radius * self._scale,
                            2 * radius * self._scale,
                            2 * radius * self._scale,
                            pen,
                        )
                        self._tag_item(item, click_owner_instance)
                        self._stats.primitive_count += 1

                for dot_block in EdifTextParser.find_classes(figure_block, "(dot"):
                    points = EdifTextParser.parse_points(dot_block)
                    if points:
                        p = transform.apply(points[0].x, points[0].y)
                        x, y = self._to_scene(p[0], p[1])
                        item = self._scene.addEllipse(x - 1.8, y - 1.8, 3.6, 3.6, pen, brush)
                        self._tag_item(item, click_owner_instance)
                        self._stats.primitive_count += 1
            elif stripped.startswith("(stringDisplay "):
                self._draw_string_displays(child, transform, library_name)
            elif stripped.startswith("(keywordDisplay "):
                # keywordDisplay carries display/origin metadata similar to stringDisplay.
                self._draw_string_displays(child, transform, library_name)
            elif stripped.startswith("(instance "):
                nested_inst = child
                nested_view_ref = EdifTextParser.first_direct_or_none(nested_inst, "(viewRef ")
                nested_cell_ref = EdifTextParser.first_any_or_none(nested_inst, "(cellRef ")
                nested_lib_ref = EdifTextParser.first_any_or_none(nested_inst, "(libraryRef ")
                if not nested_view_ref or not nested_cell_ref or not nested_lib_ref:
                    continue
                nested_view = EdifTextParser.parse_header_name(nested_view_ref, "viewRef")
                nested_cell = EdifTextParser.parse_header_name(nested_cell_ref, "cellRef")
                nested_lib = EdifTextParser.parse_header_name(nested_lib_ref, "libraryRef")
                view_block = scene_data.view_index.get((nested_lib, nested_cell, nested_view))
                if not view_block:
                    continue
                draw_block = self._resolve_view_draw_block(view_block)
                if not draw_block:
                    continue
                nested_transform = transform.compose(self._transform_from_block(nested_inst))
                self._draw_container(
                    draw_block,
                    nested_transform,
                    nested_lib,
                    scene_data,
                    depth + 1,
                    click_owner_instance=click_owner_instance,
                )

    def _draw_top_page(self, scene_data: SceneData) -> None:
        for child in EdifTextParser.direct_children(scene_data.top_page_block):
            stripped = child.lstrip()
            if stripped.startswith("(net "):
                net_block = child
                for figure_block in EdifTextParser.find_direct_classes(net_block, "(figure "):
                    wrapper = f"(tmp {figure_block})"
                    self._draw_container(wrapper, Affine2D(), scene_data.netlist.library, scene_data)
            elif stripped.startswith("(portImplementation"):
                # Top-level page ports (e.g., OUTx) are represented as portImplementation.
                self._draw_port_implementation(child, (180, 30, 30))
            elif stripped.startswith("(instance "):
                instance_block = child
                if self._is_hidden_instance(instance_block):
                    continue
                inst_transform = self._transform_from_block(instance_block)
                view_ref = EdifTextParser.first_direct_or_none(instance_block, "(viewRef ")
                cell_ref = EdifTextParser.first_any_or_none(instance_block, "(cellRef ")
                lib_ref = EdifTextParser.first_any_or_none(instance_block, "(libraryRef ")
                if not view_ref or not cell_ref or not lib_ref:
                    self._draw_instance_overlay_fallback(instance_block, inst_transform)
                    continue
                view_name = EdifTextParser.parse_header_name(view_ref, "viewRef")
                cell_name = EdifTextParser.parse_header_name(cell_ref, "cellRef")
                lib_name = EdifTextParser.parse_header_name(lib_ref, "libraryRef")
                view_block = scene_data.view_index.get((lib_name, cell_name, view_name))
                if not view_block:
                    self._draw_instance_overlay_fallback(instance_block, inst_transform)
                    continue
                draw_block = self._resolve_view_draw_block(view_block)
                if not draw_block:
                    self._draw_instance_overlay_fallback(instance_block, inst_transform)
                    continue
                instance_name = EdifTextParser.parse_header_name(instance_block, "instance")
                instance_color = self._resolve_instance_primary_color(draw_block, lib_name)
                self._draw_container(
                    draw_block,
                    inst_transform,
                    lib_name,
                    scene_data,
                    click_owner_instance=instance_name,
                )
                if self._dark_mode and self._is_mos_instance(instance_block):
                    self._draw_mos_contrast_overlay(draw_block, inst_transform)
                self._draw_port_instances(instance_block, inst_transform, instance_color, draw_block)
                self._draw_pmos_gate_bubble(instance_block, inst_transform, draw_block, instance_color)
                self._draw_instance_hitbox(instance_block, inst_transform, draw_block)
                if not self._is_port_instance_for_visualization(instance_block):
                    self._draw_string_displays(
                        instance_block,
                        Affine2D(),
                        scene_data.netlist.library,
                        text_scale=0.1,
                    )

    def _draw_logic_highlight_overlay(self, scene_data: SceneData) -> None:
        if not self._highlight_nets and not self._highlight_instances:
            return

        net_pen = QPen(QColor(79, 171, 255, 235))
        net_pen.setWidthF(2.2)
        net_pen.setCosmetic(True)

        for net in scene_data.netlist.nets:
            if net.name not in self._highlight_nets:
                continue
            for wire in net.wires:
                if len(wire.points) == 1:
                    x, y = self._to_scene(wire.points[0].x, wire.points[0].y)
                    dot = self._scene.addEllipse(x - 2.6, y - 2.6, 5.2, 5.2, net_pen, QBrush(net_pen.color()))
                    dot.setZValue(3000)
                    continue
                if len(wire.points) < 2:
                    continue
                for start, end in zip(wire.points, wire.points[1:]):
                    x1, y1 = self._to_scene(start.x, start.y)
                    x2, y2 = self._to_scene(end.x, end.y)
                    item = self._scene.addLine(x1, y1, x2, y2, net_pen)
                    item.setZValue(3000)

        # Intentionally highlight only nets/wires; no per-instance marker circles.

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

    @staticmethod
    def _collect_drawable_points(draw_block: str) -> list:
        points = []
        # Use only direct figure geometry of the symbol container to match
        # the rendered instance body dimensions; recursive figure search can
        # include nested symbol internals and inflate hitboxes.
        for figure_block in EdifTextParser.find_direct_classes(draw_block, "(figure "):
            for path_block in EdifTextParser.find_direct_classes(figure_block, "(path"):
                for point_list_block in EdifTextParser.find_classes(path_block, "(pointList"):
                    points.extend(EdifTextParser.parse_points(point_list_block))
            for arc_block in EdifTextParser.find_direct_classes(figure_block, "(arc"):
                points.extend(EdifTextParser.parse_points(arc_block))
            for rect_block in EdifTextParser.find_direct_classes(figure_block, "(rectangle"):
                points.extend(EdifTextParser.parse_points(rect_block))
            for circle_block in EdifTextParser.find_direct_classes(figure_block, "(circle"):
                points.extend(EdifTextParser.parse_points(circle_block))
            for dot_block in EdifTextParser.find_direct_classes(figure_block, "(dot"):
                points.extend(EdifTextParser.parse_points(dot_block))
        return points


def re_search_first_quoted(text: str) -> str | None:
    import re

    match = re.search(r'"([^"]+)"', text)
    return match.group(1) if match else None
