from __future__ import annotations

from dataclasses import dataclass
import math

from PyQt6.QtGui import QBrush, QColor, QFont, QPen
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

    def render(self, scene_data: SceneData) -> RenderStats:
        self._scene.clear()
        self._stats = RenderStats()
        self._build_figure_group_style_index(scene_data.source_text)
        self._draw_top_page(scene_data)
        return self._stats

    def _to_scene(self, x: float, y: float) -> tuple[float, float]:
        return x * self._scale, -y * self._scale

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

    def _pen_from_style(self, style: dict) -> QPen:
        pen = QPen(QColor(*style["color"]))
        pen.setWidthF(max(0.8, style["path_width"] * self._scale))
        pen.setCosmetic(False)
        return pen

    def _draw_polyline(self, points: list[tuple[float, float]], pen: QPen) -> None:
        if len(points) < 2:
            if points:
                x, y = self._to_scene(points[0][0], points[0][1])
                self._scene.addEllipse(x - 1.5, y - 1.5, 3.0, 3.0, pen, QBrush(pen.color()))
                self._stats.wire_segment_count += 1
            return
        for start, end in zip(points, points[1:]):
            x1, y1 = self._to_scene(start[0], start[1])
            x2, y2 = self._to_scene(end[0], end[1])
            self._scene.addLine(x1, y1, x2, y2, pen)
            self._stats.wire_segment_count += 1

    def _draw_arc(self, points: list[tuple[float, float]], pen: QPen) -> None:
        if len(points) < 3:
            self._draw_polyline(points, pen)
            return
        s1, s2, s3 = (self._to_scene(points[0][0], points[0][1]), self._to_scene(points[1][0], points[1][1]), self._to_scene(points[2][0], points[2][1]))
        x1, y1 = s1
        x2, y2 = s2
        x3, y3 = s3
        det = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(det) < 1e-6:
            self._draw_polyline([s1, s2, s3], pen)
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
            self._scene.addLine(start[0], start[1], end[0], end[1], pen)
            self._stats.wire_segment_count += 1

    def _draw_text(
        self,
        text: str,
        world_x: float,
        world_y: float,
        justify: str,
        color: tuple[int, int, int],
        text_height: int,
    ) -> None:
        if not text.strip():
            return
        sx, sy = self._to_scene(world_x, world_y)
        item = self._scene.addSimpleText(text)
        font = QFont("Arial")
        font.setPointSizeF(max(6.0, text_height * 0.65))
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

    def _draw_instance_overlay(self, instance_block: str, transform: Affine2D) -> None:
        pins = self._parse_instance_pin_points(instance_block)
        if not pins:
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
        self._scene.addRect(x, y, w, h, body_pen, body_brush)
        self._stats.primitive_count += 1

        pin_pen = QPen(QColor(15, 23, 42, 240))
        pin_pen.setWidthF(max(0.8, 0.5 * self._scale * 10.0))
        pin_brush = QBrush(QColor(15, 23, 42, 220))
        for pin_name, wx, wy in world:
            px, py = self._to_scene(wx, wy)
            self._scene.addEllipse(px - 2.0, py - 2.0, 4.0, 4.0, pin_pen, pin_brush)
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

    def _draw_string_displays(self, block: str, transform: Affine2D, library_name: str | None) -> None:
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
            self._draw_text(value, x, y, justify, style["color"], style["text_height"])

    def _draw_container(self, block: str, transform: Affine2D, library_name: str | None, scene_data: SceneData, depth: int = 0) -> None:
        if depth > 10:
            return
        for figure_block in EdifTextParser.find_direct_classes(block, "(figure "):
            style = self._resolve_style(figure_block, library_name)
            pen = self._pen_from_style(style)
            brush = QBrush(QColor(*style["color"]))

            for path_block in EdifTextParser.find_classes(figure_block, "(path"):
                for point_list_block in EdifTextParser.find_classes(path_block, "(pointList"):
                    points = EdifTextParser.parse_points(point_list_block)
                    if not points:
                        continue
                    world = [transform.apply(p.x, p.y) for p in points]
                    self._draw_polyline(world, pen)
                    self._stats.primitive_count += 1

            for arc_block in EdifTextParser.find_classes(figure_block, "(arc"):
                points = EdifTextParser.parse_points(arc_block)
                if len(points) >= 3:
                    world = [transform.apply(points[0].x, points[0].y), transform.apply(points[1].x, points[1].y), transform.apply(points[2].x, points[2].y)]
                    self._draw_arc(world, pen)
                    self._stats.primitive_count += 1

            for rect_block in EdifTextParser.find_classes(figure_block, "(rectangle"):
                points = EdifTextParser.parse_points(rect_block)
                if len(points) >= 2:
                    p1 = transform.apply(points[0].x, points[0].y)
                    p2 = transform.apply(points[1].x, points[1].y)
                    x1, y1 = self._to_scene(p1[0], p1[1])
                    x2, y2 = self._to_scene(p2[0], p2[1])
                    self._scene.addRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), pen)
                    self._stats.primitive_count += 1

            for circle_block in EdifTextParser.find_classes(figure_block, "(circle"):
                points = EdifTextParser.parse_points(circle_block)
                if len(points) >= 2:
                    center = transform.apply(points[0].x, points[0].y)
                    edge = transform.apply(points[1].x, points[1].y)
                    radius = math.hypot(edge[0] - center[0], edge[1] - center[1])
                    cx, cy = self._to_scene(center[0], center[1])
                    self._scene.addEllipse(cx - radius * self._scale, cy - radius * self._scale, 2 * radius * self._scale, 2 * radius * self._scale, pen)
                    self._stats.primitive_count += 1

            for dot_block in EdifTextParser.find_classes(figure_block, "(dot"):
                points = EdifTextParser.parse_points(dot_block)
                if points:
                    p = transform.apply(points[0].x, points[0].y)
                    x, y = self._to_scene(p[0], p[1])
                    self._scene.addEllipse(x - 1.8, y - 1.8, 3.6, 3.6, pen, brush)
                    self._stats.primitive_count += 1

        self._draw_string_displays(block, transform, library_name)

        for nested_inst in EdifTextParser.find_direct_classes(block, "(instance "):
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
            self._draw_container(draw_block, nested_transform, nested_lib, scene_data, depth + 1)

    def _draw_top_page(self, scene_data: SceneData) -> None:
        for net_block in EdifTextParser.find_direct_classes(scene_data.top_page_block, "(net "):
            for figure_block in EdifTextParser.find_direct_classes(net_block, "(figure "):
                wrapper = f"(tmp {figure_block})"
                self._draw_container(wrapper, Affine2D(), scene_data.netlist.library, scene_data)

        for instance_block in EdifTextParser.find_direct_classes(scene_data.top_page_block, "(instance "):
            view_ref = EdifTextParser.first_direct_or_none(instance_block, "(viewRef ")
            cell_ref = EdifTextParser.first_any_or_none(instance_block, "(cellRef ")
            lib_ref = EdifTextParser.first_any_or_none(instance_block, "(libraryRef ")
            if not view_ref or not cell_ref or not lib_ref:
                continue
            view_name = EdifTextParser.parse_header_name(view_ref, "viewRef")
            cell_name = EdifTextParser.parse_header_name(cell_ref, "cellRef")
            lib_name = EdifTextParser.parse_header_name(lib_ref, "libraryRef")
            view_block = scene_data.view_index.get((lib_name, cell_name, view_name))
            if not view_block:
                continue
            draw_block = self._resolve_view_draw_block(view_block)
            if not draw_block:
                continue
            inst_transform = self._transform_from_block(instance_block)
            # Always draw a schematic-like instance body/pins for readability.
            self._draw_instance_overlay(instance_block, inst_transform)
            self._draw_container(draw_block, inst_transform, lib_name, scene_data)
            self._draw_string_displays(instance_block, Affine2D(), scene_data.netlist.library)


def re_search_first_quoted(text: str) -> str | None:
    import re

    match = re.search(r'"([^"]+)"', text)
    return match.group(1) if match else None
