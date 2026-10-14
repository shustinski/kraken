"""scene_renderer: draw container."""
from __future__ import annotations
import math
from PyQt6.QtGui import QBrush, QColor
from krona.application.ports import SceneData
from krona.infrastructure.edif_parser import EdifTextParser


class DrawContainerMixin:
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
