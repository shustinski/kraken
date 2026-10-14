"""scene_renderer: collect drawable points."""
from __future__ import annotations
from krona.infrastructure.edif_parser import EdifTextParser


class CollectDrawablePointsMixin:
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
