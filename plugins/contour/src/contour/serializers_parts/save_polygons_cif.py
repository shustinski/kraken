"""serializers: save polygons cif."""
from __future__ import annotations
import csv
from pathlib import Path
from contour.domain import PolygonData, compute_polygon_metrics

from contour.serializers_parts.module_state import CIF_CUTOUT_DISPLAY_MARKER

from contour.serializers_parts.boundary_vertices_by_distance import _polygon_to_cif_line

from contour.serializers_parts.connected_cif_families import _connected_cif_families

from contour.serializers_parts.cv_object_polygon import (
    _cv_coord,
    invalidate_cif_parse_cache,
)

from contour.serializers_parts.encode_parent_holes_link import _encode_parent_with_holes_link_path

def save_polygons_cif(
    path: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    image_size: tuple[int, int],
    layer_name: str = "NM",
    *,
    cutout_display: bool = False,
) -> Path:
    output = Path(path)
    width, height = int(image_size[0]), int(image_size[1])
    lines = [
        "DS 1 1 1;",
        f"L {layer_name};",
        f"( R {Path(image_path).name} );",
        f"( S {width} {height} );",
    ]
    if cutout_display:
        lines.append(CIF_CUTOUT_DISPLAY_MARKER)
    sorted_polygons = sorted(polygons, key=lambda item: item.id)
    holes_by_parent: dict[int, list[PolygonData]] = {}
    orphan_holes: list[PolygonData] = []
    for polygon in sorted_polygons:
        if not polygon.is_hole:
            continue
        if polygon.parent_id is None:
            orphan_holes.append(polygon)
        else:
            holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)
    for polygon in sorted_polygons:
        if polygon.is_hole:
            continue
        save_families: list[tuple[PolygonData, list[PolygonData]]] = [(polygon, [])]
        if polygon.category != "via" and polygon.shape_hint != "box":
            holes = holes_by_parent.get(int(polygon.id), [])
            if holes:
                save_families = _connected_cif_families(polygon, holes)
        for family_outer, family_holes in save_families:
            save_polygon = family_outer
            if family_holes:
                linked_points = _encode_parent_with_holes_link_path(family_outer, family_holes)
                area, perimeter, bbox = compute_polygon_metrics(linked_points)
                save_polygon = PolygonData(
                    id=family_outer.id,
                    points=linked_points,
                    is_hole=False,
                    parent_id=None,
                    category=family_outer.category,
                    shape_hint=family_outer.shape_hint,
                    area=area,
                    perimeter=perimeter,
                    bbox=bbox,
                )
            line = _polygon_to_cif_line(save_polygon, image_width=width, image_height=height)
            if line:
                lines.append(line)
    for hole in orphan_holes:
        clone = hole.clone()
        clone.is_hole = False
        clone.parent_id = None
        line = _polygon_to_cif_line(clone, image_width=width, image_height=height)
        if line:
            lines.append(line)
    lines.extend(["DF;", "E"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    invalidate_cif_parse_cache([output])
    return output

def save_polygons_csv(path: str | Path, image_path: str, polygons: list[PolygonData]) -> Path:
    output = Path(path)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "polygon_id",
                "vertex_index",
                "x",
                "y",
                "is_hole",
                "parent_id",
                "category",
                "shape_hint",
                "area",
                "perimeter",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
            ]
        )
        for polygon in polygons:
            for vertex_index, (x_coord, y_coord) in enumerate(polygon.points):
                writer.writerow(
                    [
                        image_path,
                        polygon.id,
                        vertex_index,
                        str(_cv_coord(x_coord)),
                        str(_cv_coord(y_coord)),
                        int(polygon.is_hole),
                        "" if polygon.parent_id is None else polygon.parent_id,
                        polygon.category,
                        polygon.shape_hint,
                        f"{polygon.area:.6f}",
                        f"{polygon.perimeter:.6f}",
                        polygon.bbox[0],
                        polygon.bbox[1],
                        polygon.bbox[2],
                        polygon.bbox[3],
                    ]
                )
    return output

def save_polygons_txt(path: str | Path, image_path: str, polygons: list[PolygonData]) -> Path:
    output = Path(path)
    lines = [f"image_path: {image_path}", f"polygon_count: {len(polygons)}", ""]
    for polygon in polygons:
        points_repr = ", ".join(f"({_cv_coord(x)}, {_cv_coord(y)})" for x, y in polygon.points)
        lines.extend(
            [
                f"polygon_id: {polygon.id}",
                f"  is_hole: {polygon.is_hole}",
                f"  parent_id: {polygon.parent_id}",
                f"  category: {polygon.category}",
                f"  shape_hint: {polygon.shape_hint}",
                f"  area: {polygon.area:.6f}",
                f"  perimeter: {polygon.perimeter:.6f}",
                f"  bbox: {polygon.bbox}",
                f"  points: [{points_repr}]",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
