from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

from .application.processing import DisplaySettings, SaveOptions
from .domain import PolygonData, compute_polygon_metrics
from .i18n import tr
from .utils import draw_polygon_overlay, ensure_directory, imwrite_unicode_safe


def save_polygons_json(
    path: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    metadata: dict[str, object] | None = None,
) -> Path:
    output = Path(path)
    payload = {
        "image_path": image_path,
        "polygon_count": len(polygons),
        "polygons": [polygon.to_dict() for polygon in polygons],
        "metadata": metadata or {},
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _parse_cif_int(value: str) -> int:
    normalized = str(value or "").strip().rstrip(";")
    if not normalized:
        raise ValueError(tr("empty_cif_integer_token"))
    return int(normalized)


def _extract_parenthesized_tokens(line: str) -> list[str]:
    text = line.strip()
    if "(" not in text or ")" not in text:
        return []
    start = text.index("(") + 1
    end = text.rfind(")")
    if end <= start:
        return []
    return text[start:end].replace(";", " ").split()


def _read_cif_text(path: str | Path) -> str:
    cif_path = Path(path)
    payload = cif_path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp866"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("cp1251", errors="replace")


def load_polygons_cif(path: str | Path) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]]:
    cif_path = Path(path)
    lines = _read_cif_text(cif_path).splitlines()

    image_name: str | None = None
    image_size: tuple[int, int] | None = None
    polygons: list[PolygonData] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("( R "):
            tokens = _extract_parenthesized_tokens(stripped)
            if len(tokens) >= 2 and tokens[0] == "R":
                image_name = tokens[1]
            continue
        if stripped.startswith("( S "):
            tokens = _extract_parenthesized_tokens(stripped)
            if len(tokens) >= 3 and tokens[0] == "S":
                image_size = (_parse_cif_int(tokens[1]), _parse_cif_int(tokens[2]))
            continue
        if stripped.startswith("B "):
            if image_size is None:
                raise ValueError(tr("cif_size_header_missing", path=cif_path))
            payload = stripped[2:].rstrip(";").split()
            if len(payload) != 4:
                continue
            box_width = _parse_cif_int(payload[0])
            box_height = _parse_cif_int(payload[1])
            center_x = _parse_cif_int(payload[2])
            center_y = _parse_cif_int(payload[3])

            _, height = image_size
            image_center_y = float(height - center_y)
            half_width = float(box_width) / 2.0
            half_height = float(box_height) / 2.0
            image_points = [
                (float(center_x) - half_width, image_center_y - half_height),
                (float(center_x) + half_width, image_center_y - half_height),
                (float(center_x) + half_width, image_center_y + half_height),
                (float(center_x) - half_width, image_center_y + half_height),
            ]
            area, perimeter, bbox = compute_polygon_metrics(image_points)
            polygons.append(
                PolygonData(
                    id=len(polygons) + 1,
                    points=image_points,
                    is_hole=False,
                    parent_id=None,
                    category="via",
                    shape_hint="box",
                    area=area,
                    perimeter=perimeter,
                    bbox=bbox,
                )
            )
            continue
        if not stripped.startswith("P "):
            continue
        if image_size is None:
            raise ValueError(tr("cif_size_header_missing", path=cif_path))

        payload = stripped[2:].rstrip(";").split()
        if len(payload) < 6 or len(payload) % 2 != 0:
            continue

        raw_points = [
            (_parse_cif_int(payload[index]), _parse_cif_int(payload[index + 1])) for index in range(0, len(payload), 2)
        ]
        if len(raw_points) >= 2 and raw_points[0] == raw_points[-1]:
            raw_points = raw_points[:-1]
        if len(raw_points) < 3:
            continue

        _width, height = image_size
        image_points = [(float(x_coord), float(height - y_coord)) for x_coord, y_coord in raw_points]
        area, perimeter, bbox = compute_polygon_metrics(image_points)
        polygons.append(
            PolygonData(
                id=len(polygons) + 1,
                points=image_points,
                is_hole=False,
                parent_id=None,
                category="conductor",
                shape_hint="polygon",
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        )

    polygons = _recover_cut_hole_topology(polygons, image_size)
    return image_name, image_size, polygons


def _polygon_to_cif_line(polygon: PolygonData, image_width: int, image_height: int) -> str:
    if polygon.shape_hint == "box":
        x_values = [point[0] for point in polygon.points]
        y_values = [point[1] for point in polygon.points]
        if len(x_values) < 4 or len(y_values) < 4:
            return ""
        width = max(1, round(max(x_values) - min(x_values)))
        height = max(1, round(max(y_values) - min(y_values)))
        center_x = round((min(x_values) + max(x_values)) / 2.0)
        center_y = round((min(y_values) + max(y_values)) / 2.0)
        cif_x = max(0, min(image_width, center_x))
        cif_y = max(0, min(image_height, round(image_height - center_y)))
        return f"B {width} {height} {cif_x} {cif_y};"
    points = []
    for x_coord, y_coord in polygon.points:
        cif_x = max(0, min(image_width, round(x_coord)))
        cif_y = max(0, min(image_height, round(image_height - y_coord)))
        points.append((cif_x, cif_y))
    if len(points) < 3:
        return ""
    if points[0] != points[-1]:
        points.append(points[0])
    coordinates = " ".join(f"{x_coord} {y_coord}" for x_coord, y_coord in points)
    return f"P {coordinates};"


def _local_mask_bounds(points_groups: list[list[tuple[float, float]]], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    all_x: list[float] = []
    all_y: list[float] = []
    for points in points_groups:
        for x_coord, y_coord in points:
            all_x.append(float(x_coord))
            all_y.append(float(y_coord))
    min_x = max(0, int(np.floor(min(all_x))) - 3)
    min_y = max(0, int(np.floor(min(all_y))) - 3)
    max_x = min(image_width - 1, int(np.ceil(max(all_x))) + 3)
    max_y = min(image_height - 1, int(np.ceil(max(all_y))) + 3)
    return min_x, min_y, max_x, max_y


def _to_local_int_points(points: list[tuple[float, float]], left: int, top: int) -> np.ndarray:
    return np.array(
        [[round(x_coord - left), round(y_coord - top)] for x_coord, y_coord in points],
        dtype=np.int32,
    )


def _bridge_hole_to_outer(mask: np.ndarray, outer: np.ndarray, hole: np.ndarray) -> None:
    if len(outer) == 0 or len(hole) == 0:
        return
    outer_pts = outer.reshape(-1, 2)
    hole_pts = hole.reshape(-1, 2)
    best_outer = outer_pts[0]
    best_hole = hole_pts[0]
    best_dist = float("inf")
    for outer_point in outer_pts:
        dx = hole_pts[:, 0] - outer_point[0]
        dy = hole_pts[:, 1] - outer_point[1]
        distances = dx * dx + dy * dy
        nearest_index = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_index])
        if nearest_dist < best_dist:
            best_dist = nearest_dist
            best_outer = outer_point
            best_hole = hole_pts[nearest_index]
    cv2.line(
        mask,
        (int(best_outer[0]), int(best_outer[1])),
        (int(best_hole[0]), int(best_hole[1])),
        color=0,
        thickness=1,
        lineType=cv2.LINE_8,
    )


def _encode_parent_with_holes_cut_path(
    parent: PolygonData,
    holes: list[PolygonData],
    image_width: int,
    image_height: int,
) -> list[tuple[float, float]]:
    groups = [parent.points] + [hole.points for hole in holes if len(hole.points) >= 3]
    left, top, right, bottom = _local_mask_bounds(groups, image_width, image_height)
    local_width = max(1, right - left + 1)
    local_height = max(1, bottom - top + 1)
    mask = np.zeros((local_height, local_width), dtype=np.uint8)
    outer_local = _to_local_int_points(parent.points, left, top)
    cv2.fillPoly(mask, [outer_local], 255)
    hole_locals: list[np.ndarray] = []
    for hole in holes:
        hole_local = _to_local_int_points(hole.points, left, top)
        cv2.fillPoly(mask, [hole_local], 0)
        hole_locals.append(hole_local)
    outer_contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer_boundary = outer_contours[0] if outer_contours else outer_local.reshape(-1, 1, 2)
    for hole_local in hole_locals:
        _bridge_hole_to_outer(mask, outer_boundary, hole_local)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return parent.points
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return [(float(point[0] + left), float(point[1] + top)) for point in contour]


def _contour_points_to_polygon(
    contour: np.ndarray,
    *,
    left: int,
    top: int,
    polygon_id: int,
    is_hole: bool,
    parent_id: int | None,
) -> PolygonData:
    points = [(float(point[0][0] + left), float(point[0][1] + top)) for point in contour]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(
        id=polygon_id,
        points=points,
        is_hole=is_hole,
        parent_id=parent_id,
        category="conductor",
        shape_hint="polygon",
        area=area,
        perimeter=perimeter,
        bbox=bbox,
    )


def _recover_cut_hole_topology(polygons: list[PolygonData], image_size: tuple[int, int] | None) -> list[PolygonData]:
    if image_size is None:
        return polygons
    image_width, image_height = image_size
    del image_height
    recovered: list[PolygonData] = []
    next_id = 1
    for polygon in polygons:
        if polygon.shape_hint == "box" or polygon.category == "via" or len(polygon.points) < 3:
            clone = polygon.clone()
            clone.id = next_id
            recovered.append(clone)
            next_id += 1
            continue
        left = max(0, int(np.floor(min(point[0] for point in polygon.points))) - 3)
        top = max(0, int(np.floor(min(point[1] for point in polygon.points))) - 3)
        right = min(image_width - 1, int(np.ceil(max(point[0] for point in polygon.points))) + 3)
        bottom = int(np.ceil(max(point[1] for point in polygon.points))) + 3
        local_width = max(1, right - left + 1)
        local_height = max(1, bottom - top + 1)
        raw_mask = np.zeros((local_height, local_width), dtype=np.uint8)
        local_points = _to_local_int_points(polygon.points, left, top)
        cv2.fillPoly(raw_mask, [local_points], 255)
        _contours_raw, hierarchy_raw = cv2.findContours(raw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        raw_hole_count = 0
        if hierarchy_raw is not None and len(hierarchy_raw) > 0:
            raw_hole_count = int(sum(1 for item in hierarchy_raw[0] if int(item[3]) >= 0))
        best_contours: list[np.ndarray] = []
        best_hierarchy: np.ndarray | None = None
        best_hole_count = raw_hole_count
        for kernel_size in (3, 5):
            closed = cv2.morphologyEx(
                raw_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
                iterations=1,
            )
            contours_closed, hierarchy_closed = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            closed_hole_count = 0
            if hierarchy_closed is not None and len(hierarchy_closed) > 0:
                closed_hole_count = int(sum(1 for item in hierarchy_closed[0] if int(item[3]) >= 0))
            if closed_hole_count > best_hole_count:
                best_hole_count = closed_hole_count
                best_contours = [np.asarray(contour) for contour in contours_closed]
                best_hierarchy = hierarchy_closed
        if best_hole_count <= raw_hole_count or best_hierarchy is None or not best_contours or len(best_hierarchy) == 0:
            clone = polygon.clone()
            clone.id = next_id
            recovered.append(clone)
            next_id += 1
            continue
        hierarchy = best_hierarchy[0]
        contour_to_parent_id: dict[int, int] = {}
        for index, contour in enumerate(best_contours):
            parent_index = int(hierarchy[index][3])
            if parent_index >= 0:
                continue
            parent_poly = _contour_points_to_polygon(
                contour,
                left=left,
                top=top,
                polygon_id=next_id,
                is_hole=False,
                parent_id=None,
            )
            recovered.append(parent_poly)
            contour_to_parent_id[index] = next_id
            next_id += 1
        for index, contour in enumerate(best_contours):
            parent_index = int(hierarchy[index][3])
            if parent_index < 0:
                continue
            parent_id = contour_to_parent_id.get(parent_index)
            if parent_id is None:
                continue
            hole_poly = _contour_points_to_polygon(
                contour,
                left=left,
                top=top,
                polygon_id=next_id,
                is_hole=True,
                parent_id=parent_id,
            )
            recovered.append(hole_poly)
            next_id += 1
    return recovered


def save_polygons_cif(
    path: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    image_size: tuple[int, int],
    layer_name: str = "NM",
) -> Path:
    output = Path(path)
    width, height = int(image_size[0]), int(image_size[1])
    lines = [
        "DS 1 1 1;",
        f"L {layer_name};",
        f"( R {Path(image_path).name} );",
        f"( S {width} {height} );",
    ]
    sorted_polygons = sorted(polygons, key=lambda item: item.id)
    holes_by_parent: dict[int, list[PolygonData]] = {}
    for polygon in sorted_polygons:
        if polygon.is_hole and polygon.parent_id is not None:
            holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)
    for polygon in sorted_polygons:
        if polygon.is_hole:
            continue
        save_polygon = polygon
        if polygon.category != "via" and polygon.shape_hint != "box":
            holes = holes_by_parent.get(int(polygon.id), [])
            if holes:
                stitched_points = _encode_parent_with_holes_cut_path(
                    polygon, holes, image_width=width, image_height=height
                )
                area, perimeter, bbox = compute_polygon_metrics(stitched_points)
                save_polygon = PolygonData(
                    id=polygon.id,
                    points=stitched_points,
                    is_hole=False,
                    parent_id=None,
                    category=polygon.category,
                    shape_hint=polygon.shape_hint,
                    area=area,
                    perimeter=perimeter,
                    bbox=bbox,
                )
        line = _polygon_to_cif_line(save_polygon, image_width=width, image_height=height)
        if line:
            lines.append(line)
    lines.extend(["DF;", "E"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
                        f"{x_coord:.6f}",
                        f"{y_coord:.6f}",
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
        points_repr = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in polygon.points)
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


def save_svg_preview(
    path: str | Path,
    image_shape: tuple[int, ...],
    polygons: list[PolygonData],
    display_settings: DisplaySettings,
) -> Path:
    output = Path(path)
    height, width = image_shape[:2]
    alpha = max(0.0, min(1.0, display_settings.fill_opacity))
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101010"/>',
    ]
    for polygon in polygons:
        color = display_settings.hole_color if polygon.is_hole else display_settings.external_color
        if _is_ellipse_display_polygon(polygon):
            x_values = [float(point[0]) for point in polygon.points]
            y_values = [float(point[1]) for point in polygon.points]
            if len(x_values) < 3 or len(y_values) < 3:
                continue
            left = min(x_values)
            right = max(x_values)
            top = min(y_values)
            bottom = max(y_values)
            svg_lines.append(
                f'<ellipse cx="{(left + right) / 2.0:.3f}" cy="{(top + bottom) / 2.0:.3f}" '
                f'rx="{max(0.5, (right - left) / 2.0):.3f}" ry="{max(0.5, (bottom - top) / 2.0):.3f}" '
                f'fill="{color}" fill-opacity="{alpha:.3f}" stroke="{color}" '
                f'stroke-width="{display_settings.line_width:.2f}"/>'
            )
        else:
            points_attr = " ".join(f"{x:.3f},{y:.3f}" for x, y in polygon.points)
            svg_lines.append(
                f'<polygon points="{escape(points_attr)}" fill="{color}" fill-opacity="{alpha:.3f}" '
                f'stroke="{color}" stroke-width="{display_settings.line_width:.2f}"/>'
            )
    svg_lines.append("</svg>")
    output.write_text("\n".join(svg_lines), encoding="utf-8")
    return output


def save_overlay_preview(
    path: str | Path,
    source_image: np.ndarray,
    polygons: list[PolygonData],
    display_settings: DisplaySettings,
) -> Path:
    output = Path(path)
    preview = draw_polygon_overlay(source_image, polygons, display_settings)
    imwrite_unicode_safe(output, preview)
    return output


def _is_ellipse_display_polygon(polygon: PolygonData) -> bool:
    return polygon.shape_hint == "box" or polygon.category == "via"


def _copy_or_write_dataset_image(source_path: Path, target_path: Path, source_image: np.ndarray | None) -> Path:
    if source_path.exists() and source_path.is_file():
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        return target_path
    if source_image is None:
        raise FileNotFoundError(tr("unable_to_load_image", path=source_path))
    imwrite_unicode_safe(target_path, source_image)
    return target_path


def export_dataset_frame(
    dataset_directory: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    source_image: np.ndarray | None,
) -> dict[str, str]:
    root = ensure_directory(dataset_directory)
    images_root = ensure_directory(root / "images")
    cif_root = ensure_directory(root / "cif")
    source_path = Path(image_path)
    image_name = source_path.name
    if not source_path.suffix:
        image_name = f"{source_path.stem}.png"
    image_target = images_root / image_name
    cif_target = cif_root / f"{source_path.stem}.cif"

    image_size: tuple[int, int] | None = None
    if source_image is not None:
        image_size = (int(source_image.shape[1]), int(source_image.shape[0]))
    if image_size is None:
        raise ValueError(tr("dataset_source_image_missing", path=image_path))

    saved_image = _copy_or_write_dataset_image(source_path, image_target, source_image)
    saved_cif = save_polygons_cif(cif_target, str(saved_image), polygons, image_size=image_size)
    return {"image": str(saved_image), "cif": str(saved_cif)}


def save_result_bundle(
    output_directory: str | Path,
    image_path: str,
    polygons: list[PolygonData],
    source_image: np.ndarray | None,
    display_settings: DisplaySettings,
    save_options: SaveOptions | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, str]:
    options = save_options or SaveOptions()
    root = ensure_directory(output_directory)
    stem = Path(image_path).stem
    saved: dict[str, str] = {}
    image_size: tuple[int, int] | None = None
    if source_image is not None:
        image_size = (int(source_image.shape[1]), int(source_image.shape[0]))

    if options.save_cif and image_size is not None:
        path = root / f"{stem}.cif"
        saved["cif"] = str(save_polygons_cif(path, image_path, polygons, image_size=image_size))
    if options.save_json:
        path = root / f"{stem}.json"
        saved["json"] = str(save_polygons_json(path, image_path, polygons, metadata))
    if options.save_csv:
        path = root / f"{stem}.csv"
        saved["csv"] = str(save_polygons_csv(path, image_path, polygons))
    if options.save_txt:
        path = root / f"{stem}.txt"
        saved["txt"] = str(save_polygons_txt(path, image_path, polygons))
    if options.save_svg and source_image is not None:
        path = root / f"{stem}.svg"
        saved["svg"] = str(save_svg_preview(path, source_image.shape, polygons, display_settings))
    if options.save_preview and source_image is not None:
        path = root / f"{stem}_preview.png"
        saved["preview"] = str(save_overlay_preview(path, source_image, polygons, display_settings))
    return saved
