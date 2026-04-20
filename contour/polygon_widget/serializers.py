from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

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

            width, height = image_size
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
            (_parse_cif_int(payload[index]), _parse_cif_int(payload[index + 1]))
            for index in range(0, len(payload), 2)
        ]
        if len(raw_points) >= 2 and raw_points[0] == raw_points[-1]:
            raw_points = raw_points[:-1]
        if len(raw_points) < 3:
            continue

        width, height = image_size
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

    return image_name, image_size, polygons


def _polygon_to_cif_line(polygon: PolygonData, image_width: int, image_height: int) -> str:
    if polygon.shape_hint == "box":
        x_values = [point[0] for point in polygon.points]
        y_values = [point[1] for point in polygon.points]
        if len(x_values) < 4 or len(y_values) < 4:
            return ""
        width = max(1, int(round(max(x_values) - min(x_values))))
        height = max(1, int(round(max(y_values) - min(y_values))))
        center_x = int(round((min(x_values) + max(x_values)) / 2.0))
        center_y = int(round((min(y_values) + max(y_values)) / 2.0))
        cif_x = max(0, min(image_width, center_x))
        cif_y = max(0, min(image_height, int(round(image_height - center_y))))
        return f"B {width} {height} {cif_x} {cif_y};"
    points = []
    for x_coord, y_coord in polygon.points:
        cif_x = max(0, min(image_width, int(round(x_coord))))
        cif_y = max(0, min(image_height, int(round(image_height - y_coord))))
        points.append((cif_x, cif_y))
    if len(points) < 3:
        return ""
    if points[0] != points[-1]:
        points.append(points[0])
    coordinates = " ".join(f"{x_coord} {y_coord}" for x_coord, y_coord in points)
    return f"P {coordinates};"


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
    for polygon in sorted(polygons, key=lambda item: item.id):
        line = _polygon_to_cif_line(polygon, image_width=width, image_height=height)
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
