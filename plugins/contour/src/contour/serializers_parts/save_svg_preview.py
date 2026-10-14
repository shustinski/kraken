"""serializers: save svg preview."""
from __future__ import annotations
import shutil
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from ..application.processing import DisplaySettings, SaveOptions
from contour.domain import PolygonData
from contour.i18n import tr
from contour.utils import draw_polygon_overlay, ensure_directory, imwrite_unicode_safe

from contour.serializers_parts.cv_object_polygon import (
    _cv_coord,
    save_polygons_json,
)

from contour.serializers_parts.cv_point_points import save_polygons_cv

from contour.serializers_parts.save_polygons_cif import (
    save_polygons_cif,
    save_polygons_csv,
    save_polygons_txt,
)

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
            points_attr = " ".join(f"{_cv_coord(x)},{_cv_coord(y)}" for x, y in polygon.points)
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
    if options.save_cv:
        path = root / f"{stem}.cv"
        saved["cv"] = str(save_polygons_cv(path, image_path, polygons, image_size=image_size, metadata=metadata))
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
