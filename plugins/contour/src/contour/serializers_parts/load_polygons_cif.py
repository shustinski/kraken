"""serializers: load polygons cif."""
from __future__ import annotations
from pathlib import Path
from contour.domain import PolygonData
from contour.i18n import tr

from contour.serializers_parts.module_state import (
    _CIF_PARSE_CACHE,
    _CIF_PARSE_CACHE_LOCK,
)

from contour.serializers_parts.cif_box_primitive_polygon import _cif_primitives_to_polygons

from contour.serializers_parts.cif_paint_mask_ring import _ensure_cif_paint_rings

from contour.serializers_parts.cv_object_polygon import (
    _cache_cif_parse_result,
    _cif_parse_cache_key,
)

from contour.serializers_parts.recover_cut_hole_topology import _recover_cut_hole_topology

def _load_polygons_cif_via_klayout(
    cif_path: Path,
) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]] | None:
    from ..infrastructure.cif_klayout_reader import klayout_cif_reader_enabled, load_cif_primitives_klayout

    if not klayout_cif_reader_enabled():
        return None

    parsed = load_cif_primitives_klayout(cif_path)
    return _cif_primitives_to_polygons(cif_path, parsed.primitives)

def _load_polygons_cif_via_opencif(
    cif_path: Path,
) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]] | None:
    from ..infrastructure.cif_opencif import (
        load_cif_primitives,
        opencif_loader_available,
        opencif_use_enabled,
    )

    if not opencif_use_enabled() or not opencif_loader_available():
        return None

    parsed = load_cif_primitives(cif_path)
    if parsed is None:
        return None
    if parsed.status == "cant_open":
        return None

    return _cif_primitives_to_polygons(cif_path, parsed.primitives)

def load_polygons_cif(path: str | Path) -> tuple[str | None, tuple[int, int] | None, list[PolygonData]]:
    from contour.serializers_parts.cif_box_polygon import _cif_box_to_polygon, _cif_cutout_display_requested, _cif_polygon_points_to_polygon, _extract_parenthesized_tokens, _parse_cif_int, _read_cif_text
    cif_path = Path(path)
    cache_key = _cif_parse_cache_key(cif_path)
    if cache_key is not None:
        with _CIF_PARSE_CACHE_LOCK:
            cached = _CIF_PARSE_CACHE.get(cache_key)
        if cached is not None:
            image_name, image_size, polygons = cached
            return image_name, image_size, [polygon.clone() for polygon in polygons]

    klayout_payload = _load_polygons_cif_via_klayout(cif_path)
    if klayout_payload is not None:
        image_name, image_size, polygons = klayout_payload
        cutout_display = _cif_cutout_display_requested(cif_path)
        polygons = _recover_cut_hole_topology(polygons, image_size, cutout_display=cutout_display)
        _ensure_cif_paint_rings(polygons)
        if cache_key is not None:
            _cache_cif_parse_result(cache_key, image_name, image_size, polygons)
        return image_name, image_size, polygons

    opencif_payload = _load_polygons_cif_via_opencif(cif_path)
    if opencif_payload is not None:
        image_name, image_size, polygons = opencif_payload
        cutout_display = _cif_cutout_display_requested(cif_path)
        polygons = _recover_cut_hole_topology(polygons, image_size, cutout_display=cutout_display)
        _ensure_cif_paint_rings(polygons)
        if cache_key is not None:
            _cache_cif_parse_result(cache_key, image_name, image_size, polygons)
        return image_name, image_size, polygons

    lines = _read_cif_text(cif_path).splitlines()

    image_name = None
    image_size = None
    polygons = []

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
            polygons.append(
                _cif_box_to_polygon(
                    _parse_cif_int(payload[0]),
                    _parse_cif_int(payload[1]),
                    _parse_cif_int(payload[2]),
                    _parse_cif_int(payload[3]),
                    image_size,
                    len(polygons) + 1,
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
        polygon = _cif_polygon_points_to_polygon(raw_points, image_size, len(polygons) + 1)
        if polygon is not None:
            polygons.append(polygon)

    cutout_display = _cif_cutout_display_requested(cif_path)
    polygons = _recover_cut_hole_topology(polygons, image_size, cutout_display=cutout_display)
    _ensure_cif_paint_rings(polygons)
    if cache_key is not None:
        _cache_cif_parse_result(cache_key, image_name, image_size, polygons)
    return image_name, image_size, polygons
