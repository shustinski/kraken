"""Pure helpers for image/CIF basename matching and list row status logic.

Contour matches overlays by lowercase file stem; list colors follow the UX spec
implemented in ``PolygonExtractionWidget``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImageCifMatchingReport:
    stems_with_image_but_no_cif: frozenset[str]
    stems_with_cif_but_no_image: frozenset[str]


def stems_lowercase(paths: Iterable[str | Path]) -> frozenset[str]:
    return frozenset(Path(p).stem.lower() for p in paths)


def build_image_cif_matching_report(
    image_paths: Iterable[str | Path],
    cif_paths_by_stem: Mapping[str, str],
) -> ImageCifMatchingReport:
    image_stems = stems_lowercase(image_paths)
    cif_stems = frozenset(dict(cif_paths_by_stem))
    return ImageCifMatchingReport(
        stems_with_image_but_no_cif=frozenset(image_stems - cif_stems),
        stems_with_cif_but_no_image=frozenset(cif_stems - image_stems),
    )


def index_cif_file_paths(paths: Iterable[str | Path]) -> dict[str, str]:
    """Map lowercase stem → absolute resolved path for existing ``.cif`` files."""

    indexed: dict[str, str] = {}
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_file() and candidate.suffix.lower() == ".cif":
            indexed[candidate.stem.lower()] = str(candidate.resolve())
    return indexed


class VectorSideListStatus(Enum):
    UNSEEN = "unseen"
    VIEWED = "viewed"
    MODIFIED = "modified"
    SAVED = "saved"
    LOAD_ERROR = "load_error"
    NO_MATCHING_IMAGE = "no_image"


class ImageSideListPaintStatus(Enum):
    UNOPENED = "unopened"
    VIEWED = "viewed"
    MODIFIED = "modified"
    SAVED = "saved"


def classify_vector_side_status(
    *,
    has_matching_image: bool,
    cif_load_failed: bool,
    image_never_viewed: bool,
    polygons_dirty: bool,
    persist_highlight: bool,
) -> VectorSideListStatus:
    if not has_matching_image:
        return VectorSideListStatus.NO_MATCHING_IMAGE
    if cif_load_failed:
        return VectorSideListStatus.LOAD_ERROR
    if polygons_dirty:
        return VectorSideListStatus.MODIFIED
    if image_never_viewed:
        return VectorSideListStatus.UNSEEN
    if persist_highlight:
        return VectorSideListStatus.SAVED
    return VectorSideListStatus.VIEWED


def classify_image_side_paint_status(
    *,
    never_opened: bool,
    polygons_dirty: bool,
    persist_highlight: bool,
) -> ImageSideListPaintStatus:
    if never_opened:
        return ImageSideListPaintStatus.UNOPENED
    if polygons_dirty:
        return ImageSideListPaintStatus.MODIFIED
    if persist_highlight:
        return ImageSideListPaintStatus.SAVED
    return ImageSideListPaintStatus.VIEWED


# Background (list row) fills — vector list (muted for dark Kraken QSS)
_HEX_VECTOR_UNSEEN = None
_HEX_VECTOR_VIEWED = "#3d4f66"
_HEX_VECTOR_MODIFIED = "#6b3a1e"
_HEX_VECTOR_SAVED = "#1e4a35"
_HEX_VECTOR_RED = "#6b2c2c"


def background_hex_vector_status(status: VectorSideListStatus) -> str | None:
    if status == VectorSideListStatus.NO_MATCHING_IMAGE or status == VectorSideListStatus.LOAD_ERROR:
        return _HEX_VECTOR_RED
    if status == VectorSideListStatus.UNSEEN:
        return _HEX_VECTOR_UNSEEN
    if status == VectorSideListStatus.VIEWED:
        return _HEX_VECTOR_VIEWED
    if status == VectorSideListStatus.MODIFIED:
        return _HEX_VECTOR_MODIFIED
    if status == VectorSideListStatus.SAVED:
        return _HEX_VECTOR_SAVED
    return _HEX_VECTOR_UNSEEN


# Background fills — image list (same semantics; unopened has no tint)
_HEX_IMAGE_UNOPENED = None
_HEX_IMAGE_VIEWED = "#3d4f66"
_HEX_IMAGE_MODIFIED = "#6b3a1e"
_HEX_IMAGE_SAVED = "#1e4a35"


def background_hex_image_paint_status(status: ImageSideListPaintStatus) -> str | None:
    if status == ImageSideListPaintStatus.UNOPENED:
        return _HEX_IMAGE_UNOPENED
    if status == ImageSideListPaintStatus.VIEWED:
        return _HEX_IMAGE_VIEWED
    if status == ImageSideListPaintStatus.MODIFIED:
        return _HEX_IMAGE_MODIFIED
    return _HEX_IMAGE_SAVED


# Foreground for image rows — presence of indexed CIF
_HEX_IMAGE_FG_WITH_VECTOR = "#E5E7EB"
_HEX_IMAGE_FG_WITHOUT_VECTOR = "#64748B"


def foreground_hex_image_has_vector_overlay(has_matching_cif: bool) -> str:
    return _HEX_IMAGE_FG_WITH_VECTOR if has_matching_cif else _HEX_IMAGE_FG_WITHOUT_VECTOR
