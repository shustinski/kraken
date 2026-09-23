from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

import karakal.core.grid_anomaly as grid_anomaly
import karakal.core.workers as workers_module
from karakal.core.grid_anomaly import _outline_side_coverages, detect_grid_cell_anomalies
from karakal.core.domain import FrameRecord
from karakal.core.performance import PerformanceConfig, ProfilingMode
from karakal.core.workers import GridInspectionWorker
from karakal.ui.matrix_view import MatrixListWidget


GOLDEN_DIGESTS = (
    "1459a05912216239747feae38365d4d8f8402d878915f08515469d8bdb329e83",
    "f0e6cb5c973f719747009213a44668ea9da0902915c470db76cb61c666ca9360",
    "61f0338b714799f3eec395e617aa6267efdd435080d9ecb6717bf3a80f3693f8",
    "2f000b3c87d88cd326e22bac843a9402095f90e8149c6ab8ac61ea0bf5001b00",
    "44621366b8f0fd75f9df0bdd93e1276ff919d874ba2cec41e47bba49df6eebed",
    "e2323740fe4e02eefefa4e1514efd3fbd6ce1b0885035e561e645bb672913fab",
    "1459a05912216239747feae38365d4d8f8402d878915f08515469d8bdb329e83",
    "1459a05912216239747feae38365d4d8f8402d878915f08515469d8bdb329e83",
)


def _synthetic_grid_frame(variant: int, shape: tuple[int, int] = (768, 1024)) -> np.ndarray:
    height, width = shape
    rows, cols = 12, 16
    cell_w = max(8, width // (cols + 2))
    cell_h = max(8, height // (rows + 2))
    margin_x = max(4, (width - cols * cell_w) // 2)
    margin_y = max(4, (height - rows * cell_h) // 2)
    image = np.zeros((height, width), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            x = margin_x + col * cell_w
            y = margin_y + row * cell_h
            cv2.rectangle(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 255, 2)
    x = margin_x + (variant * 5 % cols) * cell_w
    y = margin_y + (variant * 3 % rows) * cell_h
    defect = variant % 8
    if defect == 1:
        cv2.rectangle(image, (x + 4, y + 4), (x + cell_w - 5, y + cell_h - 5), 255, -1)
    elif defect == 2:
        cv2.rectangle(image, (x + 4, y + cell_h // 2), (x + cell_w - 5, y + cell_h - 5), 255, -1)
    elif defect == 3:
        cv2.circle(image, (min(width - 2, x + cell_w + 5), min(height - 2, y + 5)), 3, 255, -1)
    elif defect == 4:
        cv2.line(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 0, 3)
    elif defect == 5:
        cv2.rectangle(image, (x + cell_w - 5, y + 4), (min(width - 1, x + cell_w + 8), y + cell_h - 5), 255, -1)
    elif defect == 6:
        image[:, :] = cv2.add(image, np.full_like(image, 20))
    elif defect == 7:
        noise = np.random.default_rng(variant).normal(0.0, 7.0, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return image


def _result_digest(result) -> str:
    payload = dataclasses.asdict(result)
    payload.pop("frame_id")
    payload.pop("frame_path")
    payload.pop("debug")
    payload.pop("feature_clusters", None)
    for cell in payload.get("per_cell_results", ()) or ():
        cell.pop("feature_cluster_id", None)
        cell.pop("feature_cluster_label", None)
        cell.pop("consistency_score", None)
        cell.pop("consistency_reasons", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate(
    contour_id: int,
    bbox: tuple[int, int, int, int],
    *,
    area: float | None = None,
    fill: float = 0.10,
    interior: float = 0.04,
    center: float = 0.02,
    solidity: float = 0.94,
    extent: float = 0.74,
    vertices: int = 4,
    hole: float = 0.04,
    child_count: int = 1,
    touches_border: bool = False,
):
    x, y, width, height = bbox
    bbox_area = float(width * height)
    return grid_anomaly._ContourCandidate(
        contour_id=int(contour_id),
        contour=None,
        bbox=(int(x), int(y), int(width), int(height)),
        centroid=(float(x + width / 2.0), float(y + height / 2.0)),
        area=float(bbox_area * 0.70 if area is None else area),
        bbox_area=bbox_area,
        aspect_ratio=float(width / max(1, height)),
        extent=float(extent),
        solidity=float(solidity),
        perimeter=float(width * 2 + height * 2),
        approx_vertices=int(vertices),
        fill_ratio=float(fill),
        interior_fill_ratio=float(interior),
        center_fill_ratio=float(center),
        outline_min_side_coverage=0.72,
        outline_mean_side_coverage=0.78,
        outline_side_imbalance=0.08,
        inner_hole_ratio=float(hole),
        child_count=int(child_count),
        touches_border=bool(touches_border),
    )


def test_defect_feature_clusters_assign_human_labels() -> None:
    cells = [
        grid_anomaly.GridCellAnalysisResult(0, 0, (0, 0, 10, 10), (5.0, 5.0), 1, "broken", 0.92, ("filled_cell",)),
        grid_anomaly.GridCellAnalysisResult(1, 0, (20, 0, 3, 3), (21.5, 1.5), 2, "artifact", 0.84, ("small_artifact",)),
        grid_anomaly.GridCellAnalysisResult(
            2, 0, (0, 20, 10, 10), (5.0, 25.0), 3, "broken", 0.88, ("broken_geometry",)
        ),
        grid_anomaly.GridCellAnalysisResult(
            3, 0, (20, 20, 22, 10), (31.0, 25.0), 4, "broken", 0.90, ("merged_contour",)
        ),
    ]
    entries = [
        (0, _candidate(1, (0, 0, 10, 10), fill=0.72, interior=0.54, center=0.42)),
        (1, _candidate(2, (20, 0, 3, 3), area=6.0, fill=0.28, interior=0.02, center=0.01, child_count=0)),
        (2, _candidate(3, (0, 20, 10, 10), fill=0.12, solidity=0.58, extent=0.46, vertices=17)),
        (3, _candidate(4, (20, 20, 22, 10), area=180.0, fill=0.22, interior=0.12, vertices=7)),
    ]

    clusters, by_cell = grid_anomaly._cluster_defective_cell_features(
        cells,
        entries,
        median_width=10.0,
        median_height=10.0,
        median_area=70.0,
        max_clusters=4,
    )

    labels = {cluster.label for cluster in clusters}
    assert labels == {"filled_like", "debris_like", "broken_shape", "merged_like"}
    assert {label for _cluster_id, label in by_cell.values()} == labels


@pytest.mark.parametrize("representation", ("confidence", "binary"))
def test_large_edge_conductor_residue_is_not_classified_as_merged_cell(representation: str) -> None:
    candidate = _candidate(
        10,
        (0, 30, 45, 14),
        area=285.0,
        fill=0.58,
        interior=0.46,
        center=0.40,
        solidity=0.72,
        extent=0.45,
        vertices=11,
        child_count=0,
        touches_border=True,
    )

    score, reasons = grid_anomaly._classify_detected_cell(
        candidate,
        median_width=10.0,
        median_height=10.0,
        median_area=70.0,
        median_fill=0.12 if representation == "confidence" else 0.88,
        median_interior_fill=0.04 if representation == "confidence" else 0.92,
        median_center_fill=0.02 if representation == "confidence" else 0.96,
        median_aspect=1.0,
        config=grid_anomaly.GridDamageAnalysisConfig(cell_representation=representation),
    )

    assert score >= 0.80
    assert reasons == ("conductor_residue",)
    assert grid_anomaly._status_for_reasons(reasons) == "artifact"


def _synthetic_border_merged_cells() -> np.ndarray:
    image = np.full((170, 130), 140, dtype=np.uint8)
    row_positions = (10, 42, 74, 106, 138)
    for x in (10, 68):
        for y in row_positions:
            cv2.rectangle(image, (x, y), (x + 20, y + 20), 238, 2)
    for y in row_positions:
        cv2.rectangle(image, (112, y), (142, y + 20), 238, 2)
    cv2.line(image, (128, 61), (128, 75), 238, 3)
    return image


def test_clipped_merged_cells_at_frame_border_are_marked_as_merged() -> None:
    result = detect_grid_cell_anomalies(
        _synthetic_border_merged_cells(),
        config=grid_anomaly.GridDamageAnalysisConfig(blur_radius=1),
    )

    merged = [cell for cell in result.cells if "merged_contour" in cell.reasons]
    assert result.grid_detected
    assert len(merged) == 1
    assert merged[0].bbox[0] + merged[0].bbox[2] == result.image_width
    assert "conductor_residue" not in merged[0].reasons


def _synthetic_open_outline_column() -> np.ndarray:
    image = np.full((520, 180), 140, dtype=np.uint8)
    for index in range(10):
        x, y, size = 70, 18 + index * 48, 32
        if index in {0, 9}:
            cv2.line(image, (x, y + 5), (x, y + size - 5), 238, 8)
            cv2.line(image, (x + 7, y), (x + size, y), 238, 2)
            cv2.line(image, (x + size, y), (x + size, y + size), 238, 2)
            cv2.line(image, (x + size, y + size), (x + 7, y + size), 238, 2)
        else:
            cv2.rectangle(image, (x, y), (x + size, y + size), 238, 2)
    return image


def test_detached_open_cell_edge_is_not_small_artifact_in_confidence_mode() -> None:
    result = detect_grid_cell_anomalies(_synthetic_open_outline_column())

    assert result.grid_detected
    assert result.bad_cells == 0
    assert not any("small_artifact" in cell.reasons for cell in result.cells)


def test_standalone_clean_fragment_remains_small_artifact() -> None:
    fragment = _candidate(
        1,
        (0, 0, 3, 9),
        area=15.0,
        fill=0.88,
        interior=1.0,
        center=1.0,
        solidity=0.98,
        extent=0.77,
        vertices=6,
        child_count=0,
    )

    assert not grid_anomaly._is_detached_confidence_cell_edge(
        fragment,
        [fragment],
        median_width=10.0,
        median_height=10.0,
        median_area=70.0,
        config=grid_anomaly.GridDamageAnalysisConfig(cell_representation="confidence"),
    )


def _synthetic_binary_grid_frame(*, missing_index: int | None = None) -> np.ndarray:
    height, width = 768, 1024
    rows, cols = 12, 16
    cell_w = width // (cols + 2)
    cell_h = height // (rows + 2)
    margin_x = (width - cols * cell_w) // 2
    margin_y = (height - rows * cell_h) // 2
    image = np.zeros((height, width), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if missing_index == row * cols + col:
                continue
            x = margin_x + col * cell_w
            y = margin_y + row * cell_h
            cv2.rectangle(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 255, -1)
    return image


@pytest.mark.parametrize("invert", (False, True))
def test_binary_representation_accepts_filled_rectangular_cells(invert: bool) -> None:
    image = _synthetic_binary_grid_frame()
    if invert:
        image = cv2.bitwise_not(image)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(cell_representation="binary"),
    )

    assert result.grid_detected
    assert result.detected_cells == 12 * 16
    assert result.bad_cells == 0
    assert not any("filled_cell" in cell.reasons or "partial_filled_cell" in cell.reasons for cell in result.cells)


def test_binary_standalone_flags_mid_dirt_as_suspicious_without_reference() -> None:
    image = _synthetic_binary_grid_frame()
    # Mid-size inaccurate blob (~0.25 of a typical synthetic cell ~50px).
    cv2.rectangle(image, (40, 40), (64, 64), 255, -1)
    # Speckles must be ignored.
    cv2.rectangle(image, (512, 384), (514, 386), 255, -1)
    cv2.rectangle(image, (900, 600), (902, 601), 255, -1)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=3.0,
            min_cell_size=2,
            blur_radius=1,
        ),
    )
    assert result.grid_detected
    dirt = [cell for cell in result.cells if "small_artifact" in cell.reasons]
    assert dirt
    assert all(cell.status == "suspicious" for cell in dirt)
    assert result.detected_cells >= 12 * 16


def test_confidence_standalone_flags_mid_debris_as_suspicious() -> None:
    image = np.zeros((768, 1024), dtype=np.uint8)
    rows, cols = 8, 10
    cell_w = 1024 // (cols + 2)
    cell_h = 768 // (rows + 2)
    margin_x = (1024 - cols * cell_w) // 2
    margin_y = (768 - rows * cell_h) // 2
    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * cell_w
            y = margin_y + r * cell_h
            cv2.rectangle(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 255, 2)
    # Mid inaccurate blob vs tiny speck (placed in frame margin so contours stay separate).
    cv2.rectangle(image, (10, 10), (34, 34), 255, -1)
    cv2.rectangle(image, (40, 12), (42, 14), 255, -1)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="confidence",
            min_contour_area=3.0,
            min_cell_size=2,
            blur_radius=1,
        ),
    )
    assert result.grid_detected
    dirt = [cell for cell in result.cells if "small_artifact" in cell.reasons]
    assert dirt
    assert any(cell.status == "suspicious" for cell in dirt)
    assert any(cell.status == "normal" for cell in result.cells)


def test_speckle_dirt_is_ignored_on_binary_grid() -> None:
    image = _synthetic_binary_grid_frame()
    cv2.rectangle(image, (40, 40), (42, 41), 255, -1)
    cv2.rectangle(image, (900, 600), (901, 601), 255, -1)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=3.0,
            min_cell_size=2,
            blur_radius=1,
        ),
    )
    assert result.grid_detected
    assert not any("small_artifact" in cell.reasons for cell in result.cells)


def test_grainy_mass_without_cells_is_one_conductor_not_many_merged() -> None:
    image = np.zeros((256, 256), dtype=np.uint8)
    rng = np.random.default_rng(0)
    # Dense grainy vertical band — no regular cell grid.
    band = rng.random((220, 70)) > 0.45
    image[18:238, 20:90] = np.where(band, 255, 0).astype(np.uint8)
    # Speckles in empty space.
    image[200, 200] = 255
    image[210, 210] = 255
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=4.0,
            min_cell_size=2,
            blur_radius=1,
            min_grid_candidate_cells=6,
        ),
    )
    conductors = [cell for cell in result.cells if "conductor_residue" in cell.reasons]
    merged = [cell for cell in result.cells if "merged_contour" in cell.reasons]
    assert conductors
    assert len(conductors) <= 3
    assert not merged
    assert max(cell.bbox[2] * cell.bbox[3] for cell in conductors) >= 70 * 100


def test_collapse_adjacent_conductors_into_one_rectangle() -> None:
    cells = [
        grid_anomaly.GridCellAnalysisResult(
            0, 0, (10, 10, 20, 18), (20.0, 19.0), 1, "artifact", 0.9, ("conductor_residue",)
        ),
        grid_anomaly.GridCellAnalysisResult(
            1, 0, (28, 12, 22, 16), (39.0, 20.0), 2, "artifact", 0.88, ("conductor_residue",)
        ),
        grid_anomaly.GridCellAnalysisResult(
            2, 0, (100, 100, 12, 12), (106.0, 106.0), 3, "normal", 0.0, ()
        ),
    ]
    collapsed = grid_anomaly._collapse_conductor_regions(cells, median_width=12.0, median_height=12.0)
    conductors = [cell for cell in collapsed if "conductor_residue" in cell.reasons]
    assert len(conductors) == 1
    assert conductors[0].bbox[2] >= 40
    assert any(cell.status == "normal" for cell in collapsed)


def test_binary_standalone_flags_wrong_aspect_geometry() -> None:
    image = _synthetic_binary_grid_frame()
    # Replace one square cell with a clearly elongated slot-sized blob.
    cell_w = 1024 // (16 + 2)
    cell_h = 768 // (12 + 2)
    margin_x = (1024 - 16 * cell_w) // 2
    margin_y = (768 - 12 * cell_h) // 2
    x = margin_x + 3 * cell_w
    y = margin_y + 4 * cell_h
    cv2.rectangle(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 0, -1)
    # Horizontal bar spanning nearly full cell width but only ~1/3 height.
    cv2.rectangle(image, (x + 2, y + cell_h // 3), (x + cell_w - 3, y + 2 * cell_h // 3), 255, -1)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=4.0,
            min_cell_size=3,
            blur_radius=1,
        ),
    )
    assert result.grid_detected
    bad_reasons = [cell.reasons for cell in result.cells if cell.reasons]
    assert any(
        "broken_geometry" in reasons or "small_artifact" in reasons for reasons in bad_reasons
    ), bad_reasons


def test_binary_border_crop_is_edge_clipped_not_broken_geometry() -> None:
    image = _synthetic_binary_grid_frame()
    cell_w = 1024 // (16 + 2)
    cell_h = 768 // (12 + 2)
    margin_x = (1024 - 16 * cell_w) // 2
    # Half-height cells glued to the top frame edge (typical crop).
    for col in range(4, 10):
        x = margin_x + col * cell_w
        cv2.rectangle(image, (x + 2, 0), (x + cell_w - 3, cell_h // 2), 255, -1)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=4.0,
            min_cell_size=3,
            blur_radius=1,
        ),
    )
    clipped = [cell for cell in result.cells if "edge_clipped_cell" in cell.reasons]
    assert clipped
    assert all("broken_geometry" not in cell.reasons for cell in clipped)
    from karakal.core.exports import _grid_defect_reason_rgb
    from karakal.ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_COLORS

    rgb = _grid_defect_reason_rgb(clipped[0].reasons, status=clipped[0].status)
    expected = GRID_INSPECTION_ERROR_TYPE_COLORS["edge_clipped_cell"].lstrip("#")
    assert rgb == (int(expected[0:2], 16), int(expected[2:4], 16), int(expected[4:6], 16))


def test_binary_irregular_mid_cell_is_broken_geometry() -> None:
    image = _synthetic_binary_grid_frame()
    cell_w = 1024 // (16 + 2)
    cell_h = 768 // (12 + 2)
    margin_x = (1024 - 16 * cell_w) // 2
    margin_y = (768 - 12 * cell_h) // 2
    x = margin_x + 7 * cell_w
    y = margin_y + 5 * cell_h
    cv2.rectangle(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 0, -1)
    # C-shaped / bitten occupant: clearly wrong geometry, still near slot size.
    pts = np.array(
        [
            [x + 4, y + 4],
            [x + cell_w - 6, y + 4],
            [x + cell_w - 6, y + 14],
            [x + 14, y + 14],
            [x + 14, y + cell_h - 16],
            [x + cell_w - 6, y + cell_h - 16],
            [x + cell_w - 6, y + cell_h - 6],
            [x + 4, y + cell_h - 6],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [pts], 255)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=4.0,
            min_cell_size=3,
            blur_radius=1,
        ),
    )
    assert any("broken_geometry" in cell.reasons for cell in result.cells), [
        cell.reasons for cell in result.cells if cell.reasons
    ]


def test_binary_small_but_significant_dirt_is_flagged() -> None:
    image = _synthetic_binary_grid_frame()
    cell_w = 1024 // (16 + 2)
    cell_h = 768 // (12 + 2)
    # ~12% of a slot — below the old mid-dirt size, above speckles.
    dirt = max(8, int(round(min(cell_w, cell_h) * 0.28)))
    cv2.rectangle(image, (36, 36), (36 + dirt, 36 + dirt), 255, -1)
    # True speckles must stay ignored.
    cv2.rectangle(image, (900, 600), (901, 601), 255, -1)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=3.0,
            min_cell_size=2,
            blur_radius=1,
        ),
    )
    dirt_cells = [cell for cell in result.cells if "small_artifact" in cell.reasons]
    assert dirt_cells
    assert all(cell.status == "suspicious" for cell in dirt_cells)


def test_binary_grainy_corner_dirt_is_flagged_despite_otsu_split() -> None:
    """Dim/grainy corner debris must survive Otsu fragmentation (critical FP miss)."""

    image = _synthetic_binary_grid_frame()
    rng = np.random.default_rng(7)
    # Compact grainy speck near the bottom-right (~critical dirt size), dimmer than cells.
    ys, xs = np.mgrid[738:750, 990:1004]
    mask = rng.random(ys.shape) > 0.35
    image[ys, xs] = np.where(mask, rng.integers(140, 200, size=ys.shape, dtype=np.uint8), image[ys, xs])
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=3.0,
            min_cell_size=2,
            blur_radius=1,
        ),
    )
    dirt = [
        cell
        for cell in result.cells
        if "small_artifact" in cell.reasons and cell.bbox[0] >= 970 and cell.bbox[1] >= 720
    ]
    assert dirt, [(cell.bbox, cell.reasons) for cell in result.cells if cell.reasons]


def test_binary_standalone_flags_stretched_slot_cell() -> None:
    image = _synthetic_binary_grid_frame()
    cell_w = 1024 // (16 + 2)
    cell_h = 768 // (12 + 2)
    margin_x = (1024 - 16 * cell_w) // 2
    margin_y = (768 - 12 * cell_h) // 2
    x = margin_x + 5 * cell_w
    y = margin_y + 3 * cell_h
    # Stretch a normal cell into a wide rectangle (wrong aspect, still slot-sized).
    cv2.rectangle(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 0, -1)
    cv2.rectangle(image, (x - cell_w // 4, y + 8), (x + cell_w + cell_w // 4, y + cell_h - 9), 255, -1)
    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(
            cell_representation="binary",
            min_contour_area=4.0,
            min_cell_size=3,
            blur_radius=1,
        ),
    )
    assert result.grid_detected
    assert any("broken_geometry" in cell.reasons or "merged_contour" in cell.reasons for cell in result.cells)


def test_confidence_binary_comparison_finds_missing_binary_cell() -> None:
    confidence = detect_grid_cell_anomalies(_synthetic_grid_frame(0))
    binary = detect_grid_cell_anomalies(
        _synthetic_binary_grid_frame(missing_index=17),
        config=grid_anomaly.GridDamageAnalysisConfig(cell_representation="binary"),
    )

    comparison = grid_anomaly.compare_grid_cell_analyses(confidence, binary)

    assert comparison.grid_detected
    assert any("confidence_only_cell" in cell.reasons for cell in comparison.cells)


@pytest.mark.parametrize(("variant", "expected_digest"), enumerate(GOLDEN_DIGESTS))
def test_grid_analysis_matches_pre_optimization_golden_result(variant: int, expected_digest: str) -> None:
    result = detect_grid_cell_anomalies(_synthetic_grid_frame(variant))

    assert _result_digest(result) == expected_digest


def test_outline_side_coverages_matches_numpy_reference() -> None:
    rng = np.random.default_rng(20260720)
    for height, width in ((4, 4), (17, 31), (64, 53)):
        roi = (rng.random((height, width)) > 0.72).astype(np.uint8) * 255
        band = max(1, int(round(min(width, height) * 0.18)))
        band = min(band, max(1, width), max(1, height))
        expected = (
            float(np.mean(np.any(roi[:band, :] > 0, axis=0))),
            float(np.mean(np.any(roi[height - band :, :] > 0, axis=0))),
            float(np.mean(np.any(roi[:, :band] > 0, axis=1))),
            float(np.mean(np.any(roi[:, width - band :] > 0, axis=1))),
        )

        assert _outline_side_coverages(roi) == expected


def test_grid_analysis_ignores_thin_cell_edge_fragments() -> None:
    image = np.zeros((360, 520), dtype=np.uint8)
    rows, cols = 6, 8
    cell_w, cell_h = 42, 38
    margin_x, margin_y = 70, 55
    for row in range(rows):
        for col in range(cols):
            x = margin_x + col * cell_w
            y = margin_y + row * cell_h
            cv2.rectangle(image, (x + 4, y + 4), (x + cell_w - 5, y + cell_h - 5), 255, 2)

    cv2.rectangle(image, (margin_x + cell_w - 1, margin_y + 5), (margin_x + cell_w + 3, margin_y + cell_h - 6), 255, -1)
    cv2.line(
        image,
        (margin_x + 3 * cell_w + 2, margin_y + 3 * cell_h + 2),
        (margin_x + 3 * cell_w + 2, margin_y + 4 * cell_h - 8),
        255,
        2,
    )

    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(min_contour_area=4.0, min_cell_size=2),
    )

    assert result.grid_detected
    assert not any("small_artifact" in cell.reasons for cell in result.cells)


def test_process_and_sequential_workers_produce_identical_results(tmp_path, monkeypatch) -> None:
    process_records = []
    sequential_records = []
    for index in range(12):
        image = _synthetic_grid_frame(index % 8, shape=(384, 512))
        process_path = tmp_path / f"process_{index:02d}.png"
        sequential_path = tmp_path / f"sequential_{index:02d}.png"
        assert cv2.imwrite(str(process_path), image)
        assert cv2.imwrite(str(sequential_path), image)
        process_records.append(FrameRecord(f"process_{index:02d}", process_path.name, first_path=str(process_path)))
        sequential_records.append(FrameRecord(f"sequential_{index:02d}", sequential_path.name, first_path=str(sequential_path)))

    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_WORKERS", "2")
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_CHUNK_SIZE", "4")
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_OPENCV_THREADS", "1")

    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "process")
    process_finished = []
    process_failed = []
    process_batches = []
    process_worker = GridInspectionWorker(
        process_records,
        use_cache=False,
        performance_config=PerformanceConfig(
            cpu_workers=2,
            batch_size=4,
            profiling_mode=ProfilingMode.SUMMARY,
            profiling_output_directory=str(tmp_path / "profiles"),
        ),
    )
    process_worker.finished.connect(process_finished.append)
    process_worker.failed.connect(process_failed.append)
    process_worker.partialResultsReady.connect(process_batches.append)
    process_worker.run()

    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "sequential")
    sequential_finished = []
    sequential_worker = GridInspectionWorker(sequential_records, use_cache=False)
    sequential_worker.finished.connect(sequential_finished.append)
    sequential_worker.run()

    assert process_failed == []
    assert len(process_finished) == 1
    assert len(sequential_finished) == 1
    assert 1 <= len(process_batches) <= 3
    process_digests = sorted(_result_digest(result) for result in process_finished[0].values())
    sequential_digests = sorted(_result_digest(result) for result in sequential_finished[0].values())
    assert process_digests == sequential_digests
    snapshot = process_worker.profiler.snapshot()
    assert snapshot.workers
    assert any(row["name"] == "validation.grid.worker_batch" for row in snapshot.stages)


def test_paired_grid_worker_returns_three_linked_matrices(tmp_path, monkeypatch) -> None:
    confidence_path = tmp_path / "confidence.png"
    binary_path = tmp_path / "binary.png"
    assert cv2.imwrite(str(confidence_path), _synthetic_grid_frame(0))
    assert cv2.imwrite(str(binary_path), _synthetic_binary_grid_frame(missing_index=17))
    record = FrameRecord(
        "frame",
        "frame.png",
        model_mask_paths={"model": str(binary_path)},
        model_prob_paths={"model": str(confidence_path)},
    )
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "sequential")
    worker = workers_module.PairedGridInspectionWorker([record], "model", use_cache=False)
    finished = []
    failed = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert failed == []
    assert len(finished) == 1
    assert set(finished[0]["frame"]) == {"confidence", "binary", "comparison"}
    assert any(
        "confidence_only_cell" in cell.reasons
        for cell in finished[0]["frame"]["comparison"].cells
    )


@pytest.mark.parametrize(
    ("source_kind", "expected_layer"),
    (
        ("binary", "binary"),
        ("untyped_confidence", "confidence"),
        ("explicit_confidence", "confidence"),
    ),
)
def test_grid_worker_supports_each_single_source_mode(tmp_path, monkeypatch, source_kind: str, expected_layer: str) -> None:
    source_path = tmp_path / f"{source_kind}.png"
    image = _synthetic_binary_grid_frame() if source_kind == "binary" else _synthetic_grid_frame(0)
    assert cv2.imwrite(str(source_path), image)
    mask_paths = {"model": str(source_path)} if source_kind != "explicit_confidence" else {}
    confidence_paths = {"model": str(source_path)} if source_kind == "explicit_confidence" else {}
    record = FrameRecord(
        "frame",
        "frame.png",
        model_mask_paths=mask_paths,
        model_prob_paths=confidence_paths,
    )
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "sequential")
    worker = workers_module.PairedGridInspectionWorker([record], "model", use_cache=False)
    finished = []
    failed = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert failed == []
    assert len(finished) == 1
    assert set(finished[0]["frame"]) == {expected_layer}
    assert finished[0]["frame"][expected_layer].grid_detected


@pytest.mark.parametrize("dataset_layer", ("confidence", "binary"))
def test_single_source_representation_is_selected_once_for_the_whole_dataset(
    tmp_path,
    monkeypatch,
    dataset_layer: str,
) -> None:
    records = []
    for index in range(7):
        path = tmp_path / f"{dataset_layer}_{index}.png"
        if dataset_layer == "confidence":
            image = _synthetic_grid_frame(0) if index < 6 else _synthetic_binary_grid_frame()
        else:
            image = _synthetic_binary_grid_frame() if index < 6 else _synthetic_grid_frame(0)
        assert cv2.imwrite(str(path), image)
        records.append(
            FrameRecord(
                f"frame_{index}",
                path.name,
                model_mask_paths={"model": str(path)},
            )
        )
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "sequential")
    worker = workers_module.PairedGridInspectionWorker(records, "model", use_cache=False)
    finished = []
    failed = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert failed == []
    assert len(finished) == 1
    assert len(finished[0]) == len(records)
    assert all(set(payload) == {dataset_layer} for payload in finished[0].values())


def test_grid_worker_cancelled_before_start_emits_cancelled(monkeypatch) -> None:
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "sequential")
    worker = GridInspectionWorker([], use_cache=False)
    cancelled = []
    finished = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.finished.connect(finished.append)

    worker.request_cancel()
    worker.run()

    assert cancelled == [True]
    assert finished == []


def test_compact_grid_cache_pickle_roundtrip_is_exact() -> None:
    result = detect_grid_cell_anomalies(_synthetic_grid_frame(3), frame_id="pickle")

    restored = pickle.loads(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))

    assert restored == result


def test_grid_worker_exposes_grouped_decode_errors(tmp_path, monkeypatch, caplog) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    record = FrameRecord("broken", corrupt.name, first_path=str(corrupt))
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "sequential")
    worker = GridInspectionWorker([record], use_cache=False)
    reported = []
    worker.analysisErrors.connect(reported.append)

    with caplog.at_level("WARNING"):
        worker.run()

    assert reported == [{"broken": "decode_error"}]
    assert "Grid batch completed with errors" in caplog.text
    assert worker.profiler.snapshot().counters == {}


def test_grid_worker_coalesces_partial_results_for_ui() -> None:
    worker = GridInspectionWorker([], use_cache=False)
    batches = []
    worker.partialResultsReady.connect(batches.append)

    for index in range(600):
        worker._queue_partial_results({str(index): index})
    worker._flush_partial_results()

    assert [len(batch) for batch in batches] == [256, 256, 88]
    assert set().union(*(set(batch) for batch in batches)) == {str(index) for index in range(600)}


def test_identical_grid_payload_does_not_rebuild_scene(monkeypatch) -> None:
    _app = QApplication.instance() or QApplication([])
    view = MatrixListWidget()
    view.set_grid_inspection_payloads({}, enabled=True)

    def fail_refresh() -> None:
        raise AssertionError("unchanged grid payload rebuilt the scene")

    monkeypatch.setattr(view, "refresh_scene", fail_refresh)
    view.set_grid_inspection_payloads({}, enabled=True)
    view.close()


def test_grid_worker_falls_back_when_process_pool_cannot_start(tmp_path, monkeypatch) -> None:
    records = []
    for index in range(4):
        path = tmp_path / f"fallback_{index}.png"
        assert cv2.imwrite(str(path), _synthetic_grid_frame(index, shape=(384, 512)))
        records.append(FrameRecord(f"fallback_{index}", path.name, first_path=str(path)))

    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "process")
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_CHUNK_SIZE", "2")

    class BrokenExecutor:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("process pool unavailable")

    monkeypatch.setattr(workers_module, "ProcessPoolExecutor", BrokenExecutor)
    finished = []
    failed = []
    worker = GridInspectionWorker(records, use_cache=False)
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert failed == []
    assert len(finished) == 1
    assert set(finished[0]) == {record.key for record in records}


def test_grid_cache_hit_corruption_and_file_change(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), _synthetic_grid_frame(1, shape=(384, 512)))
    monkeypatch.setattr(grid_anomaly, "GRID_DAMAGE_CACHE_DIR", cache_dir)

    first = grid_anomaly.analyze_grid_frame_path(path, frame_id="frame", use_cache=True)
    assert first is not None
    cache_files = list(cache_dir.glob("*.pickle"))
    assert len(cache_files) == 1

    original_analyze = grid_anomaly.detect_grid_cell_anomalies

    def fail_if_reanalyzed(*_args, **_kwargs):
        raise AssertionError("cache hit unexpectedly re-ran analysis")

    monkeypatch.setattr(grid_anomaly, "detect_grid_cell_anomalies", fail_if_reanalyzed)
    cached = grid_anomaly.analyze_grid_frame_path(path, frame_id="frame", use_cache=True)
    assert cached == first

    cache_files[0].write_bytes(b"not a pickle")
    monkeypatch.setattr(grid_anomaly, "detect_grid_cell_anomalies", original_analyze)
    recovered = grid_anomaly.analyze_grid_frame_path(path, frame_id="frame", use_cache=True)
    assert recovered == first

    assert cv2.imwrite(str(path), _synthetic_grid_frame(3, shape=(384, 512)))
    changed = grid_anomaly.analyze_grid_frame_path(path, frame_id="frame", use_cache=True)
    assert changed is not None
    assert _result_digest(changed) != _result_digest(first)
    assert len(list(cache_dir.glob("*.pickle"))) == 2


def test_grid_chunk_keeps_valid_results_when_one_image_is_corrupt(tmp_path) -> None:
    valid_path = tmp_path / "valid.png"
    corrupt_path = tmp_path / "corrupt.png"
    assert cv2.imwrite(str(valid_path), _synthetic_grid_frame(2, shape=(384, 512)))
    corrupt_path.write_bytes(b"not an image")

    payloads, errors = grid_anomaly.analyze_grid_frame_chunk(
        (("valid", str(valid_path)), ("corrupt", str(corrupt_path))),
        grid_anomaly.GridDamageAnalysisConfig(),
        use_cache=False,
    )

    assert set(payloads) == {"valid"}
    assert errors == {"corrupt": "decode_error"}


@pytest.mark.parametrize("chunk_size", (1, 2, 5, 16))
def test_chunk_partition_does_not_change_results(chunk_size: int, tmp_path) -> None:
    entries = []
    for index in range(5):
        path = tmp_path / f"frame_{index}.png"
        assert cv2.imwrite(str(path), _synthetic_grid_frame(index, shape=(384, 512)))
        entries.append((f"frame_{index}", str(path)))
    entries_tuple = tuple(entries)
    expected, expected_errors = grid_anomaly.analyze_grid_frame_chunk(
        entries_tuple,
        grid_anomaly.GridDamageAnalysisConfig(),
        use_cache=False,
    )

    actual = {}
    actual_errors = {}
    for offset in range(0, len(entries_tuple), chunk_size):
        payloads, errors = grid_anomaly.analyze_grid_frame_chunk(
            entries_tuple[offset : offset + chunk_size],
            grid_anomaly.GridDamageAnalysisConfig(),
            use_cache=False,
        )
        actual.update(payloads)
        actual_errors.update(errors)

    assert actual_errors == expected_errors
    assert {key: _result_digest(value) for key, value in actual.items()} == {
        key: _result_digest(value) for key, value in expected.items()
    }
