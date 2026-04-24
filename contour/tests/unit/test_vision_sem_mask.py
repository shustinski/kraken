from __future__ import annotations

import numpy as np

from polygon_widget.vision.contour_extraction.hierarchy import build_hierarchy_from_mask
from polygon_widget.vision.contour_extraction.sem_filled_mask import (
    FilledMaskSegmentationConfig,
    extract_filled_mask,
)
from polygon_widget.vision.integration import contour_output_to_polygons, run_contour_filled_mask
from polygon_widget.vision.preprocessing import PreprocessConfig
from polygon_widget.vision.schemas import AppMode, OutputShapeKind, SemPolarity


def _synthetic() -> np.ndarray:
    h, s = 200, 200
    g = np.full((h, s), 180, dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.rectangle(g, (60, 60), (140, 120), 40, -1)
    cv2.rectangle(g, (85, 80), (115, 100), 200, -1)  # hole
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return g


def test_extract_filled_mask_runs() -> None:
    g = _synthetic()
    cfg = FilledMaskSegmentationConfig(min_component_area=5)
    prep = PreprocessConfig()
    r = extract_filled_mask(g, config=cfg, preprocess=prep, polarity=SemPolarity.DARK_FOREGROUND)
    assert r.mask.shape == g.shape
    assert r.mask.dtype == np.uint8
    assert 255 in r.mask
    _contours, components = build_hierarchy_from_mask(r.mask, epsilon=1.0)
    assert len(components) >= 1


def test_contour_extraction_output_json() -> None:
    from polygon_widget.vision.io_normalize import make_image_ref
    from polygon_widget.vision.schemas import ContourExtractionOutput, HierarchicalComponent

    c = HierarchicalComponent(
        id=1,
        contour_index=0,
        parent_id=None,
        depth=0,
        is_hole=False,
        points=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        area=0.5,
        bbox_xywh=(0, 0, 1, 1),
    )
    out = ContourExtractionOutput(
        image=make_image_ref("x.tif", np.zeros((10, 10), np.uint8)),
        mode=AppMode.CONTOUR,
        output_kind=OutputShapeKind.POLYGON,
        filled_mask=np.zeros((10, 10), np.uint8),
        components=[c],
        strategy_used="adaptive_mean",
    )
    d = out.to_json_dict()
    assert d["components"][0]["id"] == 1


def test_sem_contour_backend_converts_to_box_polygons() -> None:
    g = _synthetic()
    out = run_contour_filled_mask(
        g,
        image_path="synthetic.png",
        output_kind=OutputShapeKind.AXIS_ALIGNED_BOX,
        noise_level="medium",
    )
    polygons = contour_output_to_polygons(out)

    assert out.filled_mask.shape == g.shape
    assert polygons
    assert all(polygon.shape_hint == "box" for polygon in polygons)
    assert all(len(polygon.points) == 4 for polygon in polygons)


def test_sem_contour_backend_preserves_hierarchy_metadata() -> None:
    g = _synthetic()
    out = run_contour_filled_mask(
        g,
        image_path="synthetic.png",
        output_kind=OutputShapeKind.POLYGON,
        noise_level="medium",
    )
    polygons = contour_output_to_polygons(out)

    assert polygons
    assert any(polygon.parent_id is not None for polygon in polygons) or any(polygon.is_hole for polygon in polygons)
