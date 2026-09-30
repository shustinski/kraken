"""The conductor zone comes from the binary mask and is shared by both layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from karakal.core.grid_anomaly import (
    GridDamageAnalysisConfig,
    analyze_grid_frame_pair_chunk,
    detect_grid_cell_anomalies,
)

pytest.importorskip("cv2")
import cv2  # noqa: E402


def _lattice(image: np.ndarray, *, rows: int = 6, cols: int = 8, origin: tuple[int, int] = (24, 180)) -> None:
    x0, y0 = origin
    for row in range(rows):
        for col in range(cols):
            x = x0 + col * 36
            y = y0 + row * 48
            cv2.rectangle(image, (x, y), (x + 22, y + 32), 255, -1)


def _config(**overrides) -> GridDamageAnalysisConfig:
    values = dict(
        scoring_mode="calibrated",
        blur_radius=1,
        min_contour_area=8.0,
        min_cell_size=2,
        fill_sensitivity=60,
        debris_sensitivity=75,
        geometry_sensitivity=40,
        merge_sensitivity=35,
        min_grid_candidate_cells=4,
    )
    values.update(overrides)
    return GridDamageAnalysisConfig(**values)


def _zones(result):
    return [cell for cell in result.cells if "conductor_zone" in cell.reasons]


def _defects(result):
    return [cell for cell in result.cells if cell.reasons and "conductor_zone" not in cell.reasons]


def _grain_band(image: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    for _ in range(80):
        x = int(rng.integers(10, 380))
        y = int(rng.integers(8, 140))
        cv2.rectangle(image, (x, y), (x + int(rng.integers(3, 9)), y + int(rng.integers(3, 8))), 255, -1)


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def test_solid_array_has_no_zone_on_either_layer(tmp_path: Path) -> None:
    binary = np.zeros((520, 420), dtype=np.uint8)
    _lattice(binary, origin=(24, 24))
    confidence = np.full(binary.shape, 255, dtype=np.uint8)
    confidence[binary == 0] = 0
    binary_path = tmp_path / "mask.png"
    confidence_path = tmp_path / "confidence.png"
    _write(binary_path, binary)
    _write(confidence_path, confidence)

    payloads, errors = analyze_grid_frame_pair_chunk(
        (("frame", str(confidence_path), str(binary_path)),),
        _config(),
        use_cache=False,
    )

    assert errors == {}
    for layer in ("binary", "confidence"):
        result = payloads["frame"][layer]
        assert result.model_file_status == ""
        assert _zones(result) == []


def test_grain_band_uses_the_same_zone_on_both_layers(tmp_path: Path) -> None:
    binary = np.zeros((520, 420), dtype=np.uint8)
    _lattice(binary)
    _grain_band(binary)
    confidence = np.full(binary.shape, 40, dtype=np.uint8)
    confidence[::12, :] = 220
    confidence[:, ::12] = 220
    binary_path = tmp_path / "mask.png"
    confidence_path = tmp_path / "confidence.png"
    _write(binary_path, binary)
    _write(confidence_path, confidence)

    payloads, errors = analyze_grid_frame_pair_chunk(
        (("frame", str(confidence_path), str(binary_path)),),
        _config(),
        use_cache=False,
    )

    assert errors == {}
    binary_zones = _zones(payloads["frame"]["binary"])
    confidence_zones = _zones(payloads["frame"]["confidence"])
    assert binary_zones
    assert [zone.bbox for zone in confidence_zones] == [zone.bbox for zone in binary_zones]
    alone = detect_grid_cell_anomalies(
        confidence,
        config=_config(cell_representation="confidence"),
    )
    assert _zones(alone) == []


def test_missing_mask_file_is_marked_and_not_analyzed(tmp_path: Path) -> None:
    confidence = np.full((80, 80), 255, dtype=np.uint8)
    confidence_path = tmp_path / "confidence.png"
    _write(confidence_path, confidence)

    payloads, errors = analyze_grid_frame_pair_chunk(
        (("frame", str(confidence_path), str(tmp_path / "missing.png")),),
        _config(),
        use_cache=False,
    )

    assert errors == {}
    for layer in ("binary", "confidence"):
        result = payloads["frame"][layer]
        assert result.model_file_status == "missing"
        assert result.severity_level == "missing_model_file"
        assert result.cells == ()
        assert _zones(result) == []


def test_black_and_jpeg_noise_masks_have_no_zone_or_defects() -> None:
    black = np.zeros((200, 240), dtype=np.uint8)
    noise = black.copy()
    rng = np.random.default_rng(11)
    ys = rng.integers(0, noise.shape[0], size=30)
    xs = rng.integers(0, noise.shape[1], size=30)
    noise[ys, xs] = rng.integers(1, 12, size=30).astype(np.uint8)

    for image in (black, noise):
        result = detect_grid_cell_anomalies(image, config=_config())
        assert result.model_file_status == ""
        assert _zones(result) == []
        assert _defects(result) == []
