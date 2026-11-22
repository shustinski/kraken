"""Speed-ups of a big run (profiled on 2 x 10 000 synthetic frames) keep every result the same."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core import dirt_regions
from karakal.core.cell_error_analysis import _quantile_linear
from karakal.core.dirt_regions import DirtConfig, DirtLayers, dirt_layers
from karakal.core.synthetic_masks import synthetic_frame


def _numpy_dirt_layers(mask: np.ndarray, confidence: np.ndarray | None, config: DirtConfig) -> DirtLayers:
    """dirt_layers as it was written with numpy logic: the reference of the OpenCV version."""

    cfg = config.normalized()
    m = np.asarray(mask, dtype=np.uint8)
    hi = (m >= 128).astype(np.uint8)
    mid = (m > 55) & (m < 200)
    margin = dirt_regions._round_kernel(cfg.blur_margin)
    near_white = cv2.dilate((m >= 200).astype(np.uint8), margin) > 0
    near_black = cv2.dilate((m <= 55).astype(np.uint8), margin) > 0
    blur = mid & ~(near_white & near_black)
    nicks = cv2.morphologyEx(hi, cv2.MORPH_CLOSE, np.ones((cfg.nick_size, cfg.nick_size), np.uint8)) > hi
    crumbs = hi > cv2.morphologyEx(hi, cv2.MORPH_OPEN, np.ones((cfg.crumb_size, cfg.crumb_size), np.uint8))
    if confidence is not None:
        certainty = dirt_regions.confidence_certainty(confidence)
        edge = cv2.morphologyEx(hi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        near_edge = cv2.dilate(edge, dirt_regions._round_kernel(cfg.edge_ring)) > 0
        unsure = (certainty < cfg.unsure_below) & ~near_edge
    else:
        unsure = np.zeros(m.shape, dtype=bool)
    strong = blur | unsure
    dirt = strong.astype(np.float32) + float(cfg.defect_weight) * ((nicks | crumbs) & ~strong).astype(np.float32)
    wx = int(round(cfg.window_cells * cfg.pitch_x)) | 1
    wy = int(round(cfg.window_cells * cfg.pitch_y)) | 1
    density = cv2.boxFilter(dirt, -1, (wx, wy), borderType=cv2.BORDER_REFLECT)
    return DirtLayers(blur=blur, unsure=unsure, nicks=nicks, crumbs=crumbs, density=density)


def _masks():
    rng = np.random.default_rng(5)
    yield rng.integers(0, 256, (300, 420), dtype=np.uint8), rng.integers(0, 256, (300, 420), dtype=np.uint8)
    for seed in range(3):
        frame = synthetic_frame({"bite": 2, "debris": 3, "bridge": 1}, seed=seed).jpeg(70)
        certainty = np.full(frame.mask.shape, 250, np.uint8)
        certainty[40:140, 60:220] = rng.integers(60, 210, (100, 160), dtype=np.uint8)
        yield frame.mask, certainty


@pytest.mark.parametrize("config", [DirtConfig(), DirtConfig(defect_weight=0.3, unsure_below=180, nick_size=7, crumb_size=5)])
def test_dirt_layers_equal_the_numpy_formulas(config) -> None:
    for mask, confidence in _masks():
        for certainty in (confidence, None):
            fast = dirt_layers(mask, certainty, config)
            slow = _numpy_dirt_layers(mask, certainty, config)
            for name in ("blur", "unsure", "nicks", "crumbs", "density"):
                a, b = getattr(fast, name), getattr(slow, name)
                assert a.dtype == b.dtype and a.shape == b.shape, name
                assert np.array_equal(a, b), name


def test_quantile_linear_is_numpy_quantile() -> None:
    rng = np.random.default_rng(1)
    for size in (1, 2, 3, 7, 10, 11, 64, 333):
        for values in (rng.random(size).astype(np.float32), (rng.integers(0, 5, size) / 4).astype(np.float32), rng.random(size)):
            for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                assert _quantile_linear(values.copy(), q) == float(np.quantile(values, q))


def test_sparse_step_runs_once_for_layers_sharing_a_result(monkeypatch) -> None:
    from types import SimpleNamespace

    from karakal.core import workers

    calls = []

    def fake_mark(results, frames, **_kwargs):
        calls.append(dict(results))
        return {key: ("marked", key) for key in results}, {"pitch": (1.0, 1.0)}

    monkeypatch.setattr(workers, "mark_sparse_cells", fake_mark)
    shared = {f"F{index}": object() for index in range(3)}
    payloads = {key: {"binary": value, "confidence": value} for key, value in shared.items()}
    records = [SimpleNamespace(key=key, model_mask_paths={"m": f"{key}.png"}, model_prob_paths={}, identity=None) for key in shared]
    changed = workers.settle_sparse_layer(payloads, records, "m", frames_per_row=3, reach_sites=4)
    assert len(calls) == 1
    assert all(changed[key]["binary"] == changed[key]["confidence"] == ("marked", key) for key in shared)

    # Layers holding different results are each marked on their own.
    calls.clear()
    payloads = {key: {"binary": value, "confidence": object()} for key, value in shared.items()}
    workers.settle_sparse_layer(payloads, records, "m", frames_per_row=3, reach_sites=4)
    assert len(calls) == 2


def test_matrix_takes_a_new_metric_with_slot_records(qtbot) -> None:
    """FrameRecord has slots: changing the metric used to raise while clearing the tile grids, and
    the end of every validation run never reached the matrix."""

    from karakal.core.domain import FrameRecord
    from karakal.ui.matrix_view import MatrixLayoutConfig, MatrixListWidget

    view = MatrixListWidget()
    qtbot.addWidget(view)
    view.set_layout_config(MatrixLayoutConfig(mode="indexed_grid", total_frames=2, frames_per_row=2))
    view.set_records([FrameRecord("1", "1", score=0.4, score_ready=True), FrameRecord("2", "2", score=0.6, score_ready=True)])
    view.set_metric_context("overall_frame_score", point_match_radius=3.0, bce_score_cap=10.0)
    view.set_metric_context("topology_risk", point_match_radius=3.0, bce_score_cap=10.0)


def test_full_size_three_model_comparison_limits_processes(monkeypatch) -> None:
    """2000px masks compared three ways must not start one process per core on 1 GB free."""

    from karakal.core import analytics
    from karakal.core.domain import GeometryMode

    monkeypatch.setattr(analytics.os, "name", "nt")
    monkeypatch.setattr(analytics.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(analytics, "_available_memory_bytes", lambda: 1 * 1024**3)
    monkeypatch.setattr(analytics, "_analytics_worker_env_override", lambda: None)
    common = dict(
        geometry_mode=GeometryMode.MASK,
        analysis_max_side=1024,
        include_model_confidence=False,
        include_pairwise_metrics=True,
        model_count=3,
        use_thread_pool=False,
    )
    full = analytics._analytics_worker_count(8, source_side=2000, **common)
    scaled = analytics._analytics_worker_count(8, source_side=1024, **common)
    assert 1 <= full <= 2
    assert full < scaled


def test_point_validation_uses_same_windows_worker_caps_as_polygon(monkeypatch) -> None:
    """Vias/contacts (POINT) must not bypass Windows memory/CPU caps — parallelism only."""

    from karakal.core import analytics
    from karakal.core.domain import GeometryMode

    monkeypatch.setattr(analytics.os, "name", "nt")
    monkeypatch.setattr(analytics.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(analytics, "_available_memory_bytes", lambda: 1 * 1024**3)
    monkeypatch.setattr(analytics, "_analytics_worker_env_override", lambda: None)
    common = dict(
        analysis_max_side=1024,
        include_model_confidence=False,
        include_pairwise_metrics=True,
        model_count=3,
        use_thread_pool=False,
        source_side=2000,
    )
    polygon = analytics._analytics_worker_count(8, geometry_mode=GeometryMode.MASK, **common)
    points = analytics._analytics_worker_count(8, geometry_mode=GeometryMode.POINT, **common)
    assert polygon == points
    assert 1 <= points <= 2


def test_big_validation_runs_use_processes_on_windows(monkeypatch) -> None:
    from karakal.core import analytics

    monkeypatch.delenv(analytics.ANALYTICS_THREAD_POOL_ENV, raising=False)
    monkeypatch.setattr(analytics.os, "name", "nt")
    assert analytics._use_thread_pool_for_analytics(4) is True
    assert analytics._use_thread_pool_for_analytics(analytics.ANALYTICS_WINDOWS_PROCESS_MIN_RECORDS) is False
    monkeypatch.setenv(analytics.ANALYTICS_THREAD_POOL_ENV, "1")
    assert analytics._use_thread_pool_for_analytics(10_000) is True
    monkeypatch.setenv(analytics.ANALYTICS_THREAD_POOL_ENV, "0")
    assert analytics._use_thread_pool_for_analytics(4) is False


def test_cache_entry_is_written_whole(tmp_path) -> None:
    import pickle

    from karakal.core.cache_utils import atomic_pickle_dump

    target = tmp_path / "entry.pickle"
    atomic_pickle_dump(target, {"values": list(range(1000))})
    assert pickle.loads(target.read_bytes()) == {"values": list(range(1000))}
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(Exception):
        atomic_pickle_dump(target, {"bad": lambda: None})
    assert pickle.loads(target.read_bytes()) == {"values": list(range(1000))}
    assert not list(tmp_path.glob("*.tmp"))


def _reference_foreground_count_tables(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-speedup tables: OpenCV integral plus independent int64 cumsums."""

    binary = np.asarray(mask > 0, dtype=np.uint8)
    height, width = binary.shape[:2]
    integral = cv2.integral(binary)
    if isinstance(integral, tuple):
        integral = integral[0]
    integral = np.asarray(integral)
    column_prefix = np.zeros((height + 1, width), dtype=np.int64)
    row_prefix = np.zeros((height, width + 1), dtype=np.int64)
    if height and width:
        np.cumsum(binary, axis=0, out=column_prefix[1:], dtype=np.int64)
        np.cumsum(binary, axis=1, out=row_prefix[:, 1:], dtype=np.int64)
    return integral, column_prefix, row_prefix


def test_foreground_count_tables_match_cumsum_reference() -> None:
    from karakal.core.grid_anomaly import (
        _foreground_count_tables,
        _outline_side_coverages_from_tables,
        _rect_nonzero_count,
    )

    rng = np.random.default_rng(11)
    masks = [
        np.zeros((1, 1), dtype=np.uint8),
        np.zeros((17, 23), dtype=np.uint8),
        np.ones((32, 48), dtype=np.uint8) * 255,
        (rng.random((96, 128)) > 0.7).astype(np.uint8) * 255,
        (rng.random((200, 180)) > 0.55).astype(np.uint8) * 255,
    ]
    for frame in _masks():
        masks.append((frame[0] >= 128).astype(np.uint8) * 255)
    for mask in masks:
        got_integral, got_col, got_row = _foreground_count_tables(mask)
        ref_integral, ref_col, ref_row = _reference_foreground_count_tables(mask)
        assert np.array_equal(got_integral, ref_integral)
        assert np.array_equal(got_col.astype(np.int64, copy=False), ref_col)
        assert np.array_equal(got_row.astype(np.int64, copy=False), ref_row)
        height, width = mask.shape[:2]
        if height < 8 or width < 8:
            continue
        for _ in range(40):
            x = int(rng.integers(0, max(1, width - 8)))
            y = int(rng.integers(0, max(1, height - 8)))
            bw = int(rng.integers(2, min(16, width - x + 1)))
            bh = int(rng.integers(2, min(16, height - y + 1)))
            assert _rect_nonzero_count(got_integral, x, y, bw, bh) == _rect_nonzero_count(
                ref_integral, x, y, bw, bh
            )
            assert _outline_side_coverages_from_tables(got_col, got_row, x, y, bw, bh) == (
                _outline_side_coverages_from_tables(ref_col, ref_row, x, y, bw, bh)
            )


def test_identical_masks_skip_topology_without_changing_findings() -> None:
    from karakal.core.synthetic_chip import LayerStyle
    from karakal.core.synthetic_validation import make_case
    from karakal.core.topology_events import compare_to_reference, topology_findings

    for case in ("clean", "open", "bridge", "boundary_jitter", "extra_blob"):
        frame = make_case(case, 3, LayerStyle())
        masks = {model_id: mask > 0 for model_id, mask in frame.masks.items()}
        first = next(iter(masks.values()))
        assert compare_to_reference(first, first.copy(), model_id="a", reference_id="a") == ()
        found = topology_findings(masks)
        # Two-model path still mirrors; re-running must be stable.
        assert topology_findings(masks) == found


def test_topology_difference_bbox_dilate_matches_full_frame() -> None:
    """Touched labels from a bbox dilate must match a full-frame 3×3 dilate."""

    from karakal.core.synthetic_chip import LayerStyle
    from karakal.core.synthetic_validation import make_case
    from karakal.core.topology_events import _as_uint8, _difference, _objects

    for case in ("open", "bridge", "edge_shift", "extra_blob", "boundary_jitter"):
        frame = make_case(case, 7, LayerStyle())
        mask = frame.masks["a"] > 0
        reference = frame.masks["b"] > 0
        mine = _objects(mask)
        theirs = _objects(reference)
        got = _difference(mask, reference, mine, theirs)
        differ = mask ^ reference
        flat = np.flatnonzero(differ)
        if flat.size == 0:
            assert got is None
            continue
        near = np.flatnonzero(cv2.dilate(_as_uint8(differ), np.ones((3, 3), np.uint8)))
        mine_labels = mine.labels.ravel()[near]
        theirs_labels = theirs.labels.ravel()[near]
        assert got is not None
        assert np.array_equal(got.flat, flat)
        assert got.touched_mine == np.unique(mine_labels[mine_labels > 0]).tolist()
        assert got.touched_theirs == np.unique(theirs_labels[theirs_labels > 0]).tolist()
