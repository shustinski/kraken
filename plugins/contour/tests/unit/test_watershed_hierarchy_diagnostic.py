from __future__ import annotations

import numpy as np

from scripts.diagnose_watershed_hierarchy import (
    HierarchyConfig,
    build_hierarchy,
    label_boundaries,
    replay_partition,
)


def test_watershed_hierarchy_builds_complete_connected_merge_sequence() -> None:
    yy, xx = np.indices((96, 96), dtype=np.float32)
    image = np.clip(80.0 + 28.0 * np.sin(xx / 8.0) + 20.0 * np.cos(yy / 11.0), 0.0, 255.0).astype(np.uint8)

    result = build_hierarchy(image, HierarchyConfig(minima_window_px=9))

    assert result.initial_basin_count > 1
    assert len(result.initial_rag) >= result.initial_basin_count - 1
    assert len(result.merge_records) == result.initial_basin_count - 1
    assert all(
        current.score <= following.score
        for current, following in zip(result.merge_records, result.merge_records[1:], strict=False)
    )


def test_replay_partition_reaches_one_full_plane_region() -> None:
    yy, xx = np.indices((80, 80), dtype=np.float32)
    image = np.clip(100.0 + 35.0 * np.sin(xx / 7.0) * np.cos(yy / 9.0), 0.0, 255.0).astype(np.uint8)
    result = build_hierarchy(image, HierarchyConfig(minima_window_px=7))

    final = replay_partition(result.initial_labels, result.merge_records, len(result.merge_records))

    assert np.unique(final).tolist() == [1]
    assert not np.any(label_boundaries(final))
