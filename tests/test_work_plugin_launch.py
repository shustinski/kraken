"""Work items map onto the plugins the desktop already launches."""

from __future__ import annotations

from kraken_manager.application.work_launch import plugin_id_for_capability, plugin_id_for_work_kind
from kraken_manager.domain.work_orders import WorkKind


def test_work_kind_selects_the_existing_plugin() -> None:
    assert plugin_id_for_work_kind(WorkKind.VECTOR_REVIEW) == "contour"
    assert plugin_id_for_work_kind(WorkKind.VECTORIZE) == "contour"
    assert plugin_id_for_work_kind("train") == "neuralimage"
    assert plugin_id_for_work_kind(WorkKind.STITCH) is None
    assert plugin_id_for_work_kind("unknown") is None


def test_plugin_job_capability_selects_the_existing_plugin() -> None:
    assert plugin_id_for_capability("frames.vectorize.v1") == "contour"
    assert plugin_id_for_capability("frames.dataset.prepare.v1") == "contour"
    assert plugin_id_for_capability("dataset.model.train.v1") == "neuralimage"
    assert plugin_id_for_capability("frames.binary-segment.v2") == "neuralimage"
    assert plugin_id_for_capability("frames.confidence.v1") == "neuralimage"
    assert plugin_id_for_capability("layer.confidence.analyze.v1") == "karakal"
    assert plugin_id_for_capability("not-a-capability") is None
