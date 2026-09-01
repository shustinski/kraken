"""Compatibility checks for the decomposed core.repository facade."""

from __future__ import annotations

from karakal.core import analytics, exports, frame_details, image_io, mask_metrics, metric_keys, repository


def test_repository_facade_reexports_owned_implementations() -> None:
    assert repository.collect_frame_records is analytics.collect_frame_records
    assert repository.compute_build_result_analytics is analytics.compute_build_result_analytics
    assert repository.available_result_layer_exports is exports.available_result_layer_exports
    assert repository.load_frame_detail is frame_details.load_frame_detail
    assert repository.load_grayscale_image is image_io.load_grayscale_image
    assert repository.compute_comparison is mask_metrics.compute_comparison
    assert repository.metric_higher_is_better is metric_keys.metric_higher_is_better


def test_repository_executor_functions_remain_module_level() -> None:
    assert analytics._analyze_record_payload_for_executor.__module__ == "karakal.core.analytics"
    assert analytics._analyze_record_payload_batch_for_executor.__module__ == "karakal.core.analytics"
