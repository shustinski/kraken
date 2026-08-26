from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from contour.adapters.qt.preview import PreparedImageRunnable
from contour.application.preview_cancellation import PreviewProcessingCancelled
from contour.application.use_cases.processing import PreparedImageRequest
from contour.infrastructure import profiling
from contour.infrastructure.filter_application_profiler import FilterApplicationProfile
from contour.pipeline import PreprocessingPipeline


def test_filter_application_profile_reports_enabled_operations_and_stats() -> None:
    profile = FilterApplicationProfile.begin(
        image_path="frames/sample.png",
        pipeline_config={
            "steps": [
                {"operation": "gaussian_blur", "enabled": True},
                {"operation": "threshold", "enabled": False},
            ]
        },
    )
    profile.record_phase("source_copy", 2.5)
    profile.record_step(0, "gaussian_blur", 7.25)
    profile.finish()

    summary = profile.format_summary(status="completed")

    assert "[contour filter application profiling]" in summary
    assert "status=completed" in summary
    assert "filters=1" in summary
    assert "operations='gaussian_blur'" in summary
    assert "image='sample.png'" in summary
    assert "source_copy=2.500ms" in summary
    assert "step=1 operation='gaussian_blur' wall=7.250ms" in profile.format_stats()


def test_filter_application_profile_env_switch(monkeypatch) -> None:
    for name in (
        "CONTOUR_PROFILE",
        "CONTOUR_PROFILING",
        "CONTOUR_PROFILE_ALL",
        "CONTOUR_PROFILE_FILTER_APPLICATION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(profiling, "FILTER_APPLICATION_PROFILING_ENABLED", False)
    assert not profiling.filter_application_profiling_enabled()

    monkeypatch.setenv("CONTOUR_PROFILE_FILTER_APPLICATION", "1")
    assert profiling.filter_application_profiling_enabled()


def test_pipeline_reports_each_enabled_filter_timing() -> None:
    pipeline = PreprocessingPipeline([PreprocessingPipeline.create_step("threshold")])
    timings: list[tuple[int, str, float]] = []

    result = pipeline.apply(
        np.zeros((8, 8), dtype=np.uint8),
        timing_callback=lambda *args: timings.append(args),
    )

    assert result.shape == (8, 8)
    assert len(timings) == 1
    assert timings[0][0] == 0
    assert timings[0][1] == "threshold"
    assert timings[0][2] >= 0.0


def test_prepared_image_runnable_emits_profile_without_affecting_result() -> None:
    source = np.zeros((8, 8), dtype=np.uint8)
    expected = np.full((8, 8), 7, dtype=np.uint8)
    runnable = PreparedImageRunnable(
        request_id=9,
        request=PreparedImageRequest(
            image_path="sample.png",
            source_image=source,
            pipeline_config={"steps": []},
        ),
        profile=True,
    )
    results: list[tuple[int, str, object, object]] = []
    profiles: list[tuple[int, FilterApplicationProfile, str]] = []
    runnable.signals.result.connect(lambda *args: results.append(args))
    runnable.signals.profile.connect(lambda *args: profiles.append(args))

    with patch(
        "contour.adapters.qt.preview.prepare_image_for_preview",
        return_value=expected,
    ) as prepare_image:
        runnable.run()

    prepare_image.assert_called_once()
    assert len(results) == 1
    assert np.array_equal(results[0][2], expected)
    assert len(profiles) == 1
    assert profiles[0][0] == 9
    assert profiles[0][2] == "completed"
    assert profiles[0][1].wall_ms >= 0.0


def test_cancelled_filter_application_emits_cancelled_profile() -> None:
    runnable = PreparedImageRunnable(
        request_id=10,
        request=PreparedImageRequest(
            image_path="sample.png",
            source_image=np.zeros((8, 8), dtype=np.uint8),
            pipeline_config={"steps": []},
        ),
        profile=True,
    )
    profiles: list[tuple[int, FilterApplicationProfile, str]] = []
    results: list[object] = []
    runnable.signals.profile.connect(lambda *args: profiles.append(args))
    runnable.signals.result.connect(lambda *args: results.append(args))

    with patch(
        "contour.adapters.qt.preview.prepare_image_for_preview",
        side_effect=PreviewProcessingCancelled,
    ):
        runnable.run()

    assert results == []
    assert len(profiles) == 1
    assert profiles[0][2] == "cancelled"


def test_prepared_image_copy_runs_in_worker_not_constructor() -> None:
    source = MagicMock()
    worker_source = np.zeros((8, 8), dtype=np.uint8)
    source.copy.return_value = worker_source
    runnable = PreparedImageRunnable(
        request_id=11,
        request=PreparedImageRequest(
            image_path="sample.png",
            source_image=source,
            pipeline_config={"steps": []},
        ),
    )
    source.copy.assert_not_called()

    with patch(
        "contour.adapters.qt.preview.prepare_image_for_preview",
        return_value=worker_source,
    ) as prepare_image:
        runnable.run()

    source.copy.assert_called_once_with()
    prepare_image.assert_called_once_with(
        source_image=worker_source,
        pipeline_config={"steps": []},
    )
