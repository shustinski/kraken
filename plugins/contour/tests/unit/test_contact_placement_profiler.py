from __future__ import annotations

import cProfile
from unittest.mock import patch

from contour.adapters.qt.preview import PreviewProcessingRunnable
from contour.application.processing import ContourExtractionSettings
from contour.application.use_cases.processing import PreviewProcessingRequest
from contour.infrastructure import profiling
from contour.infrastructure.contact_placement_profiler import (
    ContactDragProfile,
    ContactPlacementProfile,
    ImageRecognitionProfile,
    SceneZoomProfile,
)


def _some_contact_work() -> int:
    return sum(index * index for index in range(200))


def test_contact_profile_combines_main_and_worker_functions() -> None:
    session = ContactPlacementProfile.begin()
    assert _some_contact_work() > 0
    session.stop()

    worker = cProfile.Profile()
    worker.enable()
    assert _some_contact_work() > 0
    worker.disable()
    session.attach_worker(worker, 1.25)

    report = session.format_stats()

    assert "sort=cumulative" in report
    assert "_some_contact_work" in report
    assert session.timings_ms["preview_worker_wall"] == 1.25


def test_contact_profile_can_restart_main_stats_at_apply_boundary() -> None:
    session = ContactPlacementProfile.begin(action="multi_selection")
    assert _some_contact_work() > 0
    session.note("selection_gesture")
    session.restart_main_profiler()
    assert _some_contact_work() > 0
    session.stop()

    assert session.timings_ms["selection_gesture"] >= 0.0
    assert "_some_contact_work" in session.format_stats()


def test_contact_drag_profile_reports_frames_fps_and_commit() -> None:
    session = ContactDragProfile.begin(polygon_id=7, contact_count=11288)
    started_at = session.started_at
    assert _some_contact_work() > 0
    session.record_frame(started_at)
    assert _some_contact_work() > 0
    session.record_frame(started_at)
    session.finish(commit_ms=12.5)

    summary = session.format_summary(status="displayed")

    assert "[contour contact drag profiling]" in summary
    assert "polygon_id=7" in summary
    assert "contacts=11288" in summary
    assert "frames=2" in summary
    assert "fps=" in summary
    assert "commit=12.500ms" in summary
    assert "_some_contact_work" in session.format_stats()


def test_scene_zoom_profile_reports_frames_fps_and_zoom_range() -> None:
    session = SceneZoomProfile.begin(initial_zoom=1.0, target_zoom=1.5)
    started_at = session.started_at
    assert _some_contact_work() > 0
    session.record_frame(started_at)
    session.update_target(2.0)
    assert _some_contact_work() > 0
    session.record_frame(started_at)
    session.finish()

    summary = session.format_summary(status="displayed", final_zoom=2.0)

    assert "[contour scene zoom profiling]" in summary
    assert "zoom=1.0000->2.0000" in summary
    assert "target=2.0000" in summary
    assert "frames=2" in summary
    assert "fps=" in summary
    assert "_some_contact_work" in session.format_stats()


def test_image_recognition_profile_combines_worker_and_ui_stats() -> None:
    session = ImageRecognitionProfile.begin(
        image_path="sample.png",
        recognition_mode="via",
    )
    assert _some_contact_work() > 0
    session.stop()
    worker = cProfile.Profile()
    worker.enable()
    assert _some_contact_work() > 0
    worker.disable()
    session.attach_worker(worker, 12.5)
    session.polygon_count = 42
    session.note("result_applied_to_ui")

    summary = session.format_summary(status="displayed")

    assert "[contour image recognition profiling]" in summary
    assert "mode=via" in summary
    assert "polygons=42" in summary
    assert "worker_wall=12.500ms" in summary
    assert "_some_contact_work" in session.format_stats()


def test_worker_profiler_conflict_never_prevents_image_recognition() -> None:
    expected_result = object()
    runnable = PreviewProcessingRunnable(
        request_id=7,
        request=PreviewProcessingRequest(
            image_path="sample.png",
            pipeline_config={},
            contour_settings=ContourExtractionSettings(recognition_mode="via"),
        ),
        profile=True,
    )
    results: list[tuple[int, object]] = []
    errors: list[tuple[int, str]] = []
    runnable.signals.result.connect(
        lambda request_id, result: results.append((request_id, result))
    )
    runnable.signals.error.connect(
        lambda request_id, message: errors.append((request_id, message))
    )

    active_profiler = cProfile.Profile()
    active_profiler.enable()
    try:
        with patch(
            "contour.adapters.qt.preview.process_image_path",
            return_value=expected_result,
        ) as process_image:
            runnable.run()
    finally:
        active_profiler.disable()

    process_image.assert_called_once()
    assert results == [(7, expected_result)]
    assert errors == []


def test_contact_profile_code_variable_is_the_default(monkeypatch) -> None:
    for name in (
        "CONTOUR_PROFILE",
        "CONTOUR_PROFILING",
        "CONTOUR_PROFILE_ALL",
        "CONTOUR_PROFILE_CONTACT_PLACEMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(profiling, "CONTACT_PLACEMENT_PROFILING_ENABLED", False)
    assert not profiling.contact_placement_profiling_enabled()
    monkeypatch.setattr(profiling, "CONTACT_PLACEMENT_PROFILING_ENABLED", True)
    assert profiling.contact_placement_profiling_enabled()


def test_contact_profile_environment_can_override_code_variable(monkeypatch) -> None:
    monkeypatch.setattr(profiling, "CONTACT_PLACEMENT_PROFILING_ENABLED", False)
    monkeypatch.setenv("CONTOUR_PROFILE_CONTACT_PLACEMENT", "1")
    assert profiling.contact_placement_profiling_enabled()


def test_contact_action_profiles_have_separate_report_names() -> None:
    selection = ContactPlacementProfile.begin(action="multi_selection")
    selection.stop()
    deletion = ContactPlacementProfile.begin(action="deletion")
    deletion.stop()
    copy = ContactPlacementProfile.begin(action="copy")
    copy.stop()
    paste = ContactPlacementProfile.begin(action="paste")
    paste.stop()
    undo = ContactPlacementProfile.begin(action="undo")
    undo.stop()
    redo = ContactPlacementProfile.begin(action="redo")
    redo.stop()

    assert "[contour contact multi selection profiling]" in selection.format_summary(
        status="selected_2"
    )
    assert "[contour contact deletion profiling]" in deletion.format_summary(
        status="displayed"
    )
    assert "[contour contact copy profiling]" in copy.format_summary(status="copied_2")
    assert "[contour contact paste profiling]" in paste.format_summary(status="displayed")
    assert "[contour contact undo profiling]" in undo.format_summary(status="displayed")
    assert "[contour contact redo profiling]" in redo.format_summary(status="displayed")


def test_contact_action_code_switches_are_independent(monkeypatch) -> None:
    for name in (
        "CONTOUR_PROFILE",
        "CONTOUR_PROFILING",
        "CONTOUR_PROFILE_ALL",
        "CONTOUR_PROFILE_CONTACT_MULTI_SELECTION",
        "CONTOUR_PROFILE_CONTACT_DELETION",
        "CONTOUR_PROFILE_CONTACT_COPY",
        "CONTOUR_PROFILE_CONTACT_PASTE",
        "CONTOUR_PROFILE_CONTACT_UNDO",
        "CONTOUR_PROFILE_CONTACT_REDO",
        "CONTOUR_PROFILE_CONTACT_DRAG",
        "CONTOUR_PROFILE_SCENE_ZOOM",
        "CONTOUR_PROFILE_IMAGE_RECOGNITION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(profiling, "CONTACT_MULTI_SELECTION_PROFILING_ENABLED", True)
    monkeypatch.setattr(profiling, "CONTACT_DELETION_PROFILING_ENABLED", False)
    monkeypatch.setattr(profiling, "CONTACT_COPY_PROFILING_ENABLED", True)
    monkeypatch.setattr(profiling, "CONTACT_PASTE_PROFILING_ENABLED", False)
    monkeypatch.setattr(profiling, "CONTACT_UNDO_PROFILING_ENABLED", True)
    monkeypatch.setattr(profiling, "CONTACT_REDO_PROFILING_ENABLED", False)
    monkeypatch.setattr(profiling, "CONTACT_DRAG_PROFILING_ENABLED", True)
    monkeypatch.setattr(profiling, "SCENE_ZOOM_PROFILING_ENABLED", False)
    monkeypatch.setattr(profiling, "IMAGE_RECOGNITION_PROFILING_ENABLED", True)

    assert profiling.contact_multi_selection_profiling_enabled()
    assert not profiling.contact_deletion_profiling_enabled()
    assert profiling.contact_copy_profiling_enabled()
    assert not profiling.contact_paste_profiling_enabled()
    assert profiling.contact_undo_profiling_enabled()
    assert not profiling.contact_redo_profiling_enabled()
    assert profiling.contact_drag_profiling_enabled()
    assert not profiling.scene_zoom_profiling_enabled()
    assert profiling.image_recognition_profiling_enabled()
