from __future__ import annotations

import cProfile

from contour.infrastructure import profiling
from contour.infrastructure.contact_placement_profiler import ContactPlacementProfile


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
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(profiling, "CONTACT_MULTI_SELECTION_PROFILING_ENABLED", True)
    monkeypatch.setattr(profiling, "CONTACT_DELETION_PROFILING_ENABLED", False)
    monkeypatch.setattr(profiling, "CONTACT_COPY_PROFILING_ENABLED", True)
    monkeypatch.setattr(profiling, "CONTACT_PASTE_PROFILING_ENABLED", False)
    monkeypatch.setattr(profiling, "CONTACT_UNDO_PROFILING_ENABLED", True)
    monkeypatch.setattr(profiling, "CONTACT_REDO_PROFILING_ENABLED", False)

    assert profiling.contact_multi_selection_profiling_enabled()
    assert not profiling.contact_deletion_profiling_enabled()
    assert profiling.contact_copy_profiling_enabled()
    assert not profiling.contact_paste_profiling_enabled()
    assert profiling.contact_undo_profiling_enabled()
    assert not profiling.contact_redo_profiling_enabled()
