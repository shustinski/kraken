from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from contour.local_batch_resume import (
    LocalBatchResume,
    LocalBatchResumeError,
    ManagedBatchResume,
    is_complete_cif,
    result_parameters,
)

CIF = "DS 1 1 1;\nL NM;\n( R {name}.png );\n( S 2 2 );\nDF;\nE\n"


def _settings(threshold: int) -> dict:
    return {
        "pipeline": {"steps": []},
        "recognition_settings": {"threshold": threshold},
        "save_options": {"save_cif": True},
    }


def _job(tmp_path, count=4):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    inputs = [tmp_path / f"f{index}.png" for index in range(1, count + 1)]
    state = LocalBatchResume.load(tmp_path / "contour-resume.json")
    return ManagedBatchResume(inputs, [f"frame-{index}" for index in range(1, count + 1)], outputs, state)


def _write(job, stem, text=None):
    job.output_for(stem).write_text(CIF.format(name=stem) if text is None else text, encoding="utf-8")


def test_resume_state_preserves_settings_and_tracks_mixed_frames(tmp_path):
    state_path = tmp_path / "contour-resume.json"
    state = LocalBatchResume.load(state_path)
    first, second = _settings(1), _settings(2)
    assert not state.needs_mixed_settings_confirmation(first, set())
    state.begin(first, set())
    assert not state.needs_mixed_settings_confirmation(first, {"1"})
    assert state.needs_mixed_settings_confirmation(second, {"1"})
    state.begin(second, {"1"})

    restored = LocalBatchResume.load(state_path)
    assert restored.current_settings == second
    groups = restored.groups({"1": "frame-1", "2": "frame-2"})
    assert [(group["settings"]["recognition_settings"]["threshold"], group["frame_ids"]) for group in groups] == [
        (1, ["frame-1"]),
        (2, ["frame-2"]),
    ]


def test_existing_files_without_state_have_unknown_provenance(tmp_path):
    state = LocalBatchResume.load(tmp_path / "contour-resume.json")
    settings = _settings(1)
    assert state.needs_mixed_settings_confirmation(settings, {"old"})
    assert state.has_unknown({"old"})
    state.begin(settings, {"old"})
    groups = state.groups({"old": "frame-old", "new": "frame-new"})
    assert {"settings": None, "frame_ids": ["frame-old"]} in groups
    assert any(group["settings"] == settings and group["frame_ids"] == ["frame-new"] for group in groups)


def test_agent_legacy_record_marks_outputs_unknown_and_requires_confirmation(tmp_path):
    path = tmp_path / "contour-resume.json"
    path.write_text(
        json.dumps(
            {"version": 1, "snapshots": [], "current": None, "frames": {}, "unknown_outputs": ["1"], "complete": False}
        ),
        encoding="utf-8",
    )
    state = LocalBatchResume.load(path)
    assert state.current_settings is None
    assert state.needs_mixed_settings_confirmation(_settings(1), {"1"})


def test_corrupt_state_is_an_error_not_a_fresh_job(tmp_path):
    path = tmp_path / "contour-resume.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LocalBatchResumeError):
        LocalBatchResume.load(path)
    assert path.read_text(encoding="utf-8") == "{not json"
    path.write_text(json.dumps({"version": 1, "snapshots": [], "current": "missing"}), encoding="utf-8")
    with pytest.raises(LocalBatchResumeError):
        LocalBatchResume.load(path)


def test_only_frames_without_valid_cif_are_continued_and_progress_counts_all(tmp_path):
    job = _job(tmp_path)
    _write(job, "f3")  # finished out of order
    _write(job, "f1", "DS 1 1 1;\nL NM;\n( S 2 2 );\n")  # interrupted
    (tmp_path / "outputs/.f2.cif.abc.part").write_text("partial", encoding="utf-8")

    assert job.completed() == {"f3"}
    remaining = job.begin(_settings(1))
    assert [name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for name in remaining] == ["f1.png", "f2.png", "f4.png"]
    assert job.overall_progress(0) == (1, 4)
    assert job.overall_progress(2) == (3, 4)
    assert json.loads((tmp_path / "contour-resume.json").read_text(encoding="utf-8"))["current"]


def test_reopened_job_restores_settings_and_warns_on_changed_settings(tmp_path):
    job = _job(tmp_path)
    job.begin(_settings(1))
    _write(job, "f1")
    _write(job, "f2")
    reopened = ManagedBatchResume(
        job.input_paths,
        list(job.frame_ids.values()),
        job.output_directory,
        LocalBatchResume.load(tmp_path / "contour-resume.json"),
    )
    assert reopened.state.current_settings == _settings(1)
    assert not reopened.running
    assert reopened.completed_cifs() == [job.output_for("f1"), job.output_for("f2")]
    assert not reopened.needs_confirmation(_settings(1))
    assert reopened.needs_confirmation(_settings(9))


def test_is_complete_cif_rejects_directories_and_truncated_files(tmp_path):
    (tmp_path / "dir.cif").mkdir()
    (tmp_path / "short.cif").write_text("DS 1 1 1;\n( S 1 1 );\nDF;\n", encoding="utf-8")
    (tmp_path / "ok.cif").write_text(CIF.format(name="ok"), encoding="utf-8")
    assert not is_complete_cif(tmp_path / "dir.cif")
    assert not is_complete_cif(tmp_path / "short.cif")
    assert is_complete_cif(tmp_path / "ok.cif")


def test_result_parameters_keep_single_shape_and_record_mixed_history():
    current = {
        "format": "contour-recognition-settings",
        "version": 2,
        "pipeline": {"steps": []},
        "recognition_settings": {"threshold": 2},
    }
    assert result_parameters(current, []) == current
    single = [{"settings": {**current, "save_options": {"save_cif": True}}, "frame_ids": ["a"]}]
    assert result_parameters(current, single) == current

    mixed = [
        {"settings": {**current, "recognition_settings": {"threshold": 1}}, "frame_ids": ["a"]},
        {"settings": None, "frame_ids": ["b"]},
    ]
    payload = result_parameters(current, mixed)
    assert payload["recognition_settings"] == {"threshold": 2}
    history = payload["settings_history"]
    assert history[0]["frame_ids"] == ["a"] and history[0]["provenance"] == "recorded"
    assert history[1] == {"settings": None, "provenance": "unknown", "frame_ids": ["b"]}


def test_managed_job_open_uses_staging_state(tmp_path):
    (tmp_path / "outputs").mkdir()
    session = SimpleNamespace(
        staging_root=tmp_path,
        input_paths=(tmp_path / "a.png",),
        manifest=SimpleNamespace(inputs=(SimpleNamespace(frame_id="frame-a"),)),
        output_directory=tmp_path / "outputs",
    )
    job = ManagedBatchResume.open(session)
    assert job.frame_ids == {"a": "frame-a"}
    assert job.state.path == tmp_path / "contour-resume.json"


class _Controller:
    def __init__(self):
        self.requests = []
        self.is_running = False

    def start(self, request):
        self.requests.append(request)
        return True


class _Settings:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


def _widget(job, *, threshold=1, confirm=True):
    from contour.widget_parts.processing_mixin_parts.restore_persisted_session_selection import (
        RestorePersistedSessionSelectionMixin,
    )
    from contour.widget_parts.processing_mixin_parts.start_batch_processing import StartBatchProcessingMixin

    class Widget(StartBatchProcessingMixin, RestorePersistedSessionSelectionMixin):
        pass

    widget = Widget()
    widget.__dict__.update(
        _batch_controller=_Controller(),
        _managed_batch_resume=job,
        _ui_language="en",
        logs=[],
        questions=[],
        progress=[],
        _batch_progress_enabled=True,
        _display_settings=None,
        batch_chunk_size=4,
        output_dir_edit=SimpleNamespace(text=lambda: str(job.output_directory)),
        _workspace=SimpleNamespace(image_paths=[str(path) for path in job.input_paths]),
    )
    widget._append_log = widget.logs.append
    widget._tr = lambda key, *args, **kwargs: key
    widget.get_pipeline = lambda: {"steps": []}
    widget._current_contour_settings = lambda: _Settings({"threshold": threshold})
    widget._current_save_options = lambda: _Settings({"save_cif": True})
    widget._show_batch_progress = lambda total: widget.progress.append((0, total))
    widget._apply_batch_progress_bar = lambda current, total, **_: widget.progress.append((current, total))
    widget._set_progress_status = lambda *args, **kwargs: None
    widget._finish_shared_progress_if_idle = lambda: None
    widget.batchProgress = SimpleNamespace(emit=lambda *args: None)
    widget.batchFinished = SimpleNamespace(emit=lambda: None)

    def ask(question):
        widget.questions.append(question)
        return confirm

    widget._confirm_processing_question = ask
    return widget


def test_continue_starts_only_missing_frames_with_overall_progress(tmp_path):
    job = _job(tmp_path)
    widget = _widget(job)
    widget.start_batch_processing()
    _write(job, "f2")
    _write(job, "f4")
    widget._on_batch_finished()
    assert not job.running
    assert any("Paused: 2 of 4" in line for line in widget.logs)

    reopened = ManagedBatchResume(
        job.input_paths,
        list(job.frame_ids.values()),
        job.output_directory,
        LocalBatchResume.load(tmp_path / "contour-resume.json"),
    )
    widget = _widget(reopened)
    widget.start_batch_processing()
    request = widget._batch_controller.requests[-1]
    assert [path[-6:] for path in request.image_paths] == ["f1.png", "f3.png"]
    assert widget.questions == []
    widget._on_batch_progress(1, 2)
    assert widget.progress[-1] == (3, 4)


def test_changed_settings_while_paused_require_confirmation(tmp_path):
    job = _job(tmp_path)
    job.begin(managed_settings(1))
    _write(job, "f1")
    job.finish()
    declined = _widget(job, threshold=5, confirm=False)
    declined.start_batch_processing()
    assert declined.questions and not declined._batch_controller.requests
    accepted = _widget(job, threshold=5, confirm=True)
    accepted.start_batch_processing()
    assert len(accepted._batch_controller.requests[-1].image_paths) == 3
    _write(job, "f2")
    groups = job.groups()
    assert [group["settings"]["recognition_settings"]["threshold"] for group in groups] == [1, 5]


def test_corrupt_state_blocks_batch(tmp_path):
    from contour.managed_batch_ui import managed_batch_paths

    widget = SimpleNamespace(_managed_batch_resume_error="damaged", _ui_language="en", logs=[])
    widget._append_log = widget.logs.append
    assert managed_batch_paths(widget) is None
    assert "damaged" in widget.logs[-1]


def managed_settings(threshold):
    return {
        "format": "contour-recognition-settings",
        "version": 2,
        "pipeline": {"steps": []},
        "recognition_settings": {"threshold": threshold},
        "save_options": {"save_cif": True},
    }


def test_mixed_legacy_and_new_frames_are_reported_truthfully(tmp_path):
    (tmp_path / "contour-resume.json").write_text(
        json.dumps(
            {"version": 1, "snapshots": [], "current": None, "frames": {}, "unknown_outputs": ["f1"], "complete": False}
        ),
        encoding="utf-8",
    )
    job = _job(tmp_path, count=3)
    _write(job, "f1")
    assert job.has_unknown()
    assert job.needs_confirmation(managed_settings(7))
    job.begin(managed_settings(7))
    _write(job, "f2")
    current = {key: value for key, value in managed_settings(7).items() if key != "save_options"}
    payload = result_parameters(current, job.groups())
    assert payload["recognition_settings"] == {"threshold": 7}
    history = {entry["provenance"]: entry["frame_ids"] for entry in payload["settings_history"]}
    assert history == {"recorded": ["frame-2"], "unknown": ["frame-1"]}
    assert "frame-3" not in str(payload["settings_history"])


def test_unchanged_settings_keep_single_settings_payload(tmp_path):
    job = _job(tmp_path, count=2)
    job.begin(managed_settings(3))
    _write(job, "f1")
    job.finish()
    job.begin(managed_settings(3))
    _write(job, "f2")
    current = {key: value for key, value in managed_settings(3).items() if key != "save_options"}
    assert result_parameters(current, job.groups()) == current


def test_manual_save_records_the_settings_that_produced_it(tmp_path):
    job = _job(tmp_path, count=2)
    job.begin(managed_settings(1))
    job.finish()
    _write(job, "f2")
    job.record_manual(job.output_for("f2"), managed_settings(8))
    reopened = ManagedBatchResume(
        job.input_paths,
        list(job.frame_ids.values()),
        job.output_directory,
        LocalBatchResume.load(tmp_path / "contour-resume.json"),
    )
    groups = reopened.groups()
    assert [(g["settings"]["recognition_settings"]["threshold"], g["frame_ids"]) for g in groups] == [(8, ["frame-2"])]
    assert reopened.state.current_settings == managed_settings(1)
    job.record_manual(tmp_path / "elsewhere.cif", managed_settings(8))  # outside the job: ignored
