"""Interactive returns must settle edits before publishing immutable outputs."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from kraken_core.plugin_protocol import (
    PluginFrameInput,
    PluginJobManifest,
    PluginResultManifest,
    WorkspacePluginContextV1,
)
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QWidget

from contour.application.services.workspace_session import WorkspaceSession
from contour.application.transition_save_guard import TransitionPromptChoice
from contour.application.view import ContourMainView
from contour.kraken_bridge import (
    ContourKrakenSession,
    ContourWorkspaceSession,
    KrakenBridgeError,
    prepare_contour_launch,
)
from contour.widget_parts.processing_mixin_parts.try_leave_current_frame import TryLeaveCurrentFrameMixin

pytestmark = pytest.mark.fast


class _Editor(QWidget, TryLeaveCurrentFrameMixin):
    def __init__(self, output: Path, choice: TransitionPromptChoice, *, save_ok: bool = True):
        super().__init__()
        self.output = output
        self.choice = choice
        self.save_ok = save_ok
        self.dirty = True
        self._batch_processor = SimpleNamespace(is_running=False)
        self._workspace = SimpleNamespace(current_image_has_changes=lambda: self.dirty, set_cif_index=lambda _: None)

    def _is_extraction_mode_enabled(self):
        return False

    def _sync_editor_polygons_to_current_workspace(self):
        pass

    def _update_vector_edit_status_label(self, **_kwargs):
        pass

    def _prompt_transition_vector_save_dialog(self):
        return self.choice

    def _persist_current_overlay_changes(self):
        if self.save_ok:
            self.output.write_bytes(b"edited vectors")
            self.dirty = False
        return self.save_ok

    def _discard_current_vector_changes(self):
        self.dirty = False

    def _warn_transition_blocked_after_failed_manual_save(self):
        pass

    def load_images(self, *_args, **_kwargs):
        pass

    def get_pipeline(self):
        return []

    def _current_contour_settings(self):
        return SimpleNamespace(to_dict=lambda: {})


class _ReturnWindow(ContourMainView):
    def __init__(self, editor: _Editor):
        # Keep the actual window close guard, without constructing unrelated UI.
        QMainWindow.__init__(self)
        self._widget = editor

    def _save_window_layout(self):
        pass


def _session(tmp_path: Path, operation: str = "frames.vectorize.v1") -> ContourKrakenSession:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    output = tmp_path / "outputs"
    output.mkdir()
    manifest = PluginJobManifest(
        job_id="return-job",
        operation=operation,
        project_id="project",
        layer_id="layer",
        actor_id="actor",
        target_representation_id="target",
        inputs=(
            PluginFrameInput(
                frame_id="frame",
                x=1,
                y=1,
                artifact_version_id="version",
                media_type="image/png",
                relative_path="a.png",
                sha256=hashlib.sha256(b"image").hexdigest(),
            ),
        ),
    )
    return ContourKrakenSession(manifest, tmp_path, tmp_path / "job.json", tmp_path / "result.json", (image,), output)


def _workspace_session(session: ContourKrakenSession) -> ContourWorkspaceSession:
    context = WorkspacePluginContextV1(
        project_id="project",
        project_name="Project",
        layer_id="layer",
        layer_name="Layer",
        operation=session.manifest.operation,
        plugin_id="contour",
        run_id="run",
        input_directories={"images": str(session.staging_root)},
        proposed_output_directory=str(session.output_directory),
        result_manifest_path=str(session.result_manifest_path),
    )
    return ContourWorkspaceSession(
        context,
        session.staging_root / "context.json",
        session.staging_root,
        session.output_directory,
        session.result_manifest_path,
    )


@pytest.fixture(autouse=True)
def _dismiss_result_messages(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args: pytest.fail(str(_args)))


@pytest.mark.parametrize("workspace", [False, True])
@pytest.mark.parametrize(
    "choice,save_ok",
    [
        (TransitionPromptChoice.CANCEL, True),
        (TransitionPromptChoice.SAVE, False),
    ],
)
def test_cancel_or_failed_save_does_not_publish(tmp_path, workspace, choice, save_ok):
    session = _session(tmp_path)
    cif = session.output_directory / "a.cif"
    cif.write_bytes(b"old vectors")
    editor = _Editor(cif, choice, save_ok=save_ok)
    window = _ReturnWindow(editor)
    bridge = _workspace_session(session) if workspace else session
    bridge.attach_return_action(window)
    window._kraken_return_action.trigger()
    assert not session.result_manifest_path.exists()
    assert editor.dirty
    assert cif.read_bytes() == b"old vectors"
    editor.dirty = False
    window.close()


@pytest.mark.parametrize(
    "choice,expected",
    [
        (TransitionPromptChoice.SAVE, b"edited vectors"),
        (TransitionPromptChoice.DISCARD, b"old vectors"),
    ],
)
def test_cif_return_hashes_saved_or_discarded_edits(tmp_path, choice, expected):
    session = _session(tmp_path)
    cif = session.output_directory / "a.cif"
    cif.write_bytes(b"old vectors")
    editor = _Editor(cif, choice)
    window = _ReturnWindow(editor)
    session.attach_return_action(window)
    window._kraken_return_action.trigger()
    result = PluginResultManifest.from_json(session.result_manifest_path.read_text(encoding="utf-8"))
    assert cif.read_bytes() == expected
    assert result.outputs[0].sha256 == hashlib.sha256(expected).hexdigest()
    window.close()


def _dataset_samples(session):
    (session.output_directory / "images").mkdir()
    (session.output_directory / "cif").mkdir()
    (session.output_directory / "images/a.png").write_bytes(b"image")
    label = session.output_directory / "cif/a.cif"
    label.write_bytes(b"old vectors")
    return label


def test_dataset_return_archives_saved_edits(tmp_path):
    session = _session(tmp_path, "frames.dataset.prepare.v1")
    label = _dataset_samples(session)
    window = _ReturnWindow(_Editor(label, TransitionPromptChoice.SAVE))
    session.attach_return_action(window)
    window._kraken_return_action.trigger()
    with ZipFile(session.output_directory / "dataset.zip") as archive:
        assert archive.read("cif/a.cif") == b"edited vectors"
    window.close()


def test_dataset_archive_refreshes_changed_added_and_removed_samples(tmp_path):
    session = _session(tmp_path, "frames.dataset.prepare.v1")
    label = _dataset_samples(session)
    session.build_result()
    label.write_bytes(b"updated vectors")
    (session.output_directory / "images/b.png").write_bytes(b"second image")
    (session.output_directory / "cif/b.cif").write_bytes(b"second vectors")
    session.build_result()
    with ZipFile(session.output_directory / "dataset.zip") as archive:
        assert archive.read("cif/a.cif") == b"updated vectors"
        assert archive.read("cif/b.cif") == b"second vectors"
    label.unlink()
    (session.output_directory / "images/a.png").unlink()
    result = session.build_result()
    archive_path = session.output_directory / "dataset.zip"
    with ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {"images/b.png", "cif/b.cif"}
    assert result.outputs[0].sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()


def test_incomplete_current_dataset_is_not_replaced_by_old_archive(tmp_path):
    session = _session(tmp_path, "frames.dataset.prepare.v1")
    label = _dataset_samples(session)
    session.build_result()
    label.unlink()
    with pytest.raises(KrakenBridgeError, match="Export samples"):
        session.build_result()


def test_archive_only_dataset_export_is_still_supported(tmp_path):
    session = _session(tmp_path, "frames.dataset.prepare.v1")
    archive_path = session.output_directory / "export.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("cif/a.cif", b"exported vectors")
    result = session.build_result()
    assert result.outputs[0].relative_path == "outputs/export.zip"
    assert result.outputs[0].sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()


def test_workspace_launch_uses_staged_vector_directory(tmp_path):
    session = _session(tmp_path)
    workspace = _workspace_session(session)
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    context = replace(
        workspace.context,
        input_directories={
            "images": str(tmp_path),
            "vectors": str(vectors),
        },
    )
    context.write(workspace.context_path)
    loaded, arguments = prepare_contour_launch(
        [
            "--kraken-workspace-context",
            str(workspace.context_path),
        ]
    )
    assert isinstance(loaded, ContourWorkspaceSession)
    assert arguments[arguments.index("--cif-dir") + 1] == str(vectors)
    with pytest.raises(KrakenBridgeError, match="unexpected arguments"):
        prepare_contour_launch(
            [
                "--kraken-workspace-context",
                str(workspace.context_path),
                "--cif-dir",
                str(tmp_path),
            ]
        )


def test_workspace_launch_rejects_missing_vector_directory(tmp_path):
    workspace = _workspace_session(_session(tmp_path))
    context = replace(
        workspace.context,
        input_directories={
            "images": str(tmp_path),
            "vectors": str(tmp_path / "missing"),
        },
    )
    context.write(workspace.context_path)
    with pytest.raises(KrakenBridgeError, match="vector directory"):
        ContourWorkspaceSession.load(workspace.context_path)


@pytest.mark.parametrize("with_vectors", [False, True])
def test_workspace_attach_restores_only_staged_vector_index(tmp_path, with_vectors):
    session = _session(tmp_path)
    editor = _Editor(session.output_directory / "a.cif", TransitionPromptChoice.DISCARD)
    editor._workspace = WorkspaceSession()
    editor._workspace.set_cif_index({"a": "old-user-project.cif"})
    editor._pending_restore_vector_paths = ["old-user-project.cif"]
    editor._pending_cif_directory_state = object()
    editor._pending_cif_directory_path_after_images = "old-project"
    window = _ReturnWindow(editor)
    window._pending_persisted_session_restore = True
    workspace = _workspace_session(session)
    vector_path = None
    if with_vectors:
        vectors = tmp_path / "vectors"
        vectors.mkdir()
        vector_path = vectors / "a.cif"
        vector_path.write_bytes(b"staged vectors")
        workspace = replace(workspace, vector_directory=vectors)
    workspace.attach_return_action(window)
    assert not window._pending_persisted_session_restore
    assert editor._pending_restore_vector_paths == []
    assert editor._pending_cif_directory_state is None
    assert editor._pending_cif_directory_path_after_images is None
    assert editor._workspace.resolve_cif_path(session.input_paths[0]) == (str(vector_path) if vector_path else None)
    editor.dirty = False
    window.close()


def test_cli_workspace_launch_reaches_interactive_return(tmp_path, monkeypatch):
    from kraken_core import process_lifetime

    from contour.application import bootstrap, cli

    session = _session(tmp_path)
    workspace = _workspace_session(session)
    workspace.context.write(workspace.context_path)
    cif = session.output_directory / "a.cif"
    cif.write_bytes(b"old vectors")
    window = _ReturnWindow(_Editor(cif, TransitionPromptChoice.SAVE))
    monkeypatch.setattr(window, "show", lambda: None)
    app = SimpleNamespace(
        exec=lambda: window._kraken_return_action.trigger(),
        sendPostedEvents=lambda *_args: None,
    )
    monkeypatch.setattr(bootstrap, "build_application", lambda _args: (app, window))
    monkeypatch.setattr(process_lifetime, "bind_child_process_lifetime", lambda: None)
    cli.main(["--kraken-workspace-context", str(workspace.context_path)])
    assert session.result_manifest_path.exists()
    assert cif.read_bytes() == b"edited vectors"
