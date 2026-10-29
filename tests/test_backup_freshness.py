"""Repeat the administration backup action against a changed real project."""
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QPushButton

from kraken_hub.presentation.qt.parts.administration_panel import _administration_panel
from kraken_manager.infrastructure.migration import BundleVerifier, load_bundle_manifest
from tests.test_work_orders import _service, _project_with_frames


def test_two_backup_clicks_preserve_history_and_restore_latest(tmp_path, qtbot, monkeypatch):
    service = _service(tmp_path)
    owner, _performer, project, *_rest = _project_with_frames(service, tmp_path / "inputs", with_vectors=False)
    destination = tmp_path / "backups"
    destination.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: str(destination))
    notices, errors = [], []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args))
    panel = _administration_panel(service, owner)
    qtbot.addWidget(panel)
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
    button = next(b for b in panel.findChildren(QPushButton) if b.text() == "Создать резервную копию")
    button.click()
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
    first, = destination.iterdir()
    original = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    service.rename_project(principal=owner.principal, project=project, name="Latest", idempotency_key="rename")
    button.click()
    qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
    assert not errors and len(notices) == 2
    latest, = (p for p in destination.iterdir() if p != first)
    before, after = load_bundle_manifest(first), load_bundle_manifest(latest)
    assert before.bundle_id != after.bundle_id and after.event_count == before.event_count + 1
    assert original == {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    for bundle in (first, latest):
        BundleVerifier().verify(bundle).raise_for_errors()
    restored_service = _service(tmp_path / "restored")
    new_owner = restored_service.create_initial_account("restore", "Restore", "")
    restored = restored_service.import_backup(latest, principal=new_owner.principal, take_ownership=True)
    assert restored.name == "Latest"
    series = restored_service.list_artifact_series(restored.id)
    assert len(series) == 2
    payloads = {
        restored_service.artifact_source_path(restored.id, restored_service.artifact_versions(restored.id, item.id)[0].id).read_bytes()
        for item in series
    }
    assert payloads == {b"image-a", b"image-b"}
    panel.close()
