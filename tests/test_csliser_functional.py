"""Shown CSliser copy -> managed Hub import -> archive/reopen/backup restore."""
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image, ImageDraw
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QMessageBox

from csliser.infrastructure.settings_store import WindowSettingsStore
from csliser.presentation.qt.window import CSliserWindow
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.imports import ImportMappingMode, ImportPlanner, ImportSource
from kraken_manager.domain.project import GridOrientation, LayerType, ProjectState, RepresentationKind
from tests.test_work_orders import _service


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_qt_file_transfer_and_hub_import_archive_restore(qtbot, tmp_path, monkeypatch):
    real_root = os.environ.get("KRAKEN_CSLISER_REAL_ROOT")
    source_root = Path(real_root) if real_root else tmp_path / "representative"
    count = 32 if real_root else 4
    if not real_root:
        for name in ("jpg_metal", "cif_metal"):
            (source_root / name).mkdir(parents=True)
        for i in range(count):
            image = Image.new("RGB", (64, 64), "black")
            ImageDraw.Draw(image).rectangle((8, 8, 48, 48), fill="white")
            image.save(source_root / "jpg_metal" / f"{i:04}.jpg")
            (source_root / "cif_metal" / f"{i:04}.cif").write_text(
                f"DS 1 1 1;\nL NM;\n( R {i:04}.jpg );\n( S 64 64 );\nP 8 8 48 8 48 48 8 48;\nDF;\nE\n")
    selected = [source_root / folder / f"{i:04}{suffix}" for folder, suffix in (("jpg_metal", ".jpg"), ("cif_metal", ".cif")) for i in range(count)]
    before = {str(p): digest(p) for p in selected}
    started = time.monotonic()
    window = CSliserWindow(settings_store=WindowSettingsStore(lambda: QSettings(str(tmp_path / "csliser.ini"), QSettings.Format.IniFormat)))
    qtbot.addWidget(window)
    for folder, suffix in (("jpg_metal", ".jpg"), ("cif_metal", ".cif")):
        window._source_folders[source_root / folder] = {"all_extensions": (suffix,), "active_extensions": {suffix}}
    window._source_ui()
    output = tmp_path / "copied"
    window.destination_folder_lineedit.setText(str(output))
    window.first_frame_lineedit.setText(f"0-{count-1}")
    window.copy_all_radiobutton.setChecked(True)
    window.add_extension_checkbox.setChecked(True)
    window._refresh_state()
    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: notices.append("complete"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: notices.append("warning"))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window.show()
    assert window.copy_button.isEnabled()
    window.copy_button.click()
    qtbot.waitUntil(lambda: window._worker_thread is None, timeout=30000)
    assert window._last_result.ok and window._last_result.completed == 2*count, notices
    copies = {str(item.source): item.destination for item in window._current_plan.operations}
    assert {source: digest(target) for source, target in copies.items()} == before
    window.copy_button.click()
    qtbot.waitUntil(lambda: window._worker_thread is None, timeout=30000)
    assert not window._last_result.ok and window._last_result.completed == 0
    assert {source: digest(target) for source, target in copies.items()} == before
    window.close()

    local = _service(tmp_path)
    actor = local.create_initial_account("owner", "Functional file transfer", "").principal
    project = local.create_project(principal=actor, name="CSliser functional", width=count, height=1,
        orientation=GridOrientation.Y_DOWN, idempotency_key="project")
    layer = local.create_layer(principal=actor, project=project, name="M1", layer_type=LayerType.METAL, order=0, idempotency_key="layer")
    image = local.create_representation(principal=actor, project=local.get_project(project.id), layer=layer,
        name="Images", kind=RepresentationKind.IMAGE, active=True, idempotency_key="image")
    vector = local.create_representation(principal=actor, project=local.get_project(project.id), layer=local.list_layers(project.id)[0],
        name="CIF", kind=RepresentationKind.VECTOR, source_image_representation_id=image.id, active=True, idempotency_key="vector")
    versions = []
    for suffix, representation in ((".jpg", image), (".cif", vector)):
        paths = sorted(p for p in copies.values() if p.suffix == suffix)
        sources = tuple(ImportSource(str(p), p.name, p.stat().st_size, digest(p)) for p in paths)
        plan = ImportPlanner().plan(width=count, height=1, sources=sources, mode=ImportMappingMode.EXPLICIT,
            explicit={str(p): (int(p.stem)+1, 1) for p in paths})
        imported = local.commit_managed_import(principal=actor, project=local.get_project(project.id), layer=local.list_layers(project.id)[0],
            representation=representation, plan=plan, idempotency_key="import"+suffix)
        assert len(imported.versions) == count
        assert all(v.media_type == ("application/x-cif" if suffix == ".cif" else "image/jpeg") for v in imported.versions)
        versions.extend(imported.versions)
    local.archive_project(principal=actor, project=local.get_project(project.id), idempotency_key="archive")
    reopened = EmbeddedProjectService(local.data_dir)
    assert reopened.get_project(project.id).state is ProjectState.ARCHIVED
    local.export_backup(project.id, tmp_path / "bundle")
    restored = _service(tmp_path / "restored")
    imported_project = restored.import_backup(tmp_path / "bundle")
    assert imported_project.state is ProjectState.ARCHIVED and restored.scan_integrity().valid
    for v in versions:
        assert restored.get_artifact_version(project.id, v.id) == v
        assert digest(restored.artifact_source_path(project.id, v.id)) == v.sha256
    assert {str(p): digest(p) for p in selected} == before
    (tmp_path / "receipt.json").write_text(json.dumps({"status": "passed", "real_corpus": bool(real_root), "pairs": count,
        "project_id": str(project.id), "versions": len(versions), "original_files_unchanged": True,
        "training_started": False, "coordinate_semantics": "catalog positions only, no physical adjacency inferred",
        "seconds": time.monotonic()-started}))
