"""Read a previous deployment fixture using the currently installed Qt/runtime.

Only explicit, completed Desktop self-test profiles are accepted. This private
probe opens the old account/project and checks real managed bytes and Qt cells;
it never creates substitute versions or a fresh project in the source profile.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import traceback
from uuid import UUID, uuid4

from kraken_core.safe_files import contained_path, ensure_regular_directory, open_regular_read


def previous_fixture(path):
    path = Path(path).resolve(strict=True)
    if path.name != "report.json" or not re.fullmatch(r"desktop-self-test-[0-9a-f]{32}", path.parent.name):
        raise ValueError("Only a completed Desktop self-test report is accepted")
    root = ensure_regular_directory(path.parent)
    with open_regular_read(path, root=root) as stream:
        encoded = stream.read(4 * 1024 * 1024 + 1)
    if len(encoded) > 4 * 1024 * 1024:
        raise ValueError("Previous self-test report exceeds size limit")
    value = json.loads(encoded)
    if (value.get("status") != "passed" or Path(value.get("directory", "")).resolve() != root
            or not value.get("settings_isolated") or not value.get("source_files_unchanged")):
        raise ValueError("Previous self-test did not establish an isolated successful fixture")
    receipt = value["managed_matrix"]
    if not isinstance(receipt.get("expected"), list) or not 1 <= len(receipt["expected"]) <= 16:
        raise ValueError("Previous managed matrix is not a bounded fixture")
    for key in ("project_id", "layer_id", "image_representation_id", "vector_representation_id"):
        UUID(receipt[key])
    profile = contained_path(root, ("profile",))
    ensure_regular_directory(profile)
    return root, value


def _sha(path):
    digest = hashlib.sha256()
    with open_regular_read(path, root=path.parent) as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_files(root):
    catalog = root / "profile" / "catalog" / "projects"
    result = {}
    for path in catalog.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(catalog).parts
        if len(parts) > 1 and parts[1] in {"events", "objects", "snapshots", "project.json", "workspace.json"}:
            result[path.relative_to(root).as_posix()] = _sha(path)
    if not result:
        raise ValueError("Previous profile has no canonical project data")
    return result


def run_reopen_probe(output_directory, previous_report):
    source, previous = previous_fixture(previous_report)
    output = Path(output_directory).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Reopen output must be separate from the previous fixture")
    run = output / f"desktop-upgrade-reopen-{uuid4().hex}"
    run.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", directory=str(run), previous_report_sha256=_sha(Path(previous_report)),
                  source_profile=str(source / "profile"), ordinary_native_input=False, training=False)
    controller = app = shell = None
    before = _canonical_files(source)
    settings_root = source / "settings"
    settings_before = {str(p): _sha(p) for p in settings_root.rglob("*") if p.is_file()}
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtGui import QFont, QFontDatabase
        from PyQt6.QtWidgets import QApplication
        from kraken_core.settings import QSettings, configure_settings_directory
        from kraken_hub.composition import EmbeddedProjectService
        from kraken_hub.deployment_matrix_probe import verify_project, verify_workspace
        from kraken_hub.manager_app import DesktopController
        from kraken_hub.secret_store import KeyringSecretStore
        from kraken_manager.presentation.qt import ProjectListItem, ProjectManagerShell

        app = QApplication.instance() or QApplication([])
        configure_settings_directory(run / "settings")
        assert Path(QSettings("Kraken", "KrakenHub").fileName()).resolve().is_relative_to(run / "settings")
        if not QFontDatabase.families() and os.name == "nt":
            font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
            if font.is_file() and QFontDatabase.addApplicationFont(str(font)) >= 0:
                app.setFont(QFont("Segoe UI", 9))
        # Reading a local project does not require a newly minted credential.
        # A distinct OS namespace keeps any accidental lookup out of user data.
        service = EmbeddedProjectService(source / "profile", secret_store=KeyringSecretStore("Kraken.UpgradeRead." + uuid4().hex))
        session = service.login("self-test", "")
        if session is None:
            raise RuntimeError("Previous fixture account cannot authenticate")
        receipt = previous["managed_matrix"]
        project = service.get_project(receipt["project_id"])
        if project is None:
            raise RuntimeError("Previous project disappeared after update")
        verify_project(service, receipt)
        for expected in receipt["expected"]:
            for kind in ("image", "vector"):
                path = service.artifact_source_path(project.id, expected[kind + "_version_id"])
                if _sha(path) != expected[kind + "_sha256"]:
                    raise RuntimeError("Previous managed artifact bytes changed")
        for version_id in receipt.get("previous_vector_version_ids", ()):
            service.artifact_source_path(project.id, version_id)  # Includes historical SHA verification.
        history = service.history(project.id)
        if not history:
            raise RuntimeError("Previous canonical history is empty")
        shell = ProjectManagerShell()
        shell.resize(1280, 900)
        controller = DesktopController(shell, service, session, thumbnail_store_uri="memory://", plugin_items=[])
        controller._agent_timer.stop()
        controller.open_project(ProjectListItem(str(project.id), project.name, project.width, project.height, "local"))
        shell.show()
        verify_workspace(app, controller, receipt, run / "previous-project-reopened.png")
        result.update(status="passed", project_id=str(project.id), pairs=len(receipt["expected"]),
            managed_files=2 * len(receipt["expected"]), historical_heads=len(receipt.get("previous_vector_version_ids", ())),
            history_events=len(history), old_account_authenticated=True, qt_matrix_and_thumbnails=True)
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        if controller is not None:
            controller._agent_timer.stop()
            controller.agent_runtime.shutdown()
        if shell is not None:
            shell.close()
        if app is not None:
            app.processEvents()
        result["canonical_data_unchanged"] = _canonical_files(source) == before
        result["previous_settings_unchanged"] = {str(p): _sha(p) for p in settings_root.rglob("*") if p.is_file()} == settings_before
        if not result["canonical_data_unchanged"] or not result["previous_settings_unchanged"]:
            result.update(status="failed", preservation_error="Previous canonical data/settings changed during reopen")
        (run / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "passed" else 1
