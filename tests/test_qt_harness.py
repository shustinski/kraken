"""Self-test that the Qt GUI harness drives the REAL Hub with observable state.

Plan 5.1: the harness is proven by launching the real desktop composition
offscreen, performing one known action through the real UI, and asserting
observable state plus independent persistence — not merely "no exception".
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.project import GridOrientation

from tests.helpers.qt_harness import (
    catalog_project_names,
    click_menu,
    create_project,
    harness_capture,  # noqa: F401 — pytest fixture: the name must exist in the test module's namespace
    journal_events,
    launch_hub,
    open_project,
    wait_until,
)
from tests.test_hub_embedded_service_parts.kraken_tempdir import kraken_tempdir


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_launch_and_visible_state(qapp) -> None:
    with kraken_tempdir() as data_dir:
        with launch_hub(Path(data_dir), account=("operator", "")) as handle:
            window = handle.window
            assert window.isVisible()
            assert window.windowTitle() == "Kraken"
            catalog = handle.controller.catalog_page
            assert catalog is window.page("projects")
            assert catalog_project_names(handle) == []
            assert "Локальная сессия" in window._session_summary_label.text()
            # Bootstrap path: the first account exists in the local store.
            assert handle.service.accounts.account_count() == 1
            session = handle.service.login("operator", "")
            assert session is not None
            assert session.principal.subject == "operator"
            assert handle.messages == [] and handle.failures == []


def test_menu_navigate_and_create_project(qapp) -> None:
    with kraken_tempdir() as data_dir:
        with launch_hub(Path(data_dir), account=("operator", "")) as handle:
            project = create_project(handle, "Harness project")
            assert project.width == 2 and project.height == 1
            assert "Harness project" in catalog_project_names(handle)
            opened = open_project(handle, "Harness project")
            assert str(opened.id) == str(project.id)
            assert handle.controller._workspace is not None
            # A real menu action with the real Russian texts of the shell menu bar.
            click_menu(handle.window, "Управление → Слои…")
            wait_until(
                lambda: handle.controller._layer_dialog is not None
                and handle.controller._layer_dialog.isVisible(),
                what="layer manager dialog to open",
            )
            handle.controller._layer_dialog.close()
            # Journal: observable persistence, not "no exception".
            events = [event.event_type for event in journal_events(handle)]
            assert "ProjectCreated" in events
            assert "ProjectWorkspaceBoundV1" in events
            # Independent oracle: a fresh service over the same data dir.
            reopened = EmbeddedProjectService(str(handle.data_dir))
            stored = [item for item in reopened.list_projects() if item.name == "Harness project"]
            assert len(stored) == 1
            stored_events = [event.event_type for event in reopened.history(stored[0].id)]
            assert "ProjectCreated" in stored_events
            assert "ProjectWorkspaceBoundV1" in stored_events


def _find_nodes(node: dict, object_name: str):
    if node.get("objectName") == object_name:
        yield node
    for child in node.get("children", ()):
        yield from _find_nodes(child, object_name)


def test_failure_capture_produces_artifacts(qapp, harness_capture) -> None:  # noqa: F811
    with kraken_tempdir() as data_dir:
        with launch_hub(Path(data_dir), account=("operator", "")) as handle:
            logging.getLogger("kraken_hub.tests").warning("qt-harness log marker")
            # Controlled failure path: an invalid-state action through the real
            # service layer — a Windows reserved device name is rejected by the
            # workspace-name validation before anything is written.
            with pytest.raises(ValueError):
                handle.service.create_project(
                    principal=handle.session.principal,
                    name="aux",
                    width=2,
                    height=1,
                    orientation=GridOrientation.Y_DOWN,
                    idempotency_key="qt-harness-invalid-name",
                    layer_template=False,
                )
            png, tree_path, log_path = harness_capture["capture"]("controlled-failure")
    assert png.is_file() and png.stat().st_size > 1000
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    assert payload["window_title"] == "Kraken"
    assert payload["tree"]["class"] == "ProjectManagerShell"
    assert any(
        node["objectName"] == "projectCatalogList" for node in _find_nodes(payload["tree"], "projectCatalogList")
    ), "catalog list missing from the captured widget tree"
    assert "qt-harness log marker" in log_path.read_text(encoding="utf-8")
    assert harness_capture["paths"] == (png, tree_path, log_path)
    assert harness_capture["evidence_dir"] in png.parents


def test_wait_until_timeout_raises(qapp) -> None:
    started = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        wait_until(lambda: False, timeout=0.5, interval=0.05, what="controlled never-true predicate")
    elapsed = time.monotonic() - started
    assert "controlled never-true predicate" in str(excinfo.value)
    assert elapsed < 1.0, "wait_until must raise on timeout, not hang"
