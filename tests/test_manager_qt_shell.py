from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLabel, QTabBar, QWidget

from kraken_manager.presentation.qt import (
    FrameSelection,
    LayerListItem,
    LayerListModel,
    ProjectListItem,
    ProjectListModel,
    ProjectManagerShell,
    ProjectWorkspacePage,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_project_and_layer_models_publish_stable_roles(qapp):
    projects = ProjectListModel([ProjectListItem("p-1", "Atlas", 20, 30, "Local")])
    layers = LayerListModel([LayerListItem("l-1", "Metal 1", "metal", coverage=0.5)])

    project_index = projects.index(0, 0)
    layer_index = layers.index(0, 0)
    assert projects.data(project_index, Qt.ItemDataRole.DisplayRole) == "Atlas"
    assert projects.data(project_index, int(ProjectListModel.IdRole)) == "p-1"
    assert layers.data(layer_index, int(LayerListModel.TypeRole)) == "metal"
    assert b"projectId" in projects.roleNames().values()
    assert b"layerType" in layers.roleNames().values()


def test_project_manager_shell_has_stable_navigation_and_replaceable_pages(qapp):
    shell = ProjectManagerShell()

    assert shell.current_page_key() == "projects"
    assert len(shell.rail_actions) == 6
    assert shell.page("statistics") is not None

    replacement = QWidget()
    previous = shell.replace_page("statistics", replacement)
    shell.show_page("statistics")

    assert previous is not replacement
    assert shell.page_stack.currentWidget() is replacement
    assert shell.current_page_key() == "statistics"


def test_shell_rail_exposes_section_buttons_in_order(qapp):
    shell = ProjectManagerShell()
    assert list(shell.rail_actions) == [
        "projects", "my_work", "statistics", "performers", "plugins", "administration",
    ]
    for button in shell.rail_actions.values():
        assert not button.icon().isNull()
        assert button.toolTip()
    assert not hasattr(shell, "navigation_list")


def test_shell_rail_button_switches_section_and_highlights(qapp):
    shell = ProjectManagerShell()
    shell.rail_actions["statistics"].click()
    assert shell.current_page_key() == "statistics"
    assert shell.rail_actions["statistics"].isChecked()
    assert not shell.rail_actions["projects"].isChecked()
    shell.open_project_tab("p-1", "P1", QWidget())
    assert not any(button.isChecked() for button in shell.rail_actions.values())


def test_shell_set_page_visible_hides_rail_button_and_falls_back(qapp):
    shell = ProjectManagerShell()
    # isVisible() is only meaningful once the window is shown (Qt 6 semantics).
    shell.show()
    shell.show_page("administration")
    shell.set_page_visible("administration", False)
    assert not shell.rail_actions["administration"].isVisible()
    assert shell.current_page_key() == "projects"
    shell.set_page_visible("administration", True)
    assert shell.rail_actions["administration"].isVisible()


def test_shell_session_and_sync_labels_live_in_status_bar(qapp):
    shell = ProjectManagerShell()
    # isVisible() is only meaningful once the window is shown (Qt 6 semantics).
    shell.show()
    shell.set_session_summary("mark\nЛокальная сессия + сервер http://127.0.0.1:8080")
    shell.set_sync_status("synchronized")
    session_label = shell.findChild(QLabel, "sessionSummary")
    sync_label = shell.findChild(QLabel, "syncStatus")
    assert session_label is not None and session_label.isVisible()
    assert "mark" in session_label.text()
    assert sync_label is not None and sync_label.isVisible()
    assert sync_label.text() == "● Синхронизировано"
    shell.set_sync_status("")
    assert not sync_label.isVisible()


def test_selected_projects_navigation_can_return_from_workspace(qtbot):
    shell = ProjectManagerShell()
    qtbot.addWidget(shell)
    shell.open_project_tab("p-1", "Project", QWidget())
    shell.show()
    qtbot.mouseClick(shell.rail_actions["projects"], Qt.MouseButton.LeftButton)
    assert shell.current_page_key() == "projects"
    assert shell.current_project_id() is None


def test_project_catalog_delete_action_emits_selected_project(qapp):
    shell = ProjectManagerShell()
    page = shell.page("projects")
    assert page is not None
    item = ProjectListItem("p-delete", "Delete me", 2, 2, "Local")
    page.project_model.replace_items((item,))
    page.project_list.setCurrentIndex(page.project_model.index(0, 0))
    requested = []
    page.deleteRequested.connect(requested.append)

    page.delete_button.click()

    assert page.delete_button.isEnabled()
    assert requested == [item]


def test_shell_opens_lightweight_project_workspace(qapp):
    shell = ProjectManagerShell()
    workspace = shell.open_project_tab("p-1", "P1", ProjectWorkspacePage())

    image_requests = []
    vector_requests = []
    workspace.addImageRepresentationRequested.connect(lambda: image_requests.append(True))
    workspace.addVectorRepresentationRequested.connect(lambda: vector_requests.append(True))
    workspace.add_image_representation_button.click()
    workspace.add_vector_representation_button.click()

    assert isinstance(workspace, ProjectWorkspacePage)
    assert shell.current_page_key() == "project:p-1"
    assert workspace.matrix_view.matrix_size() == (1, 1)
    assert image_requests == [True]
    assert vector_requests == [True]


def test_workspace_uses_bottom_layer_tabs_and_status_bar_selection(qapp):
    shell = ProjectManagerShell()
    workspace = shell.open_project_tab("p-1", "P1", ProjectWorkspacePage())
    workspace.layer_model.replace_items(
        [
            LayerListItem("l-1", "Metal 1", "metal"),
            LayerListItem("l-2", "Metal 2", "metal"),
        ]
    )
    workspace.sync_layer_tabs()

    workspace.matrix_view.set_selection(FrameSelection.single(1, 1))
    workspace.matrix_view.set_matrix_size(8, 6)
    workspace._update_minimap_summary(None)
    qapp.processEvents()

    assert workspace.layer_tabs.count() == 2
    assert not workspace.add_layer_button.icon().isNull()
    assert workspace.findChild(QWidget, "matrixLodLabel") is None
    assert workspace.findChild(QWidget, "matrixZoomOutButton") is None
    assert workspace.findChild(QWidget, "matrixZoomInButton") is None
    assert workspace.findChild(QWidget, "matrixZoomFitButton") is None
    assert workspace.findChild(QWidget, "matrixZoomResetButton") is None
    assert workspace.findChild(QWidget, "matrixMinimapCheck") is None
    assert workspace.findChild(QWidget, "clearThumbnailCacheButton") is None
    assert workspace.findChild(QWidget, "sendReviewButton") is None
    assert workspace.matrix_minimap.parentWidget() is workspace.matrix_view.viewport()
    assert "8" in workspace.matrix_minimap.toolTip()
    assert shell.statusBar().currentMessage() == "Выбрано кадров: 1"
    assert shell.windowTitle() == "Kraken"


class _SelectionPage(QWidget):
    selectionCountChanged = pyqtSignal(int)


def test_shell_project_tabs_open_and_switch_project_pages(qapp):
    shell = ProjectManagerShell()
    page_a = QWidget()
    page_b = QWidget()
    shell.open_project_tab("p-a", "Alpha", page_a)
    shell.open_project_tab("p-b", "Beta", page_b)
    assert shell.projectTabs.count() == 2
    assert shell.current_project_id() == "p-b"
    assert shell.page_stack.currentWidget() is page_b
    shell.projectTabs.setCurrentIndex(0)
    assert shell.current_project_id() == "p-a"
    assert shell.page_stack.currentWidget() is page_a
    assert shell.page("project:p-b") is page_b


def test_shell_reopening_project_returns_existing_page_without_new_tab(qapp):
    shell = ProjectManagerShell()
    page_a = QWidget()
    first = shell.open_project_tab("p-a", "Alpha", page_a)
    again = shell.open_project_tab("p-a", "Alpha (renamed)", QWidget())
    assert first is page_a
    assert again is page_a
    assert shell.projectTabs.count() == 1
    assert shell.projectTabs.tabText(0) == "Alpha"
    assert shell.current_project_id() == "p-a"


def test_shell_close_project_tab_falls_back_to_neighbor_then_catalog(qapp):
    shell = ProjectManagerShell()
    for pid, name in (("p-a", "Alpha"), ("p-b", "Beta"), ("p-c", "Gamma")):
        shell.open_project_tab(pid, name, QWidget())
    shell.close_project_tab("p-c")
    assert shell.current_project_id() == "p-b"  # right neighbor
    shell.close_project_tab("p-b")
    assert shell.current_project_id() == "p-a"  # left neighbor (last one left)
    shell.close_project_tab("p-a")
    assert shell.current_project_id() is None
    assert shell.current_page_key() == "projects"  # no tabs left -> catalog
    assert shell.page("project:p-a") is None


def test_shell_closing_all_tabs_while_section_is_shown_keeps_section(qapp):
    shell = ProjectManagerShell()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    shell.show_page("statistics")
    shell.close_project_tab("p-a")  # closed tab was not the active page
    assert shell.current_project_id() is None
    assert shell.current_page_key() == "statistics"  # section stays, no crash on empty tab bar


def test_shell_rename_project_tab_updates_label(qapp):
    shell = ProjectManagerShell()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    shell.rename_project_tab("p-a", "Alfa")
    assert shell.projectTabs.tabText(0) == "Alfa"


def test_shell_project_tab_close_button_requests_close(qapp):
    shell = ProjectManagerShell()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    requested = []
    shell.project_tab_close_requested.connect(requested.append)
    close_button = shell.projectTabs.tabButton(
        shell.projectTabs.currentIndex(), QTabBar.ButtonPosition.RightSide
    )
    close_button.click()
    assert requested == ["p-a"]
    assert shell.current_project_id() == "p-a"  # page stays until the controller closes it


def test_shell_menu_enablement_follows_active_project_tab(qapp):
    shell = ProjectManagerShell()
    assert not shell.actions_menu.isEnabled()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    assert shell.actions_menu.isEnabled()
    assert shell.layers_action.isEnabled()
    shell.show_page("statistics")
    assert not shell.actions_menu.isEnabled()
    assert not shell.layers_action.isEnabled()
    shell.projectTabs.setCurrentIndex(0)
    assert shell.actions_menu.isEnabled()


def test_shell_selection_count_tracks_active_project_tab_only(qapp):
    shell = ProjectManagerShell()
    page_a = _SelectionPage()
    page_b = _SelectionPage()
    shell.open_project_tab("p-a", "Alpha", page_a)
    shell.open_project_tab("p-b", "Beta", page_b)
    page_b.selectionCountChanged.emit(3)
    assert "3" in shell.statusBar().currentMessage()
    shell.projectTabs.setCurrentIndex(0)
    assert "0" in shell.statusBar().currentMessage()  # A has no selection yet -> re-push 0
    page_b.selectionCountChanged.emit(5)
    assert "5" not in shell.statusBar().currentMessage()  # inactive project must not drive the bar
    page_a.selectionCountChanged.emit(1)
    assert "1" in shell.statusBar().currentMessage()
    assert shell.frame_properties_action.isEnabled()
    shell.projectTabs.setCurrentIndex(1)
    assert "3" in shell.statusBar().currentMessage()  # B's last count re-pushed


def test_shell_add_project_button_shows_catalog_and_emits_request(qapp):
    shell = ProjectManagerShell()
    shell.open_project_tab("p-a", "Alpha", QWidget())
    requests = []
    shell.addProjectRequested.connect(lambda: requests.append(True))
    shell.add_project_button.click()
    assert requests == [True]
    assert shell.current_page_key() == "projects"
    assert shell.current_project_id() is None


def test_shell_many_project_tabs_switch_stably(qapp):
    shell = ProjectManagerShell()
    pages = [
        shell.open_project_tab(f"p-{i}", "Проект с достаточно длинным названием " + "x" * 24, QWidget())
        for i in range(8)
    ]
    for index in range(8):
        shell.projectTabs.setCurrentIndex(index)
        assert shell.page_stack.currentWidget() is pages[index]
        assert shell.current_project_id() == f"p-{index}"
