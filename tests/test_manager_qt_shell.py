from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from kraken_manager.presentation.qt import (
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
    assert shell.navigation_list.count() == 6
    assert shell.page("statistics") is not None

    replacement = QWidget()
    previous = shell.replace_page("statistics", replacement)
    shell.show_page("statistics")

    assert previous is not replacement
    assert shell.page_stack.currentWidget() is replacement
    assert shell.current_page_key() == "statistics"


def test_shell_opens_lightweight_project_workspace(qapp):
    shell = ProjectManagerShell()
    workspace = shell.open_project_workspace()

    image_requests = []
    vector_requests = []
    workspace.addImageRepresentationRequested.connect(lambda: image_requests.append(True))
    workspace.addVectorRepresentationRequested.connect(lambda: vector_requests.append(True))
    workspace.add_image_representation_button.click()
    workspace.add_vector_representation_button.click()

    assert isinstance(workspace, ProjectWorkspacePage)
    assert shell.current_page_key() == "workspace"
    assert workspace.matrix_view.matrix_size() == (1, 1)
    assert image_requests == [True]
    assert vector_requests == [True]

