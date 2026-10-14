"""test_hub_manager_app: test created layer selected."""
from __future__ import annotations
from types import SimpleNamespace
from kraken_hub.manager_app import (
    DesktopController,
)

def test_created_layer_is_selected_and_its_pipeline_is_loaded(monkeypatch) -> None:
    controller = object.__new__(DesktopController)
    layer = SimpleNamespace(id="layer-created")
    layer_item = SimpleNamespace(layer_id="layer-created")
    selected = []
    loaded = []

    class ServiceStub:
        @staticmethod
        def get_project(project_id):
            return SimpleNamespace(id=project_id)

    class LayerModelStub:
        @staticmethod
        def layer_by_id(layer_id):
            return layer_item if layer_id == "layer-created" else None

    class TabsStub:
        current = 0

        @staticmethod
        def count():
            return 2

        @staticmethod
        def tabData(index):
            return ("layer-old", "layer-created")[index]

        @classmethod
        def currentIndex(cls):
            return cls.current

        @classmethod
        def setCurrentIndex(cls, index):
            cls.current = index

    workspace = SimpleNamespace(
        layer_model=LayerModelStub(),
        layer_tabs=TabsStub(),
    )
    controller.service = ServiceStub()
    monkeypatch.setattr(
        controller,
        "_load_layers",
        lambda current_workspace, project: loaded.append(
            (current_workspace, project.id)
        ),
    )
    monkeypatch.setattr(
        controller,
        "_select_layer",
        lambda current_workspace, project_id, item: selected.append(
            (current_workspace, project_id, item)
        ),
    )

    controller._show_created_layer(workspace, "project-1", layer)

    assert loaded == [(workspace, "project-1")]
    assert TabsStub.current == 1
    assert selected == [(workspace, "project-1", layer_item)]
