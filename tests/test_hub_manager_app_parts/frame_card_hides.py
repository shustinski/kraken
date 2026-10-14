"""test_hub_manager_app: test frame card hides."""
from __future__ import annotations
from types import SimpleNamespace
from kraken_hub.manager_app import (
    DesktopController,
)
from kraken_manager.domain.identity import Permission

def test_frame_card_hides_mutating_actions_without_permissions() -> None:
    class Combo:
        @staticmethod
        def currentData():
            return None

    class Matrix:
        @staticmethod
        def selected_coordinates(*, maximum):
            assert maximum == 2
            return ((1, 1),)

        @staticmethod
        def cell_data(_x, _y):
            return SimpleNamespace(
                payload={"frame_id": "frame-1"},
                status="ready",
                performer_initials="",
            )

    class Service:
        permissions = frozenset()

        @staticmethod
        def get_project(_project_id, *, as_of=None):
            del as_of
            return SimpleNamespace(name="Project", width=1)

        @classmethod
        def project_permissions(cls, _project_id, _principal):
            return cls.permissions

        @staticmethod
        def list_layers(_project_id):
            return (SimpleNamespace(id="layer-1", name="Layer"),)

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return ()

        @staticmethod
        def history(_project_id):
            return ()

        @staticmethod
        def list_principals(*, include_inactive):
            assert include_inactive
            return ()

        @staticmethod
        def list_artifact_series(
            _project_id,
            *,
            layer_id,
            include_archived,
        ):
            assert layer_id == "layer-1"
            assert include_archived
            return ()

        @staticmethod
        def list_notes(_project_id, *, layer_id, frame_id):
            assert (layer_id, frame_id) == ("layer-1", "frame-1")
            return ()

    workspace = SimpleNamespace(
        _selected_layer_id="layer-1",
        matrix_view=Matrix(),
        image_representation_combo=Combo(),
        vector_representation_combo=Combo(),
    )
    controller = object.__new__(DesktopController)
    controller.service = Service()
    controller.session = SimpleNamespace(principal=SimpleNamespace(id="viewer"))
    controller._workspace = workspace
    controller._project_id = "project-1"
    snapshots = []
    controller._open_properties = snapshots.append

    controller.show_selected_frame()
    viewer_snapshot = snapshots[-1]
    assert viewer_snapshot.actions == ()
    assert viewer_snapshot.file_actions == ()
    assert [label for label, _callback in viewer_snapshot.version_actions] == [
        "Экспортировать / открыть",
        "Проверить внешний файл",
    ]

    Service.permissions = frozenset(
        {Permission.ADD_NOTE, Permission.IMPORT_ARTIFACT}
    )
    controller.show_selected_frame()
    editor_snapshot = snapshots[-1]
    assert len(editor_snapshot.actions) == 2
    assert len(editor_snapshot.file_actions) == 4
    assert [label for label, _callback in editor_snapshot.version_actions] == [
        "Активировать",
        "Экспортировать / открыть",
        "Проверить внешний файл",
    ]
