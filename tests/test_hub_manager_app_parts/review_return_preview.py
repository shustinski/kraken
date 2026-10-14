"""test_hub_manager_app: test review return preview."""
from __future__ import annotations
from types import SimpleNamespace
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QTableWidget,
    QWidget,
)
from kraken_hub.manager_app import (
    DesktopController,
)

def test_review_return_preview_is_resizable(qapp, monkeypatch) -> None:
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    controller.session = SimpleNamespace(principal=object())
    report = SimpleNamespace(
        items=tuple(
            SimpleNamespace(
                relative_path=f"frames/frame-{index:03d}.cif",
                status=SimpleNamespace(value="unchanged"),
            )
            for index in range(50)
        ),
        can_commit=False,
    )

    class Service:
        def review_return_preflight(self, **_kwargs):
            return object(), SimpleNamespace(report=report)

    controller.service = Service()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: "review")

    def inspect_preview(dialog: QDialog) -> QDialog.DialogCode:
        assert dialog.minimumWidth() == 560
        assert dialog.minimumHeight() == 320
        assert dialog.width() >= 760
        assert dialog.height() >= 500
        assert dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
        assert dialog.maximumWidth() > dialog.minimumWidth()
        assert dialog.maximumHeight() > dialog.minimumHeight()
        table = dialog.findChild(QTableWidget)
        assert table is not None
        assert table.rowCount() == 50
        assert (
            table.horizontalHeader().sectionResizeMode(0)
            is QHeaderView.ResizeMode.Stretch
        )
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect_preview)

    controller.load_review_return()

def test_review_return_without_changed_files_refreshes_current_matrix(
    qapp, monkeypatch
) -> None:
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    controller.session = SimpleNamespace(principal=object())
    controller._project_id = "project-1"
    controller._workspace = SimpleNamespace(_selected_layer_id="layer-1")
    refreshed = []
    controller._load_representations = lambda *args: refreshed.append(args)
    report = SimpleNamespace(
        items=(
            SimpleNamespace(
                relative_path="frames/frame-001.cif",
                status=SimpleNamespace(value="unchanged"),
            ),
        ),
        can_commit=True,
    )

    class Service:
        def review_return_preflight(self, **_kwargs):
            return object(), SimpleNamespace(report=report)

        def commit_review_return(self, **_kwargs):
            return SimpleNamespace(candidate_versions=())

    controller.service = Service()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: "review")
    monkeypatch.setattr(QDialog, "exec", lambda _dialog: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)

    controller.load_review_return()

    assert refreshed == [(controller._workspace, "project-1", "layer-1")]
