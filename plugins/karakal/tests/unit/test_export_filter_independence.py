"""Main-window vs details-dialog error-type filters stay independent."""
from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtWidgets import QApplication, QCheckBox

from karakal.app.presenter import KarakalPresenter
from karakal.ui.ui_constants import GRID_INSPECTION_ERROR_TYPE_OPTIONS


def test_details_class_conflict_callback_is_disconnected(qapp=None) -> None:
    app = QApplication.instance() or QApplication([])
    host = SimpleNamespace(
        grid_error_type_checks={
            error_type: QCheckBox()
            for _label, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS
        },
        _class_conflict_user_wants=True,
        _details_dialogs=[],
    )
    for checkbox in host.grid_error_type_checks.values():
        checkbox.setChecked(True)
    # Simulate the presenter wiring used when opening details.
    dialog = SimpleNamespace(_on_class_conflict_user_wants_changed=lambda wants: None)
    dialog._on_class_conflict_user_wants_changed = None
    assert dialog._on_class_conflict_user_wants_changed is None
    main_types = KarakalPresenter._selected_grid_error_types(host)
    assert "small_artifact" in main_types
    # Flipping only the details-local checkbox must not change main selection.
    details_checks = {
        error_type: QCheckBox() for _label, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS
    }
    for key, checkbox in details_checks.items():
        checkbox.setChecked(key != "small_artifact")
    still_main = KarakalPresenter._selected_grid_error_types(host)
    assert "small_artifact" in still_main
    assert not details_checks["small_artifact"].isChecked()
