from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QListWidget

from contour.ui.status_list_delegate import StatusBackgroundListDelegate, attach_status_row_delegate


def test_attach_status_row_delegate_installs() -> None:
    _app = QApplication.instance() or QApplication([])
    lst = QListWidget()
    attach_status_row_delegate(lst)
    assert isinstance(lst.itemDelegate(), StatusBackgroundListDelegate)
