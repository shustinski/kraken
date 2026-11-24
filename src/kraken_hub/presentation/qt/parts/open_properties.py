"""manager_app: open properties."""

from __future__ import annotations
from kraken_manager.presentation.qt import (
    ObjectPropertiesDialog,
    ObjectPropertiesSnapshot,
)
from kraken_core.frame_matrix import profiling
from PyQt6.QtCore import QTimer


class OpenPropertiesMixin:
    def _open_properties(self, snapshot: ObjectPropertiesSnapshot) -> None:
        ObjectPropertiesDialog(
            snapshot,
            self._layer_dialog or self.shell,
        ).exec()

    def _open_node_properties(self, snapshot: ObjectPropertiesSnapshot, profile: profiling.ProfileSession) -> None:
        dialog = ObjectPropertiesDialog(snapshot, self._layer_dialog or self.shell)
        profile.checkpoint("dialog_created")

        def finish_profile() -> None:
            profile.checkpoint("first_event_loop")
            profile.finish(ready=dialog.isVisible())

        QTimer.singleShot(0, finish_profile)
        try:
            dialog.exec()
        finally:
            profile.finish(ready=False)
