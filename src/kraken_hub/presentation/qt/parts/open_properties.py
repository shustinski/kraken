"""manager_app: open properties."""
from __future__ import annotations
from kraken_manager.presentation.qt import (
    ObjectPropertiesDialog,
    ObjectPropertiesSnapshot,
)


class OpenPropertiesMixin:
    def _open_properties(self, snapshot: ObjectPropertiesSnapshot) -> None:
        ObjectPropertiesDialog(
            snapshot,
            self._layer_dialog or self.shell,
        ).exec()
