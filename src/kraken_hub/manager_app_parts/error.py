"""manager_app: error."""
from __future__ import annotations


class ErrorMixin:
    def _error(self, text: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.warning(self.shell, "Kraken", text)
