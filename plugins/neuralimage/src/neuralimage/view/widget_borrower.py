from __future__ import annotations

from PyQt6.QtWidgets import QBoxLayout, QLayout, QWidget


class WidgetBorrower:
    """Temporarily reparents widgets and restores them on release."""

    def __init__(self) -> None:
        self._records: list[tuple[QWidget, QWidget | None, QLayout | None, int | None]] = []

    def borrow(self, widget: QWidget, new_parent: QWidget, new_layout: QBoxLayout) -> None:
        old_parent = widget.parentWidget()
        old_layout = old_parent.layout() if old_parent is not None else None
        index: int | None = None
        if old_layout is not None:
            index = old_layout.indexOf(widget)
            if index >= 0:
                old_layout.removeWidget(widget)
        widget.setParent(new_parent)
        new_layout.addWidget(widget)
        self._records.append((widget, old_parent, old_layout, index if index is not None and index >= 0 else None))

    def restore_all(self) -> None:
        while self._records:
            widget, old_parent, old_layout, index = self._records.pop()
            current_parent = widget.parentWidget()
            if current_parent is not None:
                parent_layout = current_parent.layout()
                if parent_layout is not None:
                    parent_layout.removeWidget(widget)
            if old_layout is not None and index is not None:
                widget.setParent(old_parent)
                if isinstance(old_layout, QBoxLayout):
                    old_layout.insertWidget(index, widget)
                else:
                    old_layout.addWidget(widget)
            elif old_parent is not None:
                widget.setParent(old_parent)
