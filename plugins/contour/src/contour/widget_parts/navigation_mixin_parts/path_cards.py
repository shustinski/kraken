"""navigation_mixin: status, history and locking of the path cards."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QTimer

from contour.ui.path_card import (
    ROLE_DATASET,
    ROLE_OUTPUT,
    STATUS_ERROR,
    STATUS_MUTED,
    STATUS_OK,
    PathCard,
)

_CARD_ATTRIBUTES = {
    ROLE_OUTPUT: "output_dir_edit",
    ROLE_DATASET: "dataset_dir_edit",
}


class PathCardsMixin:
    # ----------------------------------------------------------- card lookup
    def _path_card(self: Any, role: str) -> PathCard | None:
        attribute = _CARD_ATTRIBUTES.get(role)
        card = getattr(self, attribute, None) if attribute else None
        return card if isinstance(card, PathCard) else None

    # --------------------------------------------------------------- history
    def _recent_paths_for_role(self: Any, role: str) -> list[str]:
        controller = getattr(self, "_path_settings", None)
        if controller is None or not hasattr(controller, "recent_paths"):
            return []
        return controller.recent_paths(role)

    def _remember_recent_path(self: Any, role: str, path: str) -> None:
        controller = getattr(self, "_path_settings", None)
        if controller is None or not hasattr(controller, "remember_recent_path") or not str(path or "").strip():
            return
        controller.remember_recent_path(role, path)

    # ------------------------------------------------------------- busy flags
    def _set_path_card_busy(self: Any, role: str, busy: bool) -> None:
        busy_roles: set[str] = self.__dict__.setdefault("_path_card_busy_roles", set())
        if busy:
            busy_roles.add(role)
        else:
            busy_roles.discard(role)
        self._schedule_path_card_status_refresh()

    def _path_card_is_busy(self: Any, role: str) -> bool:
        return role in self.__dict__.get("_path_card_busy_roles", set())

    # ---------------------------------------------------------------- locking
    def lock_paths_for_kraken(self: Any, *, lock_destinations: bool) -> None:
        """Make Kraken-owned sources read-only; destinations stay editable on request."""

        badge = "Kraken"
        locked_tooltip = self._tr("path_locked_by_kraken")
        self.lock_sources_for_kraken()
        for role in (ROLE_OUTPUT, ROLE_DATASET):
            card = self._path_card(role)
            if card is None:
                continue
            if lock_destinations:
                card.set_locked(True, locked_tooltip)
                card.set_badge(badge, locked_tooltip)
            elif card.text():
                card.set_badge(badge, self._tr("path_suggested_by_kraken"))
        self._schedule_path_card_status_refresh()

    # ---------------------------------------------------------------- statuses
    def _schedule_path_card_status_refresh(self: Any) -> None:
        timer = self.__dict__.get("_path_card_status_timer")
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(0)
            timer.timeout.connect(self._refresh_path_card_statuses)
            self.__dict__["_path_card_status_timer"] = timer
        timer.start()

    def _refresh_path_card_statuses(self: Any) -> None:
        for role in (ROLE_OUTPUT, ROLE_DATASET):
            self._refresh_destination_path_card(role)
        self._refresh_sources_panel()

    def _refresh_destination_path_card(self: Any, role: str) -> None:
        card = self._path_card(role)
        if card is None:
            return
        path = card.text().strip()
        if not path:
            card.set_status("")
        elif Path(path).is_dir():
            card.set_status(self._tr("path_status_ready"), STATUS_OK)
        elif Path(path).exists():
            card.set_status(self._tr("path_status_not_a_folder"), STATUS_ERROR)
        else:
            card.set_status(self._tr("path_status_will_be_created"), STATUS_MUTED)
