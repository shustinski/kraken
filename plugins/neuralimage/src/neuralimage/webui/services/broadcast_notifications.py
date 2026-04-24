from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock


@dataclass(frozen=True)
class BroadcastNotification:
    id: int
    message: str
    created_by: str
    created_at: datetime


class BroadcastNotificationStore:
    def __init__(self, *, limit: int = 200) -> None:
        self._limit = max(1, int(limit))
        self._lock = Lock()
        self._next_id = 1
        self._items: list[BroadcastNotification] = []

    def add(self, message: str, created_by: str = '') -> BroadcastNotification:
        with self._lock:
            notification = BroadcastNotification(
                id=self._next_id,
                message=message,
                created_by=created_by,
                created_at=datetime.now(timezone.utc),
            )
            self._next_id += 1
            self._items.append(notification)
            if len(self._items) > self._limit:
                self._items = self._items[-self._limit :]
            return notification

    def after(self, after_id: int, *, limit: int = 20) -> tuple[list[BroadcastNotification], int]:
        normalized_after_id = max(0, int(after_id))
        normalized_limit = max(1, int(limit))
        with self._lock:
            items = [item for item in self._items if item.id > normalized_after_id][:normalized_limit]
            latest_id = self._items[-1].id if self._items else normalized_after_id
            return list(items), latest_id

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._next_id = 1


_broadcast_store = BroadcastNotificationStore()


def get_broadcast_notification_store() -> BroadcastNotificationStore:
    return _broadcast_store
