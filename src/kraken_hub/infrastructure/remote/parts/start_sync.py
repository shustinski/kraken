"""remote_client: start sync."""
from __future__ import annotations
from kraken_hub.workspace_service import (
    ProjectEventWake,
)

from kraken_hub.infrastructure.remote.parts.remote_sync_client import RemoteSyncClient


class StartSyncMixin:
    def start_sync(self) -> None:
        if self._sync is not None:
            return

        def on_event(wake: ProjectEventWake) -> None:
            for handler in list(self._wake_handlers):
                handler(wake)

        def on_catalog() -> None:
            for handler in list(self._catalog_handlers):
                handler()

        def on_status(status: str) -> None:
            for handler in list(self._status_handlers):
                handler(status)

        self._sync = RemoteSyncClient(
            self.base_url,
            token=self.auth.access_token,
            on_event=on_event,
            on_catalog=on_catalog,
            on_status=on_status,
        )
        self._sync.subscribe_catalog()
        self._sync.start()

    def stop_sync(self) -> None:
        if self._sync is not None:
            self._sync.stop()
            self._sync = None

    def subscribe_project(self, project_id: object) -> None:
        self.start_sync()
        if self._sync is not None:
            self._sync.subscribe_project(str(project_id))

    def unsubscribe_project(self, project_id: object) -> None:
        if self._sync is not None:
            self._sync.unsubscribe_project(str(project_id))
