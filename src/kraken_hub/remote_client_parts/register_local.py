"""remote_client: register local."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from pathlib import Path
from kraken_manager.application.ports import StorageProfile
from kraken_manager.domain.identity import (
    Performer,
    Principal,
)
from kraken_manager.infrastructure.filesystem._codec import decode_model
from kraken_hub.workspace_service import (
    REMOTE_STORAGE_PROFILE,
    ProjectEventWake,
)

from kraken_hub.remote_client_parts.extract_review_archive import (
    RemoteAuth,
    RemoteServerError,
)

from kraken_hub.remote_client_parts.principal_server_payload import _principal_from_server_payload


class RegisterLocalMixin:
    def __init__(
        self,
        base_url: str,
        *,
        auth: RemoteAuth,
        data_dir: Path | str | None = None,
        http: RemoteHttpClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.data_dir = Path(data_dir or Path.home() / ".kraken").resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile = REMOTE_STORAGE_PROFILE
        self._http = http or RemoteHttpClient(self.base_url)
        self._remote_ids: set[str] = set()
        self._sync: RemoteSyncClient | None = None
        self._wake_handlers: list[Callable[[ProjectEventWake], None]] = []
        self._catalog_handlers: list[Callable[[], None]] = []
        self._status_handlers: list[Callable[[str], None]] = []

    @classmethod
    def authenticate_local(
        cls,
        base_url: str,
        *,
        username: str,
        password: str,
        data_dir: Path | str | None = None,
        http: RemoteHttpClient | None = None,
    ) -> RemoteServerProjectService:
        client = http or RemoteHttpClient(base_url)
        payload = client.request(
            "POST",
            "/api/v1/auth/sessions",
            token="",
            payload={"username": username, "password": password},
        )
        return cls._from_local_session(
            base_url,
            payload,
            username=username,
            data_dir=data_dir,
            http=client,
        )

    @classmethod
    def register_local(
        cls,
        base_url: str,
        *,
        username: str,
        password: str,
        display_name: str = "",
        data_dir: Path | str | None = None,
        http: RemoteHttpClient | None = None,
    ) -> RemoteServerProjectService:
        client = http or RemoteHttpClient(base_url)
        payload = client.request(
            "POST",
            "/api/v1/auth/accounts",
            token="",
            payload={
                "username": username,
                "password": password,
                "display_name": display_name or username,
            },
        )
        return cls._from_local_session(
            base_url,
            payload,
            username=username,
            data_dir=data_dir,
            http=client,
        )

    @classmethod
    def _from_local_session(
        cls,
        base_url: str,
        payload: object,
        *,
        username: str,
        data_dir: Path | str | None,
        http: RemoteHttpClient,
    ) -> RemoteServerProjectService:
        if not isinstance(payload, Mapping) or not str(payload.get("access_token", "")):
            raise RemoteServerError("Kraken Server returned an invalid local session")
        raw = payload.get("principal", {})
        principal_data = raw if isinstance(raw, Mapping) else {}
        principal = Principal.local(
            principal_id=str(principal_data.get("id", "")),
            subject=str(principal_data.get("username") or username),
            display_name=str(principal_data.get("display_name") or username),
        )
        return cls(
            base_url,
            auth=RemoteAuth(str(payload["access_token"]), principal),
            data_dir=data_dir,
            http=http,
        )

    @property
    def supports_workspace_roots(self) -> bool:
        return False

    @property
    def supports_live_sync(self) -> bool:
        return True

    def list_storage_profiles(self) -> tuple[StorageProfile, ...]:
        return (self.profile,)

    def current_principal(self) -> Principal:
        payload = self._call("GET", "/api/v1/session")
        if not isinstance(payload, Mapping):
            raise RemoteServerError("Kraken Server returned an invalid session")
        return _principal_from_server_payload(payload)

    def list_principals(
        self,
        *,
        include_inactive: bool = False,
    ) -> tuple[Principal, ...]:
        suffix = "?include_inactive=true" if include_inactive else ""
        payload = self._call("GET", f"/api/v1/principals{suffix}")
        values = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            _principal_from_server_payload(item)
            for item in values
            if isinstance(item, Mapping)
        )

    def list_project_principals(
        self,
        project_id: object,
        *,
        include_inactive: bool = False,
    ) -> tuple[Principal, ...]:
        selected: list[Principal] = []
        for principal in self.list_principals(include_inactive=include_inactive):
            try:
                roles = self.project_roles(project_id, principal.id)
            except RemoteServerError as exc:
                if exc.status == 404:
                    continue
                raise
            if roles:
                selected.append(principal)
        return tuple(selected)

    def list_performers(
        self, *, include_archived: bool = False
    ) -> tuple[Performer, ...]:
        suffix = "?include_archived=true" if include_archived else ""
        payload = self._call("GET", f"/api/v1/performers{suffix}")
        values = payload.get("items", ()) if isinstance(payload, Mapping) else ()
        return tuple(
            decode_model(Performer, item)
            for item in values
            if isinstance(item, Mapping)
        )

    def add_wake_handler(self, handler: Callable[[ProjectEventWake], None]) -> None:
        self._wake_handlers.append(handler)

    def add_catalog_handler(self, handler: Callable[[], None]) -> None:
        self._catalog_handlers.append(handler)

    def add_status_handler(self, handler: Callable[[str], None]) -> None:
        self._status_handlers.append(handler)
