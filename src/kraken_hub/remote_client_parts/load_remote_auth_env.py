"""remote_client: load remote auth env."""
from __future__ import annotations
from typing import TYPE_CHECKING
import os
from pathlib import Path
from kraken_manager.domain.identity import (
    Principal,
)

from kraken_hub.remote_client_parts.extract_review_archive import RemoteAuth

def load_remote_auth_from_env() -> RemoteAuth | None:
    token = (
        os.environ.get("KRAKEN_GITLAB_TOKEN")
        or os.environ.get("KRAKEN_SERVER_TOKEN")
        or ""
    )
    token = token.strip()
    if not token:
        return None
    subject = (
        os.environ.get("KRAKEN_GITLAB_SUBJECT", "remote-user").strip() or "remote-user"
    )
    display = (
        os.environ.get("KRAKEN_GITLAB_DISPLAY_NAME", "Remote User").strip()
        or "Remote User"
    )
    issuer = os.environ.get("KRAKEN_GITLAB_ISSUER", "https://gitlab.local").strip()
    principal = Principal.gitlab(
        issuer=issuer,
        subject=subject,
        display_name=display,
        principal_id=os.environ.get("KRAKEN_GITLAB_PRINCIPAL_ID") or None,
    )
    return RemoteAuth(access_token=token, principal=principal)

def build_remote_service(
    *,
    base_url: str | None = None,
    auth: RemoteAuth | None = None,
    data_dir: Path | str | None = None,
) -> RemoteServerProjectService | None:
    url = (base_url or os.environ.get("KRAKEN_SERVER_URL") or "").strip()
    if not url:
        return None
    credentials = auth or load_remote_auth_from_env()
    if credentials is None:
        return None
    from kraken_hub.remote_client import RemoteServerProjectService

    return RemoteServerProjectService(url, auth=credentials, data_dir=data_dir)

if TYPE_CHECKING:
    from kraken_hub.remote_client import RemoteServerProjectService
