"""Domain login helpers: server status, program account upsert, error text."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from kraken_core.settings import QSettings

from kraken_hub.application.remote_error import RemoteServerError
from kraken_hub.infrastructure.credentials.secret_store import KeyringSecretStore
from kraken_hub.infrastructure.remote.client import RemoteHttpClient
from kraken_hub.presentation.qt.preferences_dialog import (
    ProgramAccount,
    load_program_accounts,
    program_secret_store,
    program_settings,
    save_program_accounts,
)

STATUS_ENABLED = "enabled"
STATUS_DISABLED = "disabled"
STATUS_UNREACHABLE = "unreachable"


def domain_server_status(
    url: str,
    *,
    timeout: float = 3.0,
    http_factory: Callable[[str], Any] | None = None,
) -> str:
    """Check whether the server at ``url`` accepts domain logins.

    Returns one of :data:`STATUS_ENABLED`, :data:`STATUS_DISABLED` or
    :data:`STATUS_UNREACHABLE`.
    """
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        return STATUS_UNREACHABLE
    try:
        client = (http_factory or RemoteHttpClient)(url)
        payload = client.request("GET", "/api/v1/health", token="", timeout=timeout)
    except Exception:  # noqa: BLE001 - any probe failure means the server is unreachable
        return STATUS_UNREACHABLE
    if isinstance(payload, Mapping) and payload.get("domain_auth") is True:
        return STATUS_ENABLED
    return STATUS_DISABLED


def upsert_program_account(
    username: str,
    password: str,
    *,
    display_name: str = "",
    settings: QSettings | None = None,
    secrets: KeyringSecretStore | None = None,
) -> ProgramAccount:
    """Update the default local account for ``username`` (case-insensitive),
    creating it when absent. GitLab accounts are left untouched."""
    source = settings or program_settings()
    store = secrets or program_secret_store()
    accounts = load_program_accounts(source, store)
    existing = next(
        (
            account
            for account in accounts
            if account.provider == "local" and account.username.casefold() == username.casefold()
        ),
        None,
    )
    if existing is None:
        updated = ProgramAccount(str(uuid4()), "local", username, display_name or username, password, True)
        accounts.append(updated)
    else:
        updated = ProgramAccount(
            existing.account_id,
            "local",
            existing.username,
            display_name or existing.display_name,
            password,
            True,
        )
        accounts = [updated if account.account_id == existing.account_id else account for account in accounts]
    accounts = [
        ProgramAccount(
            account.account_id,
            account.provider,
            account.username,
            account.display_name,
            account.secret,
            account.account_id == updated.account_id,
        )
        for account in accounts
    ]
    save_program_accounts(accounts, source, store)
    return updated


def domain_login_error_text(exc: BaseException) -> str:
    """User-facing text for a failed domain login attempt."""
    if isinstance(exc, RemoteServerError) and exc.status is None:
        # RemoteHttpClient raises "Cannot reach Kraken Server at <url>: ..." for network failures;
        # other status-less errors (e.g. a disabled server account) keep their own message.
        if str(exc).startswith("Cannot reach Kraken Server at"):
            return "Сервер недоступен"
    return str(exc)
