"""manager_app: development session."""
from __future__ import annotations
import os
import sys
from kraken_hub import windows_credentials
from kraken_hub.composition import DesktopSession, EmbeddedProjectService

def _remember_credentials(username: str, password: str, *, data_directory=None) -> None:
    import logging

    try:
        windows_credentials.save_credentials(
            username, password, target=windows_credentials.credential_target(data_directory)
        )
    except Exception:
        logging.getLogger(__name__).exception("Не удалось сохранить учётные данные входа")

def _autofill_credentials(dialog, username, password, *, data_directory=None) -> None:
    try:
        target = windows_credentials.credential_target(data_directory)
        if not windows_credentials.credentials_available(target=target):
            return
        if not windows_credentials.verify_windows_identity(int(dialog.winId())):
            return
        remembered = windows_credentials.load_credentials(target=target)
        if remembered is None:
            return
    except Exception:
        return
    username.setText(remembered[0])
    password.setText(remembered[1])

def _development_session(service: EmbeddedProjectService) -> DesktopSession | None:
    """Return the explicit VS Code development session without showing login UI."""
    if getattr(sys, "frozen", False) or os.environ.get("KRAKEN_DEV_AUTO_LOGIN") != "1":
        return None
    username = os.environ.get("KRAKEN_DEV_USERNAME", "vscode")
    password = os.environ.get("KRAKEN_DEV_PASSWORD", "")
    if not service.has_accounts:
        return service.create_initial_account(
            username=username,
            display_name=os.environ.get("KRAKEN_DEV_DISPLAY_NAME", "VS Code Developer"),
            password=password,
        )
    session = service.login(username, password)
    if session is None:
        raise RuntimeError(
            "Dev auto-login account does not match the configured data directory; "
            "use a clean KRAKEN_DATA_DIR or update KRAKEN_DEV credentials"
        )
    return session
