"""Quality-gate test: palette scenario S01 — local login and account creation.

Covers the registry actions ``HUB-BOOTSTRAP-ACCOUNT`` and ``HUB-LOGIN``.
The scenario is offline GUI only — no beta stand, no network:

1. An empty local account store offers the account-creation window; a
   valid password pair creates the account and the window reports success
   (``HUB-BOOTSTRAP-ACCOUNT``: *"Creates a local account"*).
2. With an account on disk, the account-creation window refuses a second
   account and reports the registry's exact error text
   (``HUB-BOOTSTRAP-ACCOUNT``: *"A local account already exists"*).
3. The login window authenticates the stored account with the correct
   password (``HUB-LOGIN``: *"Authenticates the local account"*).
4. The login window rejects a wrong password with the registry's exact
   error text (``HUB-LOGIN``: *"Неверные учётные данные или аккаунт
   временно заблокирован."*) and leaves the user unauthenticated.

The GUI is driven with pytest-qt (``qtbot``), which the gui tier exports
through ``pytest.ini``'s ``addopts = --matplotlib-offscreen --qt-api=pyqt6``.
Every test points ``KRAKEN_SETTINGS_DIR`` at its own temporary directory
*before* importing any Kraken product module, so the windows' settings
read/write go to a throwaway location and the developer's real settings
never see a test account — the same isolation idiom the existing gui-tier
window tests use (see tests/test_main_window.py). ``KRAKEN_WORKSPACE_ROOT``
keeps anything workspace-scoped out of the repository as well.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module-level isolation: pin the settings and workspace roots to a
# throwaway directory before the product modules below resolve their
# defaults, and before the windows import ``kraken.settings.app_paths``.
# The directories are created here and persist for the whole module
# session; each test moves them aside to its own ``tmp_path`` (see the
# ``isolated_settings`` fixture).
# ---------------------------------------------------------------------------
os.environ.setdefault("KRAKEN_WORKSPACE_ROOT", ".test-workspace-s01-quality")
os.environ.setdefault("KRAKEN_SETTINGS_DIR", ".test-settings-s01-quality")
Path(os.environ["KRAKEN_SETTINGS_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["KRAKEN_WORKSPACE_ROOT"]).mkdir(parents=True, exist_ok=True)

# Isolated-settings imports: must happen AFTER the environment pin above.
from kraken.settings.account import LocalAccountManager  # noqa: E402
from kraken.settings.password import PasswordHasher  # noqa: E402
from kraken.ui.account_create import AccountCreateWindow  # noqa: E402
from kraken.ui.login import LoginWindow  # noqa: E402

pytestmark = pytest.mark.gui

#: The exact error texts the registry promises (docs/quality/registry/
#: actions.json). These strings are the product promise; the assertions
#: below use them verbatim.
REGISTRY_ACCOUNT_EXISTS_ERROR = "A local account already exists."
REGISTRY_INVALID_CREDENTIALS_ERROR = (
    "Неверные учётные данные или аккаунт временно заблокирован."
)


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch) -> Iterator[Path]:
    """Point the settings store at ``tmp_path`` for the duration of a test.

    ``kraken.settings.app_paths.settings_dir()`` re-reads the environment
    variable on every call, so the monkeypatched value is picked up by the
    windows and the account manager alike. The previous value is restored
    automatically when the test ends. Yields the temporary settings
    directory for direct inspection.
    """
    monkeypatch.setenv("KRAKEN_SETTINGS_DIR", str(tmp_path))
    monkeypatch.setenv("KRAKEN_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    yield tmp_path


def _create_account(settings_dir: Path, password: str) -> None:
    """Write a local account with *password* directly into *settings_dir*."""
    manager = LocalAccountManager(PasswordHasher(), base_dir=str(settings_dir))
    manager.create_account(password)


def _create_window(qtbot, parent=None) -> AccountCreateWindow:
    window = AccountCreateWindow(parent=parent)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def _login_window(qtbot, parent=None) -> LoginWindow:
    window = LoginWindow(parent=parent)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


# ---------------------------------------------------------------------------
# HUB-BOOTSTRAP-ACCOUNT — create the first local account
# ---------------------------------------------------------------------------


def test_s01_account_creation_creates_first_local_account(qtbot, isolated_settings) -> None:
    """An empty store accepts a valid password pair and creates the account.

    Registry HUB-BOOTSTRAP-ACCOUNT: entering a valid password pair into the
    account-creation window creates a local account (``has_account()``
    becomes true) and the window reports success through
    ``account_created``.
    """
    window = _create_window(qtbot)
    assert not window.has_account(), (
        "the isolated settings directory must start without an account"
    )

    window.password_edit.setText("correct horse battery staple")
    window.confirm_edit.setText("correct horse battery staple")

    created = []
    window.account_created.connect(lambda: created.append(True))
    qtbot.mouseClick(window.submit_button, 1)

    assert created == [True], (
        "the window must emit account_created after a valid password pair"
    )
    assert window.has_account(), (
        "the product must report a local account right after creation"
    )
    manager = LocalAccountManager(
        PasswordHasher(), base_dir=str(isolated_settings)
    )
    assert manager.has_account(), (
        "the account must be persisted in the isolated settings store"
    )
    assert window.status_label.text() == ""
    assert not window.isVisible(), (
        "the window hides itself once the account is created"
    )


# ---------------------------------------------------------------------------
# HUB-BOOTSTRAP-ACCOUNT — a second account is refused with the exact
# registry text
# ---------------------------------------------------------------------------


def test_s01_account_creation_refuses_when_account_exists(
    qtbot, isolated_settings
) -> None:
    """With an account on disk, the creation window shows the registry error.

    Registry HUB-BOOTSTRAP-ACCOUNT error condition: *"A local account
    already exists."* The window must show that text verbatim when a
    submit is attempted while an account already exists.
    """
    _create_account(isolated_settings, "first correct password")
    window = _create_window(qtbot)
    assert window.has_account(), (
        "the product must detect the existing local account"
    )

    created = []
    window.account_created.connect(lambda: created.append(True))
    window.password_edit.setText("another password")
    window.confirm_edit.setText("another password")
    qtbot.mouseClick(window.submit_button, 1)

    assert window.status_label.text() == REGISTRY_ACCOUNT_EXISTS_ERROR, (
        "the window must show the registry's exact error text "
        f"({REGISTRY_ACCOUNT_EXISTS_ERROR!r})"
    )
    assert created == [], (
        "the window must not report success when an account already exists"
    )


# ---------------------------------------------------------------------------
# HUB-LOGIN — authenticate the stored local account
# ---------------------------------------------------------------------------


def test_s01_local_login(qtbot, isolated_settings) -> None:
    """The login window authenticates the stored account with its password.

    Registry HUB-LOGIN: the local account's correct password authenticates
    the user and the window emits ``login_succeeded`` with the username.
    """
    _create_account(isolated_settings, "right horse battery staple")

    window = _login_window(qtbot)
    successes: list[str] = []
    window.login_succeeded.connect(lambda username: successes.append(username))

    window.user_edit.setText("shustinsky")
    window.password_edit.setText("right horse battery staple")
    qtbot.mouseClick(window.login_button, 1)

    assert successes == ["shustinsky"], (
        "the window must emit login_succeeded with the local username"
    )
    assert window.status_label.text() == ""


# ---------------------------------------------------------------------------
# HUB-LOGIN — wrong password is rejected with the exact registry text
# ---------------------------------------------------------------------------


def test_s01_local_login_wrong_password_rejected(qtbot, isolated_settings) -> None:
    """A wrong password is rejected with the registry's exact error text.

    Registry HUB-LOGIN error condition: *"Неверные учётные данные или
    аккаунт временно заблокирован."* The window must show that text
    verbatim and must not report a successful login.
    """
    _create_account(isolated_settings, "the right password")

    window = _login_window(qtbot)
    successes: list[str] = []
    window.login_succeeded.connect(lambda username: successes.append(username))

    window.user_edit.setText("shustinsky")
    window.password_edit.setText("absolutely the wrong password")
    qtbot.mouseClick(window.login_button, 1)

    assert window.status_label.text() == REGISTRY_INVALID_CREDENTIALS_ERROR, (
        "the window must show the registry's exact error text "
        f"({REGISTRY_INVALID_CREDENTIALS_ERROR!r})"
    )
    assert successes == [], (
        "the window must not emit login_succeeded for a wrong password"
    )
