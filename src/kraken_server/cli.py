"""Packaged Kraken Server setup, recovery, and machine-token commands."""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from kraken_manager.domain.identity import Principal
from kraken_manager.infrastructure.auth.identity_store import LocalIdentityAclStore
from kraken_manager.infrastructure.auth.local import (
    Argon2PasswordHasher,
    LocalAccountStore,
    ScryptPasswordHasher,
)

from .configuration import (
    ServerConfig,
    default_config_path,
    run_migrations,
    write_config,
)


def _password(prompt: str) -> str:
    value = getpass.getpass(prompt)
    confirmation = getpass.getpass("Repeat password: ")
    if not value:
        raise SystemExit("Password must not be empty")
    if value != confirmation:
        raise SystemExit("Passwords do not match")
    return value


def _postgres_store(database_url: str) -> Any:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise SystemExit("The packaged PostgreSQL runtime is missing") from exc
    from kraken_manager.infrastructure.postgres import PostgresAccountStore

    return PostgresAccountStore(create_engine(database_url, pool_pre_ping=True), Argon2PasswordHasher())


def _database_url(args: argparse.Namespace) -> str:
    if getattr(args, "config", None):
        return ServerConfig.load(args.config).database_url
    value = str(getattr(args, "database_url", "") or "").strip()
    if value:
        return value
    return getpass.getpass("PostgreSQL URL: ").strip()


def _bootstrap(store: Any, username: str, display_name: str) -> str:
    if store.accounts_with_global_role("server_admin"):
        raise SystemExit("A Server Administrator already exists")
    account = store.create_account(username, display_name, _password("New administrator password: "))
    store.grant_global_role(account.account_id, "server_admin")
    return str(account.account_id)


def _default_blob_root() -> Path:
    return default_config_path().parent / "blobs"


def _default_server_executable() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).with_name("KrakenServer.exe")
    return Path(sys.executable)


def _install_service(server_executable: Path, config: Path, *, start: bool) -> None:
    if os.name != "nt":
        raise SystemExit("Windows Service installation is available only on Windows")
    executable = server_executable.resolve(strict=True)
    configuration = config.resolve(strict=True)
    if executable == Path(sys.executable).resolve() and not getattr(sys, "frozen", False):
        command = f'"{executable}" -m kraken_server --service --config "{configuration}"'
    else:
        command = f'"{executable}" --service --config "{configuration}"'
    result = subprocess.run(
        [
            "sc.exe",
            "create",
            "KrakenServer",
            f"binPath= {command}",
            "start= auto",
            "DisplayName= Kraken Server",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 and "1073" not in (result.stdout + result.stderr):
        raise SystemExit((result.stdout + result.stderr).strip())
    subprocess.run(
        ["sc.exe", "description", "KrakenServer", "Kraken shared project server"],
        check=True,
    )
    subprocess.run(
        ["sc.exe", "failure", "KrakenServer", "reset= 86400", "actions= restart/5000/restart/15000/none/0"],
        check=True,
    )
    if start:
        subprocess.run(["sc.exe", "start", "KrakenServer"], check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kraken-admin")
    subcommands = parser.add_subparsers(dest="command", required=True)

    setup = subcommands.add_parser("setup-server", help="Initialize a packaged PostgreSQL server installation")
    setup.add_argument("--config", type=Path, default=default_config_path())
    setup.add_argument("--database-url", help="Prompted securely when omitted")
    setup.add_argument("--blob-root", type=Path, default=_default_blob_root())
    setup.add_argument("--host", default="127.0.0.1")
    setup.add_argument("--port", type=int, default=8080)
    setup.add_argument("--tls-cert-file", type=Path)
    setup.add_argument("--tls-key-file", type=Path)
    setup.add_argument("--username")
    setup.add_argument("--display-name")
    setup.add_argument("--install-service", action="store_true")
    setup.add_argument("--server-executable", type=Path, default=_default_server_executable())

    bootstrap = subcommands.add_parser("bootstrap-admin", help="Create the first server-local administrator")
    database = bootstrap.add_mutually_exclusive_group(required=True)
    database.add_argument("--database", type=Path, help="Legacy SQLite account database")
    database.add_argument("--database-url")
    database.add_argument("--config", type=Path)
    bootstrap.add_argument("--username", required=True)
    bootstrap.add_argument("--display-name", required=True)

    recover = subcommands.add_parser("recover-admin", help="Locally restore server_admin to an existing account")
    recover.add_argument("--config", type=Path, default=default_config_path())
    recover.add_argument("--username", required=True)

    install = subcommands.add_parser("install-service", help="Register packaged Kraken Server as a Windows service")
    install.add_argument("--config", type=Path, default=default_config_path())
    install.add_argument("--server-executable", type=Path, default=_default_server_executable())
    install.add_argument("--no-start", action="store_true")

    local = subcommands.add_parser("bootstrap-local", help="Create a workstation-local account")
    local.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("KRAKEN_DATA_DIR", Path.home() / ".kraken")),
    )
    local.add_argument("--username", required=True)
    local.add_argument("--display-name", required=True)

    create_agent = subcommands.add_parser("create-agent-token", help="Create a revocable Kraken Agent machine token")
    create_agent.add_argument("--database-url")
    create_agent.add_argument("--config", type=Path)
    create_agent.add_argument("--name", required=True)
    create_agent.add_argument("--capability", action="append", required=True)
    revoke_agent = subcommands.add_parser("revoke-agent-token", help="Revoke a Kraken Agent machine token")
    revoke_agent.add_argument("--database-url")
    revoke_agent.add_argument("--config", type=Path)
    revoke_agent.add_argument("--token-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "setup-server":
        database_url = _database_url(args)
        if not database_url:
            raise SystemExit("PostgreSQL URL is required")
        run_migrations(database_url)
        configuration = write_config(
            args.config,
            database_url=database_url,
            blob_root=args.blob_root,
            host=args.host,
            port=args.port,
            project_access_mode="acl",
            tls_cert_file=args.tls_cert_file,
            tls_key_file=args.tls_key_file,
        )
        username = str(args.username or input("Administrator username: ")).strip()
        display_name = str(args.display_name or input("Administrator display name: ")).strip()
        account_id = _bootstrap(_postgres_store(configuration.database_url), username, display_name)
        print(f"Kraken Server configured: {configuration.path}")
        print(f"Created first Server Administrator: {account_id}")
        if args.install_service:
            _install_service(args.server_executable, configuration.path, start=True)
            print("Kraken Server Windows service is installed and started")
    elif args.command == "bootstrap-admin":
        store = (
            LocalAccountStore(args.database, Argon2PasswordHasher())
            if args.database
            else _postgres_store(_database_url(args))
        )
        print(f"Created first Server Administrator: {_bootstrap(store, args.username, args.display_name)}")
    elif args.command == "recover-admin":
        store = _postgres_store(_database_url(args))
        account = store.get_by_username(args.username)
        if account is None:
            raise SystemExit("Account was not found; recovery never creates an unknown account")
        store.grant_global_role(account.account_id, "server_admin", actor_id=None)
        store.set_enabled(account.account_id, True, actor_id=None)
        store.revoke_all_sessions(account.account_id, actor_id=None)
        print(f"Restored Server Administrator: {account.account_id}")
    elif args.command == "install-service":
        _install_service(args.server_executable, args.config, start=not args.no_start)
        print("Kraken Server Windows service is installed")
    elif args.command == "bootstrap-local":
        password = _password("New local account password: ")
        args.data_dir.mkdir(parents=True, exist_ok=True)
        accounts = LocalAccountStore(args.data_dir / "accounts.sqlite3", ScryptPasswordHasher())
        account = accounts.create_account(args.username, args.display_name, password)
        identities = LocalIdentityAclStore(args.data_dir / "identity.sqlite3")
        identities.save(
            Principal.local(
                subject=account.username,
                display_name=account.display_name,
                principal_id=account.account_id,
            )
        )
        print(f"Created workstation-local account {account.account_id}")
    elif args.command in {"create-agent-token", "revoke-agent-token"}:
        from sqlalchemy import create_engine

        from .agent_auth import PostgresAgentTokenStore

        store = PostgresAgentTokenStore(create_engine(_database_url(args), pool_pre_ping=True))
        if args.command == "create-agent-token":
            identity, token = store.create(args.name, frozenset(args.capability))
            print(f"Agent token id: {identity.token_id}")
            print("Copy this token now; it cannot be shown again:")
            print(token)
        elif not store.revoke(args.token_id):
            raise SystemExit("Agent token was not found or was already revoked")
        else:
            print(f"Revoked agent token {args.token_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
