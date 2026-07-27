"""Administrative bootstrap commands without hard-coded credentials."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from kraken_manager.domain.identity import Principal
from kraken_manager.infrastructure.auth.identity_store import LocalIdentityAclStore
from kraken_manager.infrastructure.auth.local import Argon2PasswordHasher, LocalAccountStore, ScryptPasswordHasher


def main() -> int:
    parser = argparse.ArgumentParser(prog="kraken-admin")
    subcommands = parser.add_subparsers(dest="command", required=True)
    bootstrap = subcommands.add_parser("bootstrap-admin", help="Create the first server-local administrator")
    database = bootstrap.add_mutually_exclusive_group(required=True)
    database.add_argument("--database", type=Path, help="Legacy standalone SQLite account database")
    database.add_argument("--database-url", help="PostgreSQL SQLAlchemy URL used by Kraken Server")
    bootstrap.add_argument("--username", required=True)
    bootstrap.add_argument("--display-name", required=True)
    local = subcommands.add_parser("bootstrap-local", help="Create a workstation-local account")
    local.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("KRAKEN_DATA_DIR", Path.home() / ".kraken")),
    )
    local.add_argument("--username", required=True)
    local.add_argument("--display-name", required=True)
    args = parser.parse_args()
    if args.command == "bootstrap-admin":
        password = getpass.getpass("New administrator password: ")
        confirmation = getpass.getpass("Repeat password: ")
        if password != confirmation:
            raise SystemExit("Passwords do not match")
        if args.database_url:
            try:
                from sqlalchemy import create_engine
            except ImportError as exc:
                raise SystemExit("Install Kraken with the 'postgres' extra") from exc
            from kraken_manager.infrastructure.postgres import PostgresAccountStore

            store = PostgresAccountStore(create_engine(args.database_url, pool_pre_ping=True), Argon2PasswordHasher())
        else:
            store = LocalAccountStore(args.database, Argon2PasswordHasher())
        if store.accounts_with_global_role("server_admin"):
            raise SystemExit("A Server Admin already exists; use the audited role-management command")
        account = store.create_account(args.username, args.display_name, password)
        store.grant_global_role(account.account_id, "server_admin")
        print(f"Created first Server Admin {account.account_id}")
    elif args.command == "bootstrap-local":
        password = getpass.getpass("New local account password: ")
        confirmation = getpass.getpass("Repeat password: ")
        if password != confirmation:
            raise SystemExit("Passwords do not match")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
