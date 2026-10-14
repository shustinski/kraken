"""cli: execute."""
from __future__ import annotations
import argparse
import json
from kraken_manager.domain.identity import Principal
from kraken_manager.infrastructure.auth.identity_store import LocalIdentityAclStore
from kraken_manager.infrastructure.auth.local import (
    Argon2PasswordHasher,
    LocalAccountStore,
    ScryptPasswordHasher,
)
from kraken_server.configuration import (
    run_migrations,
    write_config,
)

from kraken_admin.cli_parts.api_request import (
    _create_project,
    _service_status,
    _uninstall_service,
)

from kraken_admin.cli_parts.doctor import _doctor

from kraken_admin.cli_parts.initialize_local_server import _initialize_local_server

from kraken_admin.cli_parts.install_service import (
    _blob_gateway_public_url,
    _bootstrap,
    _data_root,
    _database_url,
    _install_service,
    _password,
    _postgres_store,
)

from kraken_admin.cli_parts.list_projects import _list_projects

from kraken_admin.cli_parts.run_local_admin import _run_local_admin

def _execute(args: argparse.Namespace) -> int:
    if args.command in {"init", "init-local-server"}:
        _initialize_local_server(args)
    elif args.command == "doctor":
        return _doctor(args)
    elif args.command == "blob-benchmark":
        from kraken_server.blob_benchmark import run_blob_benchmark

        result = run_blob_benchmark(args.config, clients=args.clients, size_mib=args.size_mib)
        payload = result.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(
                f"Blob Gateway: {payload['gigabits_per_second']:.3f} Гбит/с, "
                f"{payload['clients']} потоков, {payload['elapsed_seconds']:.3f} с"
            )
        return 0
    elif args.command == "project-create":
        _create_project(args)
    elif args.command == "project-list":
        _list_projects(args)
    elif args.command == "setup-server":
        database_url = _database_url(args)
        if not database_url:
            raise SystemExit("PostgreSQL URL is required")
        run_migrations(database_url)
        configuration = write_config(
            args.config,
            database_url=database_url,
            blob_root=args.blob_root,
            source_root=_data_root(args.source_root, flag="--source-root", non_interactive=False),
            derived_root=_data_root(args.derived_root, flag="--derived-root", non_interactive=False),
            host=args.host,
            port=args.port,
            project_access_mode="acl",
            tls_cert_file=args.tls_cert_file,
            tls_key_file=args.tls_key_file,
            blob_gateway_public_url=_blob_gateway_public_url(args),
            blob_gateway_bind=f"{args.blob_gateway_host}:{args.blob_gateway_port}",
            blob_gateway_executable=args.blob_gateway_executable,
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
    elif args.command in {
        "account-list",
        "account-create",
        "account-enable",
        "account-disable",
        "grant-admin",
        "revoke-admin",
        "reset-password",
        "revoke-sessions",
        "audit",
        "revoke-maintainer",
    }:
        _run_local_admin(args)
    elif args.command == "install-service":
        _install_service(args.server_executable, args.config, start=not args.no_start)
        print("Kraken Server Windows service is installed")
    elif args.command == "uninstall-service":
        _uninstall_service()
        print("Kraken Server Windows service is removed")
    elif args.command == "service-status":
        print(_service_status())
    elif args.command == "migrate":
        from kraken_server.configuration import run_migrations

        run_migrations(_database_url(args))
        print("Migrations are at head")
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
    elif args.command in {"create-agent-token", "revoke-agent-token", "list-agent-tokens"}:
        from sqlalchemy import create_engine

        from kraken_server.agent_auth import PostgresAgentTokenStore

        store = PostgresAgentTokenStore(create_engine(_database_url(args), pool_pre_ping=True))
        if args.command == "create-agent-token":
            identity, token = store.create(args.name, frozenset(args.capability))
            print(f"Agent token id: {identity.token_id}")
            print("Copy this token now; it cannot be shown again:")
            print(token)
        elif args.command == "revoke-agent-token":
            if not store.revoke(args.token_id):
                raise SystemExit("Agent token was not found or was already revoked")
            print(f"Revoked agent token {args.token_id}")
        else:
            tokens = store.list_tokens()
            if args.json:
                print(json.dumps({"items": tokens}, ensure_ascii=False, default=str))
            elif not tokens:
                print("No agent tokens")
            else:
                for item in tokens:
                    status = "revoked" if item["revoked"] else "active"
                    capabilities = ", ".join(str(value) for value in item["capabilities"])
                    print(f"{item['token_id']}\t{status}\t{item['name']}\t{capabilities}")
    return 0
