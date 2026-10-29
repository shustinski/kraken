"""cli: run local admin."""
from __future__ import annotations
import argparse
import json

from kraken_admin.presentation.cli.install_service import (
    _database_url,
    _password,
    _postgres_store,
)

def _run_local_admin(args: argparse.Namespace) -> None:
    from kraken_server.configuration import ServerConfig
    from kraken_admin.application.accounts import (
        create_account,
        reset_password,
        revoke_maintainer,
        revoke_sessions,
        set_account_enabled,
        set_server_admin,
    )
    from kraken_admin.infrastructure.composition import open_project_services

    store = _postgres_store(_database_url(args))
    command = args.command
    if command == "account-list":
        items = [
            {
                "account_id": account.account_id,
                "username": account.username,
                "display_name": account.display_name,
                "enabled": account.enabled,
                "system_roles": sorted(store.global_roles_for(account.account_id)),
                "created_at": str(account.created_at),
            }
            for account in store.list_accounts(include_disabled=True)
        ]
        if args.json:
            print(json.dumps({"items": items}, ensure_ascii=False))
            return
        for item in items:
            roles = ", ".join(item["system_roles"]) or "-"
            state = "active" if item["enabled"] else "disabled"
            print(f"{item['username']}\t{state}\t{roles}\t{item['display_name']}")
        return
    if command == "account-create":
        account_id = create_account(
            store,
            args.username,
            args.display_name,
            _password("Пароль новой учётной записи: "),
        )
        print(f"Создана учётная запись {account_id}")
        return
    if command == "account-enable":
        print(f"Учётная запись включена: {set_account_enabled(store, args.username, enabled=True)}")
        return
    if command == "account-disable":
        print(f"Учётная запись отключена: {set_account_enabled(store, args.username, enabled=False)}")
        return
    if command == "grant-admin":
        print(f"Назначен локальный администратор: {set_server_admin(store, args.username, grant=True)}")
        return
    if command == "revoke-admin":
        print(f"Роль администратора снята: {set_server_admin(store, args.username, grant=False)}")
        return
    if command == "reset-password":
        print(f"Пароль обновлён: {reset_password(store, args.username, _password('Новый пароль: '))}")
        return
    if command == "revoke-sessions":
        print(f"Сеансы завершены: {revoke_sessions(store, args.username)}")
        return
    if command == "audit":
        items = list(store.administration_audit(limit=500))
        if args.json:
            print(json.dumps({"items": items}, ensure_ascii=False, default=str))
            return
        for item in items:
            print(f"{item.get('recorded_at')}\t{item.get('action')}\t{item.get('target_account_id') or ''}")
        return
    if command == "revoke-maintainer":
        config = ServerConfig.load(args.config)
        services = open_project_services(config.database_url, config.blob_root)
        revoke_maintainer(services, store, project=args.project, username=args.username)
        print(f"Роль maintainer снята с {args.username} в проекте {args.project}")
        return
    raise SystemExit(f"Неизвестная локальная команда: {command}")
