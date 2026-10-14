"""cli: initialize local server."""
from __future__ import annotations
import argparse
import sys
from kraken_server.configuration import (
    run_migrations,
    write_config,
)
from kraken_server.provisioning import (
    new_postgres_provisioning,
    provision_postgres,
    remove_provisioned_postgres,
)

from kraken_admin.cli_parts.install_service import (
    _blob_gateway_public_url,
    _bootstrap,
    _data_root,
    _install_service,
    _integer_value,
    _password,
    _postgres_store,
    _secret,
    _text_value,
)

def _initialize_local_server(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    secret_path = config_path.parent / "database-url.secret"
    if config_path.exists() or secret_path.exists():
        raise ValueError(
            f"Kraken уже настроен в {config_path.parent}. Для проверки выполните 'KrakenAdmin doctor --config <путь>'."
        )
    database_host = _text_value(
        args.database_host,
        "Адрес PostgreSQL",
        default="127.0.0.1",
        non_interactive=args.non_interactive,
    )
    database_port = _integer_value(
        args.database_port,
        "Порт PostgreSQL",
        default=5432,
        non_interactive=args.non_interactive,
    )
    postgres_admin = _text_value(
        args.postgres_admin,
        "Логин администратора PostgreSQL",
        default="postgres",
        non_interactive=args.non_interactive,
    )
    postgres_password = _secret(
        "Пароль администратора PostgreSQL: ",
        environment="KRAKEN_POSTGRES_ADMIN_PASSWORD",
        non_interactive=args.non_interactive,
    )
    database_name = _text_value(
        args.database_name,
        "Имя базы данных Kraken",
        default="kraken_local",
        non_interactive=args.non_interactive,
    )
    database_user = _text_value(
        args.database_user,
        "Логин служебного пользователя БД",
        default="kraken_local_app",
        non_interactive=args.non_interactive,
    )
    database_password = _password(
        "Пароль служебного пользователя БД: ",
        environment="KRAKEN_DATABASE_PASSWORD",
        non_interactive=args.non_interactive,
    )
    username = _text_value(
        args.username,
        "Логин администратора Kraken",
        default="admin",
        non_interactive=args.non_interactive,
    )
    display_name = _text_value(
        args.display_name,
        "Отображаемое имя администратора Kraken",
        default="Administrator",
        non_interactive=args.non_interactive,
    )
    request = new_postgres_provisioning(
        host=database_host,
        port=database_port,
        administrator=postgres_admin,
        administrator_password=postgres_password,
        database=database_name,
        application_user=database_user,
        application_password=database_password,
    )
    administrator_password = _password(
        "Новый пароль администратора Kraken: ",
        non_interactive=args.non_interactive,
    )

    print("[1/5] Создание отдельного пользователя и базы PostgreSQL...")
    provision_postgres(request)
    try:
        print("[2/5] Создание и обновление схемы Kraken...")
        run_migrations(request.database_url)
        print("[3/5] Запись защищённой конфигурации сервера...")
        configuration = write_config(
            config_path,
            database_url=request.database_url,
            blob_root=args.blob_root,
            source_root=_data_root(args.source_root, flag="--source-root", non_interactive=args.non_interactive),
            derived_root=_data_root(args.derived_root, flag="--derived-root", non_interactive=args.non_interactive),
            host=args.host,
            port=args.port,
            project_access_mode="acl",
            blob_gateway_public_url=_blob_gateway_public_url(args),
            blob_gateway_bind=f"{args.blob_gateway_host}:{args.blob_gateway_port}",
            blob_gateway_executable=args.blob_gateway_executable,
            tls_cert_file=args.tls_cert_file,
            tls_key_file=args.tls_key_file,
        )
        print("[4/5] Создание первого администратора Kraken...")
        account_id = _bootstrap(
            _postgres_store(configuration.database_url),
            username,
            display_name,
            password=administrator_password,
        )
    except BaseException:
        config_path.unlink(missing_ok=True)
        secret_path.unlink(missing_ok=True)
        (config_path.parent / "blob-gateway.secret").unlink(missing_ok=True)
        try:
            remove_provisioned_postgres(request)
        except Exception as cleanup_error:  # noqa: BLE001 - preserve original setup failure
            print(f"Предупреждение: автоматический откат PostgreSQL не выполнен: {cleanup_error}", file=sys.stderr)
        raise

    print("[5/5] Локальный Kraken Server полностью настроен.")
    print(f"Конфигурация: {configuration.path}")
    print(f"Адрес сервера: http://{configuration.host}:{configuration.port}")
    print(f"Администратор: {username} ({account_id})")
    print(f'Следующий шаг: KrakenServer.exe --config "{configuration.path}"')
    if args.install_service:
        _install_service(args.server_executable, configuration.path, start=True)
        print("Windows-служба KrakenServer установлена и запущена.")
