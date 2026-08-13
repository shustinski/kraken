"""Packaged Kraken Server setup, recovery, and machine-token commands."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    default_local_config_path,
    run_migrations,
    write_config,
)
from .provisioning import (
    new_postgres_provisioning,
    provision_postgres,
    remove_provisioned_postgres,
)


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if (
            action.option_strings
            and action.default is not None
            and action.default is not argparse.SUPPRESS
            and "%(default)" not in help_text
        ):
            help_text += " (по умолчанию: %(default)s)"
        return help_text


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.pop("add_help", None)
        super().__init__(*args, add_help=False, **kwargs)
        self.add_argument("-h", "--help", action="help", help="Показать эту справку и выйти")
        self._positionals.title = "Позиционные аргументы"
        self._optionals.title = "Параметры"


def _secret(prompt: str, *, environment: str, non_interactive: bool = False) -> str:
    value = os.environ.get(environment, "")
    if value:
        return value
    if non_interactive:
        raise ValueError(f"Для неинтерактивного режима задайте переменную окружения {environment}")
    return getpass.getpass(prompt)


def _text_value(
    value: object,
    prompt: str,
    *,
    default: str,
    non_interactive: bool,
) -> str:
    supplied = str(value or "").strip()
    if supplied:
        return supplied
    if non_interactive:
        return default
    entered = input(f"{prompt} [{default}]: ").strip()
    return entered or default


def _integer_value(
    value: object,
    prompt: str,
    *,
    default: int,
    non_interactive: bool,
) -> int:
    supplied = str(value or "").strip()
    if not supplied and non_interactive:
        return default
    entered = supplied or input(f"{prompt} [{default}]: ").strip()
    try:
        return int(entered or default)
    except ValueError as exc:
        raise ValueError(f"{prompt}: требуется целое число") from exc


def _password(
    prompt: str,
    *,
    environment: str = "KRAKEN_INITIAL_ADMIN_PASSWORD",
    non_interactive: bool = False,
) -> str:
    from_environment = os.environ.get(environment, "")
    value = _secret(prompt, environment=environment, non_interactive=non_interactive)
    if from_environment:
        confirmation = value
    else:
        confirmation = getpass.getpass("Повторите пароль: ")
    if not value:
        raise ValueError("Пароль не может быть пустым")
    if value != confirmation:
        raise ValueError("Пароли не совпадают")
    return value


def _postgres_store(database_url: str) -> Any:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise SystemExit("The packaged PostgreSQL runtime is missing") from exc
    from kraken_manager.infrastructure.postgres import PostgresAccountStore

    return PostgresAccountStore(create_engine(database_url, pool_pre_ping=True), Argon2PasswordHasher())


def _database_url(args: argparse.Namespace) -> str:
    value = str(getattr(args, "database_url", "") or "").strip()
    if value:
        return value
    config = getattr(args, "config", None)
    if config is not None and Path(config).is_file():
        return ServerConfig.load(config).database_url
    return getpass.getpass("PostgreSQL URL: ").strip()


def _bootstrap(
    store: Any,
    username: str,
    display_name: str,
    *,
    password: str | None = None,
) -> str:
    if store.accounts_with_global_role("server_admin"):
        raise ValueError("Администратор Kraken Server уже существует")
    account = store.create_account(
        username,
        display_name,
        password or _password("New administrator password: "),
    )
    store.grant_global_role(account.account_id, "server_admin")
    return str(account.account_id)


def _default_blob_root() -> Path:
    return default_config_path().parent / "blobs"


def _default_local_blob_root() -> Path:
    return default_local_config_path().parent / "blobs"


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
            host=args.host,
            port=args.port,
            project_access_mode="acl",
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


def _doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, object]] = []

    def record(name: str, success: bool, detail: str) -> None:
        checks.append({"check": name, "ok": success, "detail": detail})

    try:
        configuration = ServerConfig.load(args.config)
        record("configuration", True, str(configuration.path))
    except Exception as exc:  # noqa: BLE001 - diagnostic must report every configuration failure
        record("configuration", False, str(exc))
        configuration = None
    if configuration is not None:
        try:
            from sqlalchemy import create_engine, inspect, text

            engine = create_engine(configuration.database_url, pool_pre_ping=True)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                tables = set(inspect(connection).get_table_names())
            required = {"accounts", "domain_events", "alembic_version"}
            missing = sorted(required - tables)
            record(
                "database",
                not missing,
                "подключение и схема готовы" if not missing else f"отсутствуют таблицы: {', '.join(missing)}",
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic must report every database failure
            record("database", False, str(exc))
        blob_ready = configuration.blob_root.is_dir()
        record(
            "blob_storage",
            blob_ready,
            str(configuration.blob_root) if blob_ready else "каталог не существует",
        )
    if args.json:
        print(json.dumps({"ok": all(bool(item["ok"]) for item in checks), "checks": checks}, ensure_ascii=False))
    else:
        for item in checks:
            marker = "OK" if item["ok"] else "ERROR"
            print(f"[{marker}] {item['check']}: {item['detail']}")
    return 0 if all(bool(item["ok"]) for item in checks) else 2


def _api_request(
    server: str,
    method: str,
    path: str,
    *,
    token: str = "",
    payload: dict[str, object] | None = None,
    idempotency_key: str | None = None,
) -> object:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    request = urllib.request.Request(
        server.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            detail = str(parsed.get("detail") or parsed.get("title") or detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(f"Kraken Server вернул HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось подключиться к Kraken Server {server}: {exc.reason}") from exc
    return json.loads(raw) if raw else {}


def _server_session(args: argparse.Namespace) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(args.server)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Адрес сервера должен быть корректным URL http:// или https://")
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Сетевые подключения к Kraken Server требуют HTTPS")
    password = _secret(
        "Пароль учётной записи Kraken: ",
        environment="KRAKEN_ACCOUNT_PASSWORD",
        non_interactive=args.non_interactive,
    )
    session = _api_request(
        args.server,
        "POST",
        "/api/v1/auth/sessions",
        payload={"username": args.username, "password": password},
    )
    if not isinstance(session, dict) or not session.get("access_token"):
        raise RuntimeError("Kraken Server вернул некорректный ответ авторизации")
    return args.server, str(session["access_token"])


def _create_project(args: argparse.Namespace) -> None:
    if args.width < 1 or args.height < 1:
        raise ValueError("Ширина и высота проекта должны быть больше нуля")
    server, token = _server_session(args)
    project = _api_request(
        server,
        "POST",
        "/api/v1/projects",
        token=token,
        idempotency_key=args.idempotency_key or str(uuid4()),
        payload={
            "name": args.name,
            "width": args.width,
            "height": args.height,
            "orientation": args.orientation,
        },
    )
    if args.json:
        print(json.dumps(project, ensure_ascii=False))
    elif isinstance(project, dict):
        print(f"Проект создан: {project.get('name', args.name)}")
        print(f"Идентификатор проекта: {project.get('project_id', '')}")


def _list_projects(args: argparse.Namespace) -> None:
    server, token = _server_session(args)
    response = _api_request(server, "GET", "/api/v1/projects", token=token)
    items = response.get("items", []) if isinstance(response, dict) else []
    if args.json:
        print(json.dumps(items, ensure_ascii=False))
        return
    if not items:
        print("Доступных проектов нет.")
        return
    for project in items:
        print(
            f"{project.get('project_id', '')}  {project.get('name', '')}  "
            f"{project.get('width', '?')}x{project.get('height', '?')}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="KrakenAdmin",
        formatter_class=_HelpFormatter,
        description="""Управление Kraken Server без ручной работы с SQL.

Для первого локального запуска используйте init. Команда сама
создаст пользователя и базу PostgreSQL, выполнит миграции, запишет защищённую
конфигурацию и создаст первого администратора Kraken.""",
        epilog="""Примеры:
  KrakenAdmin init
  KrakenAdmin doctor --config "%LOCALAPPDATA%\\Kraken\\LocalServer\\server.toml"
  KrakenAdmin project-create --name "Тест" --width 10 --height 10

Справка по конкретной команде:
  KrakenAdmin КОМАНДА --help""",
    )
    subcommands = parser.add_subparsers(
        dest="command",
        required=True,
        title="Команды",
        metavar="КОМАНДА",
    )

    def command(
        name: str,
        *,
        help: str,
        description: str,
        epilog: str = "",
        aliases: tuple[str, ...] = (),
    ) -> argparse.ArgumentParser:
        return subcommands.add_parser(
            name,
            aliases=aliases,
            help=help,
            description=description,
            epilog=epilog,
            formatter_class=_HelpFormatter,
        )

    local_setup = command(
        "init",
        aliases=("init-local-server",),
        help="Полностью подготовить локальный сервер и PostgreSQL для теста",
        description="""Одна команда для первичной локальной установки.

Требуется только установленный и запущенный PostgreSQL. SQL и psql не нужны.
Мастер последовательно запросит адрес, имя БД, логин и пароль БД, а затем
логин и пароль первого администратора Kraken.""",
        epilog="""Примеры:
  KrakenAdmin init
  KrakenAdmin init --database-name kraken_demo --port 9080

Автоматизация без запросов паролей:
  set KRAKEN_POSTGRES_ADMIN_PASSWORD=...
  set KRAKEN_DATABASE_PASSWORD=...
  set KRAKEN_INITIAL_ADMIN_PASSWORD=...
  KrakenAdmin init --non-interactive""",
    )
    local_setup.add_argument(
        "--config",
        type=Path,
        default=default_local_config_path(),
        help="Куда записать понятный server.toml и защищённый секрет БД",
    )
    local_setup.add_argument(
        "--blob-root",
        type=Path,
        default=_default_local_blob_root(),
        help="Каталог серверного файлового хранилища",
    )
    local_setup.add_argument("--host", default="127.0.0.1", help="Интерфейс Kraken Server")
    local_setup.add_argument("--port", type=int, default=8080, help="Порт Kraken Server")
    local_setup.add_argument(
        "--database-host",
        help="Адрес PostgreSQL; при пустом вводе 127.0.0.1",
    )
    local_setup.add_argument(
        "--database-port",
        type=int,
        help="Порт PostgreSQL; при пустом вводе 5432",
    )
    local_setup.add_argument(
        "--postgres-admin",
        help="Администратор PostgreSQL, имеющий право создать БД и пользователя; при пустом вводе postgres",
    )
    local_setup.add_argument(
        "--database-name",
        help="Имя новой БД; при пустом вводе kraken_local",
    )
    local_setup.add_argument(
        "--database-user",
        help="Имя нового служебного пользователя БД; при пустом вводе kraken_local_app",
    )
    local_setup.add_argument(
        "--username",
        help="Логин первого администратора Kraken; при пустом вводе admin",
    )
    local_setup.add_argument(
        "--display-name",
        help="Отображаемое имя администратора Kraken; при пустом вводе Administrator",
    )
    local_setup.add_argument(
        "--non-interactive",
        action="store_true",
        help="Не задавать вопросов; брать пароли из KRAKEN_POSTGRES_ADMIN_PASSWORD, KRAKEN_DATABASE_PASSWORD и KRAKEN_INITIAL_ADMIN_PASSWORD",
    )
    local_setup.add_argument(
        "--install-service",
        action="store_true",
        help="После настройки установить и запустить Windows-службу",
    )
    local_setup.add_argument(
        "--server-executable",
        type=Path,
        default=_default_server_executable(),
        help="Путь к KrakenServer.exe для Windows-службы",
    )

    doctor = command(
        "doctor",
        help="Проверить конфигурацию, PostgreSQL и файловое хранилище",
        description="Проверяет конфигурацию и соединение, не изменяя данные.",
        epilog="Пример:\n  KrakenAdmin doctor --config server.toml\n  KrakenAdmin doctor --json",
    )
    doctor.add_argument("--config", type=Path, default=default_local_config_path(), help="Файл server.toml")
    doctor.add_argument("--json", action="store_true", help="Машиночитаемый JSON для скриптов")

    def add_server_login(target: argparse.ArgumentParser) -> None:
        target.add_argument("--server", default="http://127.0.0.1:8080", help="Адрес Kraken Server")
        target.add_argument("--username", default="admin", help="Логин Kraken")
        target.add_argument(
            "--non-interactive",
            action="store_true",
            help="Взять пароль из KRAKEN_ACCOUNT_PASSWORD и не задавать вопросов",
        )
        target.add_argument("--json", action="store_true", help="Машиночитаемый JSON для скриптов")

    create_project = command(
        "project-create",
        help="Создать серверный проект из CLI или скрипта",
        description="Входит в Kraken Server и создаёт новый проект с указанной матрицей.",
        epilog="""Примеры:
  KrakenAdmin project-create --name "Тестовый проект" --width 20 --height 30
  KrakenAdmin project-create --name CI --width 5 --height 5 --non-interactive --json""",
    )
    add_server_login(create_project)
    create_project.add_argument("--name", required=True, help="Название проекта")
    create_project.add_argument("--width", type=int, required=True, help="Ширина матрицы, больше нуля")
    create_project.add_argument("--height", type=int, required=True, help="Высота матрицы, больше нуля")
    create_project.add_argument(
        "--orientation",
        choices=("y_down", "y_up"),
        default="y_down",
        help="Направление оси Y",
    )
    create_project.add_argument(
        "--idempotency-key",
        help="Стабильный ключ повтора; одинаковый ключ не создаёт дубликат",
    )

    list_projects = command(
        "project-list",
        help="Показать доступные серверные проекты",
        description="Входит в Kraken Server и выводит список доступных проектов.",
        epilog="Пример:\n  KrakenAdmin project-list\n  KrakenAdmin project-list --json --non-interactive",
    )
    add_server_login(list_projects)

    setup = command(
        "setup-server",
        help="Подключить Kraken к уже подготовленной PostgreSQL (расширенный режим)",
        description="Для инфраструктуры, где БД создаётся внешней системой. Обычной локальной установке эта команда не нужна.",
        epilog="Пример:\n  KrakenAdmin setup-server --database-url postgresql+psycopg://... --username admin",
    )
    setup.add_argument("--config", type=Path, default=default_config_path(), help="Файл server.toml")
    setup.add_argument("--database-url", help="URL существующей PostgreSQL; безопасно запрашивается, если не указан")
    setup.add_argument("--blob-root", type=Path, default=_default_blob_root(), help="Файловое хранилище сервера")
    setup.add_argument("--host", default="127.0.0.1", help="Интерфейс Kraken Server")
    setup.add_argument("--port", type=int, default=8080, help="Порт Kraken Server")
    setup.add_argument("--tls-cert-file", type=Path, help="PEM-сертификат TLS")
    setup.add_argument("--tls-key-file", type=Path, help="Закрытый PEM-ключ TLS")
    setup.add_argument("--username", help="Логин первого администратора Kraken")
    setup.add_argument("--display-name", help="Отображаемое имя администратора")
    setup.add_argument("--install-service", action="store_true", help="Установить и запустить Windows-службу")
    setup.add_argument(
        "--server-executable",
        type=Path,
        default=_default_server_executable(),
        help="Путь к KrakenServer.exe для Windows-службы",
    )

    bootstrap = command(
        "bootstrap-admin",
        help="Создать первого администратора в существующей конфигурации",
        description="Аварийная/расширенная команда для уже подготовленного хранилища учётных записей.",
        epilog="Пример:\n  KrakenAdmin bootstrap-admin --config server.toml --username admin --display-name Administrator",
    )
    database = bootstrap.add_mutually_exclusive_group(required=True)
    database.add_argument("--database", type=Path, help="Legacy SQLite account database")
    database.add_argument("--database-url", help="URL PostgreSQL")
    database.add_argument("--config", type=Path, help="Существующий server.toml")
    bootstrap.add_argument("--username", required=True, help="Логин")
    bootstrap.add_argument("--display-name", required=True, help="Отображаемое имя")

    recover = command(
        "recover-admin",
        help="Восстановить роль server_admin существующей учётной записи",
        description="Локальное аварийное восстановление. Новая учётная запись не создаётся.",
        epilog="Пример:\n  KrakenAdmin recover-admin --username admin",
    )
    recover.add_argument("--config", type=Path, default=default_config_path(), help="Существующий server.toml")
    recover.add_argument("--username", required=True, help="Существующий логин Kraken")

    install = command(
        "install-service",
        help="Установить Kraken Server как Windows-службу",
        description="Регистрирует автозапускаемую службу с указанной конфигурацией.",
        epilog='Пример:\n  KrakenAdmin install-service --config server.toml --server-executable "KrakenServer.exe"',
    )
    install.add_argument("--config", type=Path, default=default_config_path(), help="Существующий server.toml")
    install.add_argument(
        "--server-executable", type=Path, default=_default_server_executable(), help="Путь к KrakenServer.exe"
    )
    install.add_argument("--no-start", action="store_true", help="Только установить, не запускать")

    local = command(
        "bootstrap-local",
        help="Создать локальную учётную запись рабочей станции (legacy)",
        description="Не относится к серверной PostgreSQL и оставлена для совместимости.",
        epilog="Пример:\n  KrakenAdmin bootstrap-local --username local --display-name Operator",
    )
    local.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("KRAKEN_DATA_DIR", Path.home() / ".kraken")),
        help="Каталог локальных данных Desktop",
    )
    local.add_argument("--username", required=True, help="Логин")
    local.add_argument("--display-name", required=True, help="Отображаемое имя")

    create_agent = command(
        "create-agent-token",
        help="Создать отзывной машинный токен Kraken Agent",
        description="Выводит токен один раз; сохраните его в защищённом хранилище.",
        epilog="Пример:\n  KrakenAdmin create-agent-token --config server.toml --name worker-1 --capability contour.run",
    )
    create_agent.add_argument("--database-url", help="URL PostgreSQL")
    create_agent.add_argument("--config", type=Path, help="Существующий server.toml")
    create_agent.add_argument("--name", required=True, help="Понятное имя агента")
    create_agent.add_argument("--capability", action="append", required=True, help="Разрешённая операция; повторяется")
    revoke_agent = command(
        "revoke-agent-token",
        help="Отозвать машинный токен Kraken Agent",
        description="Немедленно запрещает дальнейшее использование токена.",
        epilog="Пример:\n  KrakenAdmin revoke-agent-token --config server.toml --token-id TOKEN_ID",
    )
    revoke_agent.add_argument("--database-url", help="URL PostgreSQL")
    revoke_agent.add_argument("--config", type=Path, help="Существующий server.toml")
    revoke_agent.add_argument("--token-id", required=True, help="Идентификатор токена")
    return parser


def _execute(args: argparse.Namespace) -> int:
    if args.command in {"init", "init-local-server"}:
        _initialize_local_server(args)
    elif args.command == "doctor":
        return _doctor(args)
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


def main() -> int:
    args = _parser().parse_args()
    try:
        return _execute(args)
    except KeyboardInterrupt:
        print("\nОперация отменена.", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        if os.environ.get("KRAKEN_DEBUG") == "1":
            raise
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
