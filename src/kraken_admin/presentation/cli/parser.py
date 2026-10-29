"""cli: parser."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from kraken_server.configuration import (
    default_config_path,
    default_local_config_path,
)

from kraken_admin.presentation.cli.install_service import (
    _ArgumentParser,
    _HelpFormatter,
    _default_blob_gateway_executable,
    _default_blob_root,
    _default_local_blob_root,
    _default_server_executable,
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
  KrakenAdmin
  KrakenAdmin init
  KrakenAdmin doctor --config "%LOCALAPPDATA%\\Kraken\\LocalServer\\server.toml"

Без команды открывается локальное окно. Справка по команде:
  KrakenAdmin КОМАНДА --help""",
    )
    subcommands = parser.add_subparsers(
        dest="command",
        required=False,
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
    local_setup.add_argument(
        "--source-root",
        help="Каталог исходных типов img, ssc, prv, aux",
    )
    local_setup.add_argument(
        "--derived-root",
        help="Каталог производных типов dataset, result, vector",
    )
    local_setup.add_argument("--host", default="127.0.0.1", help="Интерфейс Kraken Server")
    local_setup.add_argument("--port", type=int, default=8080, help="Порт Kraken Server")
    local_setup.add_argument(
        "--blob-gateway-host", default="127.0.0.1", help="Интерфейс высокоскоростного файлового шлюза"
    )
    local_setup.add_argument("--blob-gateway-port", type=int, default=8081, help="Порт файлового шлюза")
    local_setup.add_argument(
        "--blob-gateway-public-url",
        help="Публичный HTTPS-адрес файлового шлюза; для локального теста вычисляется автоматически",
    )
    local_setup.add_argument(
        "--blob-gateway-executable",
        type=Path,
        default=_default_blob_gateway_executable(),
        help="Путь к KrakenBlobGateway.exe",
    )
    local_setup.add_argument("--tls-cert-file", type=Path, help="PEM-сертификат TLS для сетевого доступа")
    local_setup.add_argument("--tls-key-file", type=Path, help="Закрытый PEM-ключ TLS для сетевого доступа")
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

    blob_benchmark = command(
        "blob-benchmark",
        help="Проверить суммарную скорость Blob Gateway параллельными потоками",
        description="Запускает воспроизводимую потоковую загрузку без создания проектов и SQL-запросов.",
        epilog="Примеры:\n  KrakenAdmin blob-benchmark\n  KrakenAdmin blob-benchmark --clients 30 --size-mib 1024 --json",
    )
    blob_benchmark.add_argument("--config", type=Path, default=default_local_config_path(), help="Файл server.toml")
    blob_benchmark.add_argument("--clients", type=int, default=30, help="Количество одновременных загрузок")
    blob_benchmark.add_argument("--size-mib", type=int, default=64, help="Объём одного потока в МиБ")
    blob_benchmark.add_argument("--json", action="store_true", help="Машиночитаемый результат")

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
    setup.add_argument("--source-root", help="Каталог исходных типов img, ssc, prv, aux")
    setup.add_argument("--derived-root", help="Каталог производных типов dataset, result, vector")
    setup.add_argument("--host", default="127.0.0.1", help="Интерфейс Kraken Server")
    setup.add_argument("--port", type=int, default=8080, help="Порт Kraken Server")
    setup.add_argument("--tls-cert-file", type=Path, help="PEM-сертификат TLS")
    setup.add_argument("--tls-key-file", type=Path, help="Закрытый PEM-ключ TLS")
    setup.add_argument("--username", help="Логин первого администратора Kraken")
    setup.add_argument("--display-name", help="Отображаемое имя администратора")
    setup.add_argument("--install-service", action="store_true", help="Установить и запустить Windows-службу")
    setup.add_argument("--blob-gateway-host", default="127.0.0.1", help="Интерфейс файлового шлюза")
    setup.add_argument("--blob-gateway-port", type=int, default=8081, help="Порт файлового шлюза")
    setup.add_argument("--blob-gateway-public-url", help="Публичный HTTPS-адрес файлового шлюза")
    setup.add_argument(
        "--blob-gateway-executable",
        type=Path,
        default=_default_blob_gateway_executable(),
        help="Путь к KrakenBlobGateway.exe",
    )
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

    def local_admin_command(name: str, help_text: str, description: str, epilog: str):
        parser = command(name, help=help_text, description=description, epilog=epilog)
        parser.add_argument("--config", type=Path, default=default_local_config_path(), help="Локальный server.toml")
        parser.add_argument("--json", action="store_true", help="Машиночитаемый JSON")
        return parser

    account_list = local_admin_command(
        "account-list",
        "Показать учётные записи сервера",
        "Читает PostgreSQL на этой машине. По сети запрос не уходит.",
        "Пример:\n  KrakenAdmin account-list",
    )
    del account_list
    account_create = local_admin_command(
        "account-create",
        "Создать учётную запись сервера",
        "Пароль спрашивается локально и не передаётся по сети.",
        "Пример:\n  KrakenAdmin account-create --username operator --display-name Operator",
    )
    account_create.add_argument("--username", required=True)
    account_create.add_argument("--display-name", required=True)
    for name, help_text, description in (
        ("account-enable", "Включить учётную запись", "Локально включает существующий логин."),
        ("account-disable", "Отключить учётную запись", "Локально отключает логин и его сеансы. Последнего администратора отключить нельзя."),
        ("grant-admin", "Назначить локального администратора", "Даёт роль server_admin. Сама роль не открывает действий по HTTPS."),
        ("revoke-admin", "Снять роль локального администратора", "Последнего включённого администратора снять нельзя."),
        ("reset-password", "Задать новый пароль учётной записи", "Пароль спрашивается локально."),
        ("revoke-sessions", "Завершить все сеансы учётной записи", "Отзывает выданные этой учётной записи токены."),
    ):
        target = local_admin_command(name, help_text, description, f"Пример:\n  KrakenAdmin {name} --username operator")
        target.add_argument("--username", required=True)
    audit = local_admin_command(
        "audit",
        "Показать журнал локальных административных действий",
        "Журнал хранится в PostgreSQL и читается только с этой машины.",
        "Пример:\n  KrakenAdmin audit",
    )
    del audit
    revoke_role = local_admin_command(
        "revoke-maintainer",
        "Отозвать maintainer у участника проекта",
        "Локальный администратор может снять роль у любого сопровождающего, в том числе у последнего в проекте.",
        "Пример:\n  KrakenAdmin revoke-maintainer --project demo --username operator",
    )
    revoke_role.add_argument("--project", required=True, help="Идентификатор или имя проекта")
    revoke_role.add_argument("--username", required=True, help="Логин сопровождающего")

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
        epilog=(
            "Пример:\n  KrakenAdmin create-agent-token --config server.toml --name worker-1 "
            "--capability frames.vectorize.v1 --capability frames.binary-segment.v2"
        ),
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

    list_agent = command(
        "list-agent-tokens",
        help="Показать машинные токены Kraken Agent и их статус",
        description="Секреты токенов не печатаются.",
        epilog="Пример:\n  KrakenAdmin list-agent-tokens --config server.toml",
    )
    list_agent.add_argument("--database-url", help="URL PostgreSQL")
    list_agent.add_argument("--config", type=Path, help="Существующий server.toml")
    list_agent.add_argument("--json", action="store_true", help="Одна JSON-строка")

    migrate = command(
        "migrate",
        help="Применить миграции Alembic до головы",
        description="Не создаёт базу и не меняет учётные записи.",
        epilog="Пример:\n  KrakenAdmin migrate --config server.toml",
    )
    migrate.add_argument("--database-url", help="URL PostgreSQL")
    migrate.add_argument("--config", type=Path, help="Существующий server.toml")

    uninstall = command(
        "uninstall-service",
        help="Остановить и удалить Windows-службу Kraken Server",
        description="Служба удаляется из SCM. Конфигурация и база сохраняются.",
        epilog="Пример:\n  KrakenAdmin uninstall-service",
    )
    uninstall.add_argument('--server-executable', type=Path, default=_default_server_executable(),
                           help='Executable that must own the service before removal')
    command(
        "service-status",
        help="Показать состояние Windows-службы Kraken Server",
        description="Не меняет службу.",
        epilog="Пример:\n  KrakenAdmin service-status",
    )
    backup = command("backup-server", help="Согласованная копия PostgreSQL и серверных файлов",
        description="Копирует PostgreSQL, BlobStore и настроенные source/derived корни. На время копирования проектные операции приостановлены. Каталог копии должен быть новым.")
    backup.add_argument("--config", type=Path, required=True, help="Существующий server.toml")
    backup.add_argument("--destination", type=Path, required=True, help="Новый каталог архива")
    verify_backup = command("backup-verify", help="Проверить SHA и полноту серверного архива",
        description="Проверяет архив без подключения к БД и без изменения файлов.")
    verify_backup.add_argument("--backup", type=Path, required=True)
    restore = command("restore-server", help="Восстановить сервер в пустую БД и новые каталоги",
        description="Существующие данные не перезаписываются. URL новой БД берётся из переменной окружения. SQL и квитанция фиксируются одной транзакцией; --resume продолжает тот же архив/назначение после сбоя.")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True, help="Новый корень восстановления с подкаталогами хранилищ")
    restore.add_argument("--config-dir", type=Path, required=True, help="Отдельный новый каталог server.toml и защищённых секретов")
    restore.add_argument("--database-url-env", default="KRAKEN_RESTORE_DATABASE_URL", help="Переменная окружения с URL пустой БД")
    restore.add_argument("--resume", action="store_true")
    restore.add_argument("--host", help="Заменить интерфейс из архива")
    restore.add_argument("--port", type=int, help="Заменить порт из архива")
    restore.add_argument("--tls-cert-file", type=Path)
    restore.add_argument("--tls-key-file", type=Path)
    restore.add_argument("--gitlab-ca-file", type=Path)
    restore.add_argument("--blob-gateway-executable", type=Path)
    for item in (backup, restore):
        item.add_argument("--pg-bin-dir", type=Path, help="Каталог pg_dump/pg_restore/psql; по умолчанию PATH")
        item.add_argument("--timeout", type=float, default=3600, help="Таймаут отдельного инструмента PostgreSQL, секунд")
    return parser
