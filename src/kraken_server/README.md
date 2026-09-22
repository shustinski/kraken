# Kraken Server и Kraken Admin

`kraken-server` — HTTP/WebSocket-сервер общих проектов. `kraken-admin` —
консоль настройки этого сервера: база, первый администратор, роли восстановления,
токены агентов и проверка конфигурации. Права на проекты в интерфейсе выдаёт
Kraken Hub; админская консоль готовит сервер и учётные записи, без которых Hub
не к чему подключаться.

Оба входа ставятся из корня репозитория:

```powershell
uv sync --extra server --extra postgres --extra reports
```

Отладка: **Run and Debug** → **Kraken Server (development)**, **Kraken Server (local config)** или **Kraken Admin**.

## Роли проекта

Роль, от имени которой выполняется действие, уходит на сервер заголовком `X-Kraken-Role`. Сервер проверяет, что роль назначена, и что ей разрешено это действие. Отказ приходит как уведомление «Недостаточно прав».

| Роль | Что включает | Кого может назначать и отзывать |
|---|---|---|
| admin | все остальные | любую роль в любом проекте |
| maintainer | elementer, sewer, viewer и через них corrector | maintainer, sewer, corrector, elementer, viewer |
| elementer | corrector и viewer | никого, в том числе corrector |
| corrector | viewer | никого |
| sewer | viewer | никого |
| viewer | — | никого |

Создатель проекта становится его maintainer. Если maintainer отозван и другой назначающей роли у него не осталось, снимаются и роли, которые он выдал. Новую роль или действие можно добавить через `RoleCatalog.register_role` и `register_action` в `kraken_manager.domain.roles`: у действия `roles=None` означает «доступно всем ролям».

Дерево ролей в Desktop: участники проекта, прямоугольники и стрелки включения.

## kraken-server

```powershell
uv run kraken-server --help
```

| Параметр | По умолчанию | Пояснение |
|---|---|---|
| `--host` | `127.0.0.1` | Интерфейс. Используется, только если нет `--config` |
| `--port` | `8080` | Порт. Используется, только если нет `--config` |
| `--config` | `%PROGRAMDATA%\Kraken\Server\server.toml` (Linux: `/etc/kraken/server.toml`) | Если файл есть, host, port, TLS и состав сервисов берутся из него |
| `--development` | выключен | Память вместо PostgreSQL и вход любым Bearer-токеном. Для отладки API, не для общей сети |
| `--reload` | выключен | Перезапуск Uvicorn при изменении кода. В отладчике не используется: дочерний процесс отрывается от сессии |
| `--service` | выключен | Режим службы Windows Service Control Manager |

Без `--development` и без настроенного состава сервер не стартует. Локальный контур:

```powershell
uv run kraken-admin init
uv run kraken-server --config "$env:LOCALAPPDATA\Kraken\LocalServer\server.toml"
```

В режиме `--development` любой непустой `Authorization: Bearer …` считается
администратором. Проверка: `GET http://127.0.0.1:8080/api/v1/health`.

Шаблоны полей `server.toml`: [`config/templates`](../../config/templates/README.md).

## kraken-admin

```powershell
uv run kraken-admin --help
uv run kraken-admin КОМАНДА --help
```

Код возврата: `0` — успех, `2` — ошибка проверки или операции, `130` — отмена.

Пароли в аргументы не передаются. Неинтерактивный режим читает их из окружения
(`--non-interactive`).

| Команда | Что делает |
|---|---|
| `init` | Первичная локальная установка. Создаёт пользователя и базу PostgreSQL, выполняет миграции, пишет `server.toml` и секреты, создаёт первого `server_admin`. Псевдоним: `init-local-server` |
| `doctor` | Проверяет `server.toml`, соединение с PostgreSQL, каталог файлов и Blob Gateway. Данные не меняет |
| `project-create` | Входит на сервер и создаёт проект с матрицей `--width` и `--height` |
| `project-list` | Входит на сервер и печатает доступные проекты |
| `blob-benchmark` | Гоняет параллельные потоки через Blob Gateway и печатает скорость. Проекты и SQL не использует |
| `setup-server` | Подключает Kraken к уже существующей PostgreSQL, пишет конфиг и первого администратора |
| `bootstrap-admin` | Создаёт первого `server_admin` в уже подготовленном хранилище учётных записей |
| `recover-admin` | Возвращает роль `server_admin` существующему логину, включает учётную запись и отзывает её сессии. Новую учётную запись не создаёт |
| `install-service` | Регистрирует автозапуск `KrakenServer` в Windows |
| `bootstrap-local` | Создаёт локальную учётную запись Desktop в SQLite. К серверной PostgreSQL не относится |
| `create-agent-token` | Выпускает отзывной машинный токен агента. Секрет печатается один раз |
| `revoke-agent-token` | Запрещает дальнейшее использование токена |

### init

Мастер спрашивает адрес PostgreSQL, имя базы, служебного пользователя и первого
администратора Kraken. Пустой ввод принимает значения по умолчанию: PostgreSQL
`127.0.0.1:5432`, пользователь `postgres`, база `kraken_local`, логин Kraken
`admin`.

| Параметр | По умолчанию | Пояснение |
|---|---|---|
| `--config` | `%LOCALAPPDATA%\Kraken\LocalServer\server.toml` | Куда записать `server.toml` |
| `--blob-root` | каталог `blobs` рядом с конфигом | Файловое хранилище сервера |
| `--host`, `--port` | `127.0.0.1`, `8080` | Адрес Kraken Server |
| `--blob-gateway-host`, `--blob-gateway-port` | `127.0.0.1`, `8081` | Адрес файлового шлюза |
| `--blob-gateway-public-url` | для loopback вычисляется сам | Для сети обязателен HTTPS |
| `--blob-gateway-executable` | собранный `kraken-blob-gateway` в `blob_gateway/target/release` | Путь к бинарнику шлюза |
| `--tls-cert-file`, `--tls-key-file` | нет | PEM-сертификат и ключ для сетевого доступа |
| `--database-host`, `--database-port` | вопрос, иначе `127.0.0.1:5432` | Адрес PostgreSQL |
| `--postgres-admin` | вопрос, иначе `postgres` | Логин, который может создать базу и пользователя |
| `--database-name` | вопрос, иначе `kraken_local` | Имя новой базы |
| `--database-user` | вопрос, иначе `kraken_local_app` | Служебный логин приложения |
| `--username`, `--display-name` | вопрос, иначе `admin` / `Administrator` | Первый администратор Kraken |
| `--non-interactive` | выключен | Не задавать вопросов |
| `--install-service` | выключен | После настройки поставить и запустить службу Windows |
| `--server-executable` | текущий интерпретатор или `KrakenServer.exe` | Процесс для службы |

Неинтерактивные секреты: `KRAKEN_POSTGRES_ADMIN_PASSWORD`,
`KRAKEN_DATABASE_PASSWORD`, `KRAKEN_INITIAL_ADMIN_PASSWORD`.

Повторный `init` в тот же каталог не выполняется. Проверка: `doctor --config <путь>`.

### doctor и blob-benchmark

| Параметр | Пояснение |
|---|---|
| `--config` | `server.toml`, по умолчанию локальный путь `init` |
| `--json` | Одна JSON-строка для скрипта |

У `blob-benchmark` дополнительно `--clients` (30) и `--size-mib` (64).

### project-create и project-list

| Параметр | Пояснение |
|---|---|
| `--server` | Адрес Kraken Server, по умолчанию `http://127.0.0.1:8080` |
| `--username` | Логин Kraken, по умолчанию `admin` |
| `--non-interactive` | Пароль из `KRAKEN_ACCOUNT_PASSWORD` |
| `--json` | Машиночитаемый результат |
| `--name`, `--width`, `--height` | Только `project-create`, обязательны. Размеры больше нуля |
| `--orientation` | `y_down` (по умолчанию) или `y_up` |
| `--idempotency-key` | Повтор с тем же ключом не создаёт второй проект |

### setup-server

Для случая, когда базу создаёт внешняя система. Обычной локальной установке
нужен `init`, не эта команда.

| Параметр | По умолчанию | Пояснение |
|---|---|---|
| `--config` | `%PROGRAMDATA%\Kraken\Server\server.toml` | Куда записать конфиг |
| `--database-url` | скрытый вопрос | URL уже существующей PostgreSQL |
| `--blob-root` | `blobs` рядом с промышленным конфигом | Каталог объектов |
| `--host`, `--port` | `127.0.0.1`, `8080` | Адрес сервера |
| `--tls-cert-file`, `--tls-key-file` | нет | PEM TLS |
| `--username`, `--display-name` | вопрос | Первый администратор |
| `--install-service` | выключен | Поставить службу Windows |
| `--blob-gateway-host`, `--blob-gateway-port` | `127.0.0.1`, `8081` | Адрес шлюза |
| `--blob-gateway-public-url` | для loopback вычисляется сам | Вне loopback только HTTPS |
| `--blob-gateway-executable` | бинарник из `blob_gateway/target/release` | Путь к шлюзу |
| `--server-executable` | текущий интерпретатор или `KrakenServer.exe` | Процесс службы |

### Учётные записи и агенты

```powershell
uv run kraken-admin bootstrap-admin --config server.toml --username admin --display-name Administrator
uv run kraken-admin recover-admin --username admin
uv run kraken-admin bootstrap-local --username local --display-name Operator
uv run kraken-admin create-agent-token --config server.toml --name worker-1 --capability contour.run
uv run kraken-admin revoke-agent-token --config server.toml --token-id TOKEN_ID
```

| Параметр | Команды | Пояснение |
|---|---|---|
| `--database` | `bootstrap-admin` | Legacy SQLite учётных записей. Нельзя вместе с `--database-url` и `--config` |
| `--database-url` | `bootstrap-admin`, токены агента | URL PostgreSQL. Если пусто, берётся вопрос или конфиг |
| `--config` | те же и `recover-admin` | Существующий `server.toml` |
| `--username`, `--display-name` | `bootstrap-admin`, `bootstrap-local` | Логин и отображаемое имя. Для `recover-admin` нужен только существующий `--username` |
| `--data-dir` | `bootstrap-local` | Каталог Desktop, по умолчанию `%USERPROFILE%\.kraken` или `KRAKEN_DATA_DIR` |
| `--name` | `create-agent-token` | Понятное имя агента |
| `--capability` | `create-agent-token` | Разрешённая операция. Флаг повторяется |
| `--token-id` | `revoke-agent-token` | Идентификатор, который напечатал `create-agent-token` |

Пароль новой учётной записи читается из `KRAKEN_INITIAL_ADMIN_PASSWORD` или
спрашивается дважды.

### install-service

Только Windows. `--config` — готовый `server.toml`, `--server-executable` —
`KrakenServer.exe` или интерпретатор, `--no-start` — зарегистрировать службу и
не запускать её.
