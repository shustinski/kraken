# Запуск Kraken

Команды выполняйте в PowerShell из корня репозитория. Выберите [режим](../components.md) перед запуском. Из исходников нужны подготовленные зависимости: для Desktop `uv sync --extra desktop --extra agent --extra registration --extra electrical`, для Server `uv sync --extra server --extra postgres --extra reports --extra packages --extra agent --extra registration --extra electrical`. В закрытой сети сначала подготовьте [офлайн-зависимости](../deployment/closed-network.md); не запускайте `uv sync` вслепую.

## Локальный Desktop без Server

Первый раз создайте локальную учетную запись, затем запускайте Hub без повторного bootstrap:

```powershell
uv run kraken-admin bootstrap-local --username admin --display-name Administrator
uv run kraken-hub
```

Этот аккаунт хранится в локальной SQLite. Он не создает роль PostgreSQL или администратора Server. Для завершения закройте Hub. [Команды Hub](../../src/kraken_hub/README.md).

## Постоянный Server и общий Desktop

При **первой** установке запустите PostgreSQL и интерактивный мастер:

```powershell
uv run kraken-admin init
```

Мастер создает служебную роль и БД PostgreSQL, выполняет миграции, создает первого `server_admin`, конфигурацию и защищенные файлы учетных данных. Не повторяйте `init` для готовой установки. Если сейчас не хотите хранить секреты, отложите эту команду; чтение документации ничего не создает. [Поля мастера и установка](../deployment/server.md).

При последующих запусках из исходников:

```powershell
uv run kraken-server --config "$env:LOCALAPPDATA\Kraken\LocalServer\server.toml"
```

Server запускает Blob Gateway как дочерний процесс, если шлюз включен. Оставьте окно Server открытым. В другом окне проверьте:

```powershell
uv run kraken-admin doctor
Invoke-RestMethod http://127.0.0.1:8080/api/v1/health
Invoke-RestMethod http://127.0.0.1:8081/health
```

Проверка порта 8081 применима только при включенном Gateway. Запустите Hub в отдельном окне и выберите вход через учетную запись Kraken:

```powershell
$env:KRAKEN_SERVER_URL = 'http://127.0.0.1:8080'
uv run kraken-hub
```

Для установленной поставки используйте `KrakenAdmin.exe init` один раз, затем `KrakenServer.exe --config "$env:LOCALAPPDATA\Kraken\LocalServer\server.toml"` либо установленную службу KrakenServer. Не запускайте службу и ручной экземпляр одновременно на тех же портах. Для удаленного доступа используйте TLS и настройку Gateway из [руководства развертывания](../deployment/server.md).

`kraken-server --development` нужен только для отладки API: данные временны, любой непустой Bearer-токен считается администратором. Не используйте его для общего проекта.

## Agent и плагины

Локальный Agent:

```powershell
uv run kraken-agent --plugins-config <путь-к-plugins.json>
```

Серверный Agent требует действующего машинного токена с нужными `--capability`, URL Server и реестр операций. Он читает токен из `KRAKEN_AGENT_TOKEN` окружения процесса:

```powershell
uv run kraken-agent --server-url http://127.0.0.1:8080 --plugins-config <путь-к-plugins.json> --independent
```

`--independent` позволяет ему продолжить работу после выхода Hub. Не записывайте токен в команду или vault. Плагины имеют отдельные окружения; например `cd plugins\contour; uv sync`. Подробности о реестре и режимах: [Agent README](../../src/kraken_agent/README.md).

## Beta-стенд

Это приемочный контур, отдельный от обычной установки:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-beta-stand.ps1
powershell -ExecutionPolicy Bypass -File scripts\stop-beta-stand.ps1
```

`-WithWorkerAgent` включайте только когда нужен фоновый worker: он может перехватить задание тестового one-shot worker. `-Restart` удаляет тестовую БД или схему и поднимает стенд заново. Стенд обычно использует `kraken_beta_test`; если у служебной роли нет `CREATEDB`, создает отдельную схему в `kraken_local`, не затрагивая `public`. Режим записан как `db_mode` в `.test_runtime\beta-stand\state.json`. [Подробный контракт и остановка](../quality/beta-stand.md).

## Изолированный тест backup/restore (S14)

Существующий сценарий создает новый временный кластер PostgreSQL с ролью `kraken_audit`, выбирает свободный loopback-порт, задает `KRAKEN_TEST_POSTGRES_URL` и `KRAKEN_TEST_PG_BIN` только на время прогона и останавливает кластер в `finally`:

```powershell
& docs\quality\reproduce-stabilization.ps1 -Block backup -PostgresBin 'C:\Program Files\PostgreSQL\18\bin'
```

Путь к бинарникам сверьте на своей машине (`Get-Command pg_dump` или каталог PostgreSQL). Значение `-PostgresBin` по умолчанию в скрипте указывает на PostgreSQL 16 и может быть устаревшим. Блок запускает `tests/test_server_backup.py` и `tests/test_server_backup_gateway.py`; JUnit и журналы остаются в `.test_runtime\quality-<uuid>`. Не направляйте эти тесты на рабочую БД: они создают и удаляют две базы `kraken_backup_test_<uuid>`. Наличие скрипта само по себе не меняет вердикт чекпоинта; нужны свежие успешные прогоны.

## Быстрая диагностика

| Симптом | Сначала проверить |
|---|---|
| Server не стартует | Есть ли `server.toml`; `--development` допустим только для отладки |
| Hub не видит Server | `KRAKEN_SERVER_URL`, `/api/v1/health`, выбранный способ входа |
| Gateway недоступен | `doctor`, `[blob_gateway]`, бинарник и порт из конфигурации |
| Agent не берет задания | URL, реестр операций, действительность токена и его capabilities |
| Backup-тесты пропущены | Обе тестовые переменные или изолированный сценарий выше |
| `kraken_audit` нет в рабочем PostgreSQL | Это ожидаемо: роль создается в другом временном кластере |

Проверка имен и прав: [роли и токены](roles.md).
